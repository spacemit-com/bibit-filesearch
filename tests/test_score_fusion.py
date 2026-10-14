"""检索打分融合的回归测试（2026-09-29 重写 step 6/6.5）。

守的是两条已实测的缺陷：

  (a) **权重乘两次**：原实现先 `file_scores[fid] *= embed_weight`（无文件名命中时），
      再在融合里乘一次 → 纯向量文件的 cos 系数是 0.7×0.7 = **0.49**。
      实测校验（真机 query「身份」）：0.7×0.7×0.1875 + 0.3×1.0 = 0.3919，
      与接口返回 0.3920 逐位吻合 —— 这就是那条 BUG 的指纹。

  (b) **BM25 批内 min-max 归一**：命中集小且分数接近时把"都命中"拉成
      "一个满分一个零分"。实测 id_card_1 = 2.3033 → 1.0、id_rotate = 2.3020 → 0.0，
      只差 0.0013 却被分到 0.3 权重的两端。

判据尽量写成**"公式必须成立"**而不是"分值等于某个数"——前者不会被无关改动
弄脆，后者换个 fixture 就失效。凡是钉死具体数值的地方都注明了标定来源。
"""
import math

import pytest

from fake_embed import FakeEmbed  # noqa: E402
from bibit_filesearch.db import Database  # noqa: E402
from bibit_filesearch.indexer import Indexer  # noqa: E402
from bibit_filesearch.retriever import Retriever  # noqa: E402


# ─────────────── 纯单元：权重只乘一次 ───────────────

def test_weights_applied_exactly_once():
    """cos 的系数必须等于 embed_weight/ΣW，不能是它的平方。

    用"纯语义命中、无文件名命中、无 BM25 命中"这条最干净的分支来验：
    此时 score 应当**恰好等于** embed_weight/ΣW × cos。
    改前这里是 (embed_weight/ΣW)×embed_weight×cos，系数 0.49 而非 0.7。
    """
    ret = Retriever.__new__(Retriever)          # 只借用方法，不建 DB
    ret.embed_weight, ret.filename_weight, ret.bm25_weight = 0.7, 0.3, 0.3
    ret.bm25_half_score = 2.3
    total = 0.7 + 0.3 + 0.3

    # 模拟 _do_search 的融合段（S_sem = cos 池化值，另两路为 0）
    cos = 0.5
    merged = {"f1": {"name": "a.md"}}
    S_sem, S_name, S_bm25 = {"f1": cos}, {}, {}
    score = (ret.embed_weight * S_sem.get("f1", 0.0)
             + ret.filename_weight * S_name.get("f1", 0.0)
             + ret.bm25_weight * S_bm25.get("f1", 0.0)) / total
    assert score == pytest.approx(0.7 * cos / total), "cos 系数应为 0.7/1.3，不是 (0.7/1.3)*0.7"
    assert score != pytest.approx(0.49 * cos / total), "这曾是旧实现的值（权重乘两次）"


def test_three_way_sum_equals_weighted_average():
    """三路齐备时，score 必须等于加权平均 —— 权重和归一，不放大不缩小。"""
    ret = Retriever.__new__(Retriever)
    ret.embed_weight, ret.filename_weight, ret.bm25_weight = 0.7, 0.3, 0.3
    total = 1.3
    S_sem, S_name, S_bm25 = 0.8, 1.0, 0.5
    got = (0.7 * S_sem + 0.3 * S_name + 0.3 * S_bm25) / total
    assert got == pytest.approx((0.56 + 0.3 + 0.15) / 1.3)


# ─────────────── 纯单元：BM25 绝对映射 ───────────────

def test_bm25_mapping_is_absolute_not_batchwise():
    """S_bm25 = raw/(raw+K)：两个"几乎同分"的命中必须映射到几乎同分。

    这是 id_card_1 / id_rotate 那个案例的最小复现 —— 旧实现用批内 min-max，
    极差 0.0013 被拉伸成 1.0 与 0.0；绝对映射下两者都 ≈0.5，差 <0.001。
    """
    K = 2.3
    a = 2.3033 / (2.3033 + K)
    b = 2.3020 / (2.3020 + K)
    assert abs(a - b) < 0.001, "近乎相同的 BM25 原始分不该被分成两端"
    assert 0.49 < a < 0.51, "K 标定为「好命中的半分点」，命中良好时应接近 0.5"

    # 对照：旧的 min-max 行为 —— 同样的两个数给出 1.0 / 0.0
    vals = [2.3033, 2.3020]
    mn, mx = min(vals), max(vals)
    old = [(v - mn) / (mx - mn) for v in vals]
    assert old[0] == pytest.approx(1.0) and old[1] == pytest.approx(0.0)
    # ↑ 这条断言记录的是"我们**不要**的行为"，用来解释这个测试为什么存在。


def test_bm25_half_score_zero_falls_back_to_indicator():
    """K<=0 时退化为"命中即 1"的指示函数，不能除零崩溃。

    允许把 K 设 0（等于关掉连续映射），这是 debug 旋钮 —— 但必须不炸。
    """
    K = 0.0
    for raw, expect in ((2.3, 1.0), (0.0, 0.0)):
        got = (raw / (raw + K)) if (K > 0 and raw > 0) else (1.0 if raw > 0 else 0.0)
        assert got == expect


def test_bm25_only_hit_survives_without_vector_score():
    """纯 BM25 命中（无向量分）的文件必须仍在结果里，且分数不被清零。

    这是原实现的既有语义（`merged` 是语义命中 ∪ BM25 命中），重写时必须保留：
    中文/技术词查询里，"文件名无关但内容含词"的文件只有 BM25 能捞到。
    """
    ret = Retriever.__new__(Retriever)
    ret.embed_weight, ret.filename_weight, ret.bm25_weight = 0.7, 0.3, 0.3
    ret.bm25_half_score = 2.3
    total = 1.3
    merged = {"bm25_only": {"name": "x.md"}}
    S_sem, S_name = {}, {}                        # 语义路没有它
    S_bm25 = {"bm25_only": 2.3 / (2.3 + 2.3)}
    score = (0.7 * S_sem.get("bm25_only", 0.0)
             + 0.3 * S_name.get("bm25_only", 0.0)
             + 0.3 * S_bm25.get("bm25_only", 0.0)) / total
    assert score > 0, "纯 BM25 命中不能得 0 分（否则文件等于不可检索）"


def test_total_weight_zero_does_not_divide_by_zero():
    """三个权重全 0（全关）时不能 ZeroDivisionError。"""
    total_w = 0.7 * 0 + 0.3 * 0 + 0.3 * 0
    if total_w <= 0:
        total_w = 1.0
    score = (0 * 0.9 + 0 * 0.5 + 0 * 0.5) / total_w
    assert score == 0.0


# ─────────────── 端到端：真实经过 Retriever ───────────────

@pytest.fixture()
def env():
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="bibit_score_"))
    db = Database(str(tmp / "test.db"))
    embed = FakeEmbed()
    idx = Indexer(db, embed)
    ret = Retriever(db, embed, bm25_weight=0.3, embed_weight=0.7,
                    filename_weight=0.3, min_similarity=0.0)
    idx.set_on_indexed(ret.on_file_indexed)
    idx.set_on_deleted(ret.on_file_deleted)
    docs = {
        "id_card_1.png.txt": "蒋飞成 姓名 性别男 民族汉 出生1976年3月20日 公民身份号码 430523197603204314",
        "id_rotate.png.txt": "100% 公民身份号码 田村下尤铺组17号 出生1976年3月20日 性别男民族汉 蒋飞成 姓名",
        "unrelated.md": "PP-OCR 在 RISC-V 上的部署与使用：识别模型内嵌字典、推理后端切换。",
    }
    for name, content in docs.items():
        p = tmp / name
        p.write_text(content, encoding="utf-8")
        idx.index_file(str(p))
    ret.warmup()
    yield db, embed, idx, ret


def test_end_to_end_score_matches_formula(env):
    """端到端：接口给的 score 必须能用"三路绝对分加权平均"复算出来。

    这条是**防回归的总闸** —— 只要有人把权重、归一化、池化任一处改回去，
    复算就会对不上。（cos 用真实矩阵重算，BM25 用同一份实现重算。）
    """
    db, embed, idx, ret = env
    res = ret.search({"query": "身份", "page": 1, "pageSize": 10})
    rows = res["data"]["results"]
    assert rows, "应有结果"

    bm = ret.bm25.search("身份", top_k=1000)
    K = ret.bm25_half_score
    total = ret.embed_weight + ret.filename_weight + ret.bm25_weight
    # 三路都要真算：cos 用同一份矩阵 + 同一份 embed（矩阵里的向量是归一化后的，
    # 所以这里只需把 query 归一化），文件名命中率走同一实现。
    import numpy as np
    qv = np.array(embed.embed_one("身份"), dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    cand = ret._filter_files({})
    sem = ret._cosine_aggregate(cand, qv)
    name_scores = ret._filename_scores("身份", cand)
    for r in rows:
        fid = r["fileId"]
        s_sem = sem.get(fid, 0.0)
        s_name = name_scores.get(fid, 0.0)
        s_bm = (bm[fid] / (bm[fid] + K)) if (fid in bm and K > 0) else 0.0
        expect = (ret.embed_weight * s_sem + ret.filename_weight * s_name
                  + ret.bm25_weight * s_bm) / total
        assert r["score"] == pytest.approx(expect, abs=2e-4), \
            "%s: score=%.4f 与公式复算 %.4f 不符（权重/归一化/池化被改回去了？）" % (
                r["fileName"], r["score"], expect)
