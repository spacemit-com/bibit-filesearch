"""语义门槛（min_similarity）的机制守卫 + 标定守卫。

背景：`min_similarity = 0.35` 有两层问题。

**一、机制坏掉（2026-09-15 写本文件的测试时发现）**：门槛只从 `merged` 里删除，
但下游归一化与结果组装遍历的是 `file_scores` —— 于是**被剔除的文件照样出现在结果
里**。也就是说这个门槛自加入以来从未真正过滤过任何输出，与标定准不准无关。只有
「全部候选都被剔除 → merged 空 → 早退」这一条路径有效。已修（把 file_scores 收敛
到存活集合），`test_gate_drops_file_below_threshold` 就是钉这个的回归测试。

**二、标定过期**：即便机制修好，0.35 这个值也挡不住任何东西。实测 10 组查询（含 5
组纯噪声）无一被剔除 —— 因为 2026-09-09 清空 `query_instruction` 前缀后所有余弦被
整体抬高，而门槛数值与注释还停留在「带前缀时期」的测量（无关 0.14~0.28 / 相关
0.53+）。`test_default_threshold_separates_real_from_noise` 是钉标定的活体探针。

约定：`min_similarity <= 0` 表示**关闭门槛**（与 bm25_weight/filename_weight 一致）。
"""
import json
import tempfile
from pathlib import Path

import pytest

from fake_embed import FakeEmbed
from bibit_filesearch.config import DEFAULTS
from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer
from bibit_filesearch.retriever import Retriever

FIXTURE = Path(__file__).parent / "fixtures" / "semantic_gate_calibration.json"

# 9 个 token 的长查询：让「只共享 1 个 token」的文件余弦明显低于 0.35
# （FakeEmbed 的余弦 ≈ 共享 token 比例的一半量级，实测约 0.24）。
QUERY = "alpha beta gamma delta epsilon zeta eta theta iota"
STRONG = "alpha beta gamma delta epsilon zeta eta theta iota"  # 9/9 → cos ≈ 1.0
WEAK = "alpha zzz"  # 1/9 → cos ≈ 0.24


def _build(min_similarity: float, bm25_weight: float = 0.0):
    """造一个只含 strong/weak 两个文件的检索器，返回 (retriever, idx, db, tmp)。"""
    tmp = Path(tempfile.mkdtemp(prefix="gate_"))
    db = Database(str(tmp / "gate.db"))
    embed = FakeEmbed()
    idx = Indexer(db, embed)
    retriever = Retriever(
        db, embed, candidate_pool=50, top_k=50,
        embed_weight=0.7, filename_weight=0.3,
        min_similarity=min_similarity, bm25_weight=bm25_weight,
    )
    idx.set_on_indexed(retriever.on_file_indexed)
    idx.set_on_deleted(retriever.on_file_deleted)
    (tmp / "strong.md").write_text(STRONG, encoding="utf-8")
    (tmp / "weak.txt").write_text(WEAK, encoding="utf-8")
    for name in ("strong.md", "weak.txt"):
        idx.index_file(str(tmp / name))
    return retriever, idx, db, tmp


def _cosines(retriever) -> dict[str, float]:
    """{文件名: 该文件的 max 池化余弦}（门槛比对的就是这个量）。"""
    import numpy as np

    q = np.array(retriever.embed.embed_one(QUERY), dtype=np.float32)
    scores = retriever._cosine_aggregate(retriever._filter_files({}), q)
    return {retriever.db.conn.execute(
        "SELECT name FROM files WHERE file_id=?", (fid,)).fetchone()["name"]: s
        for fid, s in scores.items()}


def _names(result: dict) -> set[str]:
    return {r["fileName"] for r in result["data"]["results"]}


# ─────────────────────────────────────────────────────────────
# 一、机制守卫：门槛真的会剔除
# ─────────────────────────────────────────────────────────────

def test_premise_weak_file_cosine_is_below_default_threshold():
    """前提校验：weak 的余弦确实低于门槛，strong 高于门槛——否则下面几条无意义。"""
    r, _idx, db, tmp = _build(min_similarity=DEFAULTS["search"]["min_similarity"])
    cos = _cosines(r)
    thr = r.min_similarity
    assert cos["weak.txt"] < thr <= cos["strong.md"], (
        f"测试前提不成立：weak={cos['weak.txt']:.4f} strong={cos['strong.md']:.4f} "
        f"threshold={thr}——FakeEmbed 的相似度尺度变了，请调整 QUERY/WEAK 词数"
    )


def test_gate_drops_file_below_threshold():
    """低于门槛的文件不出现在结果里（这就是「门槛生效」的定义）。"""
    r, _idx, db, tmp = _build(min_similarity=DEFAULTS["search"]["min_similarity"])
    assert "strong.md" in _names(r.search({"query": QUERY, "page": 1, "pageSize": 50}))
    assert "weak.txt" not in _names(r.search({"query": QUERY, "page": 1, "pageSize": 50})), \
        "余弦低于 min_similarity 的文件必须被剔除"


def test_gate_disabled_at_zero_returns_weak_file():
    """门槛设 0 时同一文件被返回——反证上一条是门槛的功劳，而非别的机制。"""
    r, _idx, db, tmp = _build(min_similarity=0.0)
    assert "weak.txt" in _names(r.search({"query": QUERY, "page": 1, "pageSize": 50})), \
        "min_similarity=0 表示关闭门槛，weak 文件应出现"


def test_gate_with_impossible_threshold_drops_everything():
    """门槛设成不可能达到的值（无 BM25 路）→ 结果为空，证明门槛作用于最终输出。"""
    r, _idx, db, tmp = _build(min_similarity=2.0, bm25_weight=0.0)
    out = r.search({"query": QUERY, "page": 1, "pageSize": 50})
    assert out["data"]["total"] == 0, "门槛高于任何余弦时不应有结果"


def test_bm25_hit_bypasses_gate():
    """纯 BM25 命中的文件不受门槛约束（它们没有 embed 分）——钉住这个已声明例外。"""
    r, _idx, db, tmp = _build(min_similarity=2.0, bm25_weight=0.3)
    out = r.search({"query": QUERY, "page": 1, "pageSize": 50})
    assert out["data"]["total"] > 0, \
        "BM25 命中的文件应绕过语义门槛（源码注释声明的例外），否则文档检索体验劣化"


def test_no_overlap_query_returns_nothing():
    """与全库零重叠的查询：既无向量分也非 BM25 命中 → 空结果。"""
    r, _idx, db, tmp = _build(min_similarity=DEFAULTS["search"]["min_similarity"], bm25_weight=0.3)
    out = r.search({"query": "qqqq wwww eeee", "page": 1, "pageSize": 50})
    assert out["data"]["total"] == 0, "完全无重叠的噪声查询应返回空"


# ─────────────────────────────────────────────────────────────
# 二、标定守卫：门槛必须能分开真实与噪声（当前不成立）
# ─────────────────────────────────────────────────────────────

def test_calibration_fixture_matches_shipped_embedding_config():
    """标定数据必须仍对应当前出厂的 embedding 配置，否则测的是别的东西。

    标定是**相对某套 embedding 配置**测出来的：换模型/换维度（更别说换 pooling）后
    数值尺度全变。这条守卫在配置漂移时报警，提示重新测量并更新 fixture——
    否则上面那条 xfail 探针会拿过期数据下结论。
    """
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    embed_cfg = DEFAULTS["models"]["embed"]
    assert data["config_model"] == embed_cfg["model"], (
        f"标定数据是在 model={data['config_model']} 下测的，出厂已改为 {embed_cfg['model']}"
        f"——请重新测量并更新 {FIXTURE.name}"
    )
    assert data["dim"] == embed_cfg["dim"], (
        f"标定数据是在 dim={data['dim']} 下测的，出厂已改为 {embed_cfg['dim']}"
        f"——请重新测量并更新 {FIXTURE.name}"
    )
    # pooling 不是配置项（是 embed 服务的启动参数），只能靠人工核对：
    # 修好 P1（改 --pooling last）后必须重测，并把这里的注释与数据一起更新。
    assert data["pooling"] == "mean"


@pytest.mark.xfail(
    strict=True,
    reason="已知缺陷（2026-09-15 实测）：门槛 0.35 落在噪声区间内部，两组重叠 → 门槛从不触发。"
           "本测试是活体探针：修好 embedding 配置或重新标定后它会 XPASS(strict) 而让套件变红，"
           "提示同步更新 fixtures/semantic_gate_calibration.json 与出厂默认值。",
)
def test_default_threshold_separates_real_from_noise():
    """出厂 min_similarity 必须落在「噪声最高分」与「真实最低分」之间。

    门槛的作用是「剔除不相关的结果」，所以它必须同时满足：
      · 噪声查询的最高分 < 门槛   （否则噪声照样返回，门槛形同虚设）
      · 门槛 <= 真实查询的最低分  （否则会误杀相关内容）
    数据来自 tests/fixtures/semantic_gate_calibration.json（真实服务实测）。
    """
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    real = data["real"]
    noise = data["noise"]
    threshold = DEFAULTS["search"]["min_similarity"]

    noise_top = max(noise.values())
    real_floor = min(real.values())

    assert noise_top < threshold, (
        f"门槛 {threshold} 挡不住噪声：噪声最高分 {noise_top}"
        f"（{max(noise, key=noise.get)}）≥ 门槛 → 无关文件照样返回"
    )
    assert threshold <= real_floor, (
        f"门槛 {threshold} 会误杀真实结果：真实最低分 {real_floor}"
        f"（{min(real, key=real.get)}）< 门槛"
    )
