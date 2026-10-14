"""A：极短块不向量化 —— 机制守卫 + 三个前置坑的回归测试。

背景（实测，见 indexer.index_file 注释）：embedding 对 1~2 token 的输入会退化——
任意两个 1-token 输入的余弦 0.93~0.99，而「1-token 查询 × 500 字符文档块」只有
0.26~0.37。于是内容只有 `1` 的块对**任何**查询都拿 0.90+，稳定挤掉全部真实文档
（24 个查询里 1~2 token 查询 12/12 由它占第一；换 'a'/'x'/'好' 结论相同）。

修法：写入侧按字符数跳过向量化（`scan.min_embed_chars`，默认 3），**文本保留**——
BM25 仍可按字面命中短标识符，语义是「不适合向量检索」而非「不可检索」。

本文件钉三件事（后两件是 A 的前置，不修就是负收益）：
  坑 1  on_file_indexed 的 `if not vectors: return` 早退在「移除旧行」之前 →
        文件被改短后 DB 无向量、内存矩阵旧向量还活着（内容改短反而搜得到旧内容）。
  坑 2  warmup 建 BM25 用 `vector IS NOT NULL` 过滤、增量路径不过滤 →
        「重启前搜得到、重启后搜不到」。
  坑 3  存量的极短块向量不会因 content_hash 幂等被清 → 水位线 rev 2 + requeue_short_chunks。
"""
import tempfile
from pathlib import Path

import pytest

from fake_embed import FakeEmbed
from bibit_filesearch.config import DEFAULTS
from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer, CHUNK_REVISION
from bibit_filesearch.retriever import Retriever


def _env(min_embed_chars: int = 3, bm25_weight: float = 0.0):
    tmp = Path(tempfile.mkdtemp(prefix="bibit_short_"))
    db = Database(str(tmp / "short.db"))
    embed = FakeEmbed()
    idx = Indexer(db, embed, chunk_size=500, chunk_overlap=0,
                  min_embed_chars=min_embed_chars)
    r = Retriever(db, embed, candidate_pool=50, top_k=50, bm25_weight=bm25_weight)
    idx.set_on_indexed(r.on_file_indexed)
    idx.set_on_deleted(r.on_file_deleted)
    return db, embed, idx, r, tmp


def _write(p: Path, content: str) -> Path:
    p.write_text(content, encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────
# 一、机制：极短块保留文本、不落向量
# ─────────────────────────────────────────────────────────────

def test_default_threshold_is_three_chars():
    """出厂默认必须与测试假定一致（否则下面几条测的就不是出厂行为了）。"""
    assert DEFAULTS["scan"]["min_embed_chars"] == 3


def test_short_chunk_keeps_text_without_vector():
    """内容只有 '1' 的块：文本入库、向量为 NULL。"""
    db, embed, idx, r, tmp = _env()
    p = _write(tmp / "a.txt", "1")
    out = idx.index_file(str(p))
    assert out["status"] == "indexed"
    assert out["chunks"] == 1
    assert out["embedded"] == 0, "极短块不应送去 embed"
    assert out["short_skipped"] == 1
    chunks = db.get_chunks(out["file_id"])
    assert len(chunks) == 1
    assert chunks[0]["text"].strip() == "1", "文本必须保留（BM25 路要用）"
    assert chunks[0]["vector"] is None, "极短块不应有向量"


def test_normal_chunk_still_embedded():
    """正常长度的块照常向量化（A 不该误伤）。"""
    db, embed, idx, r, tmp = _env()
    p = _write(tmp / "doc.md", "# 标题\n\n这是一段足够长的正文，描述 GNOME 会话管理与 mutter 的关系。\n")
    out = idx.index_file(str(p))
    chunks = db.get_chunks(out["file_id"])
    assert all(c["vector"] is not None for c in chunks)
    assert out["short_skipped"] == 0
    assert out["embedded"] == out["chunks"]


def test_mixed_file_only_short_chunk_lacks_vector():
    """同一文件里长短块混排：只有短块没向量，块的**顺序与数量**都不变。

    分块规则：chunk_size=500/overlap=0 → 501 字符切出 [0:500] + [500:501]，
    尾块 1 字符。极短块判定不该改动块的切分，只影响它有没有向量。
    """
    db, embed, idx, r, tmp = _env()
    body = "这是正文内容。" * 71 + "。。。" + "短"  # 497 + 3 + 1 = 501
    p = _write(tmp / "mix.md", body)
    out = idx.index_file(str(p))
    chunks = db.get_chunks(out["file_id"])
    assert len(chunks) == out["chunks"] == 2, "前提：该文本必须切出 2 块"
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0]["vector"] is not None, "长块照常有向量"
    assert chunks[1]["vector"] is None, "尾块只有 1 字符 → 不向量化"
    assert chunks[1]["text"].strip() == "短", "文本必须保留"
    assert out["short_skipped"] == 1
    assert out["embedded"] == 1


def test_threshold_zero_disables_the_rule():
    """min_embed_chars=0 = 关闭（与 bm25_weight/min_similarity 的 <=0 约定一致）。"""
    db, embed, idx, r, tmp = _env(min_embed_chars=0)
    p = _write(tmp / "a.txt", "1")
    out = idx.index_file(str(p))
    assert out["short_skipped"] == 0
    assert db.get_chunks(out["file_id"])[0]["vector"] is not None


# ─────────────────────────────────────────────────────────────
# 二、坑 1：内容改短后，旧向量必须从矩阵里消失
# ─────────────────────────────────────────────────────────────

def test_reindex_to_short_removes_old_vector_from_matrix():
    """长内容 → 改成 '1' 重索引 → 内存矩阵里必须**没有**该文件的任何行。

    钉的就是 `on_file_indexed` 的早退顺序：早退在「移除旧行」之前时，
    下面第一条断言拿到 1（残留），这就是「文件改短后反而搜得到旧内容」。
    """
    db, embed, idx, r, tmp = _env()
    p = _write(tmp / "shrink.md", "这是一段足够长的内容，描述内核报错的排查步骤与调用栈定位方法。\n")
    out1 = idx.index_file(str(p))
    fid = out1["file_id"]
    r._ensure_loaded()
    assert sum(1 for f in r._fids if f == fid) == 1, "前提：首次索引后矩阵有 1 行"

    p.write_text("1", encoding="utf-8")
    out2 = idx.index_file(str(p))
    assert out2["status"] == "updated"
    assert out2["embedded"] == 0

    r._ensure_loaded()
    assert sum(1 for f in r._fids if f == fid) == 0, \
        "内容改短后旧向量仍留在内存矩阵 = on_file_indexed 早退在移除之前（坑 1）"


def test_reindex_to_short_clears_vector_in_db():
    """同一场景下 DB 侧也必须是 NULL（与矩阵一致）。"""
    db, embed, idx, r, tmp = _env()
    p = _write(tmp / "shrink2.md", "足够长的内容，讲 Wayland 协议与合成器渲染链路。\n")
    out1 = idx.index_file(str(p))
    p.write_text("1", encoding="utf-8")
    out2 = idx.index_file(str(p))
    assert out2["file_id"] == out1["file_id"]
    chunks = db.get_chunks(out2["file_id"])
    assert len(chunks) == 1
    assert chunks[0]["text"].strip() == "1"
    assert chunks[0]["vector"] is None


def test_on_file_indexed_with_empty_vectors_clears_matrix():
    """直接钉回调本身：空向量列表也必须清掉该 file_id 的旧行。"""
    db, embed, idx, r, tmp = _env()
    p = _write(tmp / "shrink3.md", "足够长的内容，讲 GNOME 会话管理。\n")
    out = idx.index_file(str(p))
    fid = out["file_id"]
    r._ensure_loaded()
    assert sum(1 for f in r._fids if f == fid) == 1

    r.on_file_indexed(fid, [])
    r._ensure_loaded()
    assert sum(1 for f in r._fids if f == fid) == 0, "空向量回调必须清旧行"


# ─────────────────────────────────────────────────────────────
# 三、坑 2：两条 BM25 建索引路径口径一致
# ─────────────────────────────────────────────────────────────

def _bm25_files(r) -> set[str]:
    return set(r.bm25._chunk_files.values())


def test_warmup_indexes_short_chunks_into_bm25():
    """启动重建（warmup）必须收录极短块——否则该文件语义路被挡、稀疏路也没有。

    钉的是 warmup 的 `WHERE vector IS NOT NULL`：有那个过滤时，同一个短块在
    增量路径进得去、重启后进不去（「重启前搜得到、重启后搜不到」）。

    用 'ab'（2 字符 < 阈值 3）而不是 '1'：BM25 的 tokenizer 有意丢弃单字符 token
    （`if len(s) >= 2`，单字符噪声大），所以 '1' 本来就不在稀疏路里；短标识符类
    （'liuyingjie'、'ab'）才是「A 之后仍需靠 BM25 兜住」的那一档。
    """
    db, embed, idx, r, tmp = _env(bm25_weight=0.3)
    out = idx.index_file(str(_write(tmp / "a.txt", "ab")))
    fid = out["file_id"]

    # 模拟重启：新建 retriever，只走 warmup 全量重建
    r2 = Retriever(db, embed, candidate_pool=50, top_k=50, bm25_weight=0.3)
    r2.warmup()
    assert fid in _bm25_files(r2), "warmup 必须把无向量的短块也收进 BM25"


def test_bm25_membership_same_incremental_and_warmup():
    """同一份数据：增量路径与启动重建路径的 BM25 覆盖面必须一致。"""
    db, embed, idx, r, tmp = _env(bm25_weight=0.3)
    for name, body in (("a.txt", "ab"), ("doc.md", "这是一段足够长的正文，讲内核报错的排查。\n")):
        idx.index_file(str(_write(tmp / name, body)))

    incr = _bm25_files(r)
    r2 = Retriever(db, embed, candidate_pool=50, top_k=50, bm25_weight=0.3)
    r2.warmup()
    assert incr == _bm25_files(r2), \
        f"增量 {incr} 与 warmup {_bm25_files(r2)} 口径不一致（坑 2）"
    assert incr, "前提：至少有一个文件进了 BM25"


def test_short_identifier_still_found_by_bm25():
    """A 之后短标识符仍可按字面搜到（这是「留文本」这个决定的验收）。"""
    db, embed, idx, r, tmp = _env(bm25_weight=0.3)
    out = idx.index_file(str(_write(tmp / "a.txt", "liuyingjie")))  # 10 字符但向量也退化
    assert out["chunks"] == 1
    # 注：10 字符 >= 阈值 3，所以它仍有向量；这里钉的是 BM25 侧必然收录
    assert out["file_id"] in _bm25_files(r)
    res = r.search({"query": "liuyingjie", "page": 1, "pageSize": 10})
    assert res["data"]["total"] >= 1, "短标识符必须可检索"


# ─────────────────────────────────────────────────────────────
# 四、坑 3：存量极短块的旧向量靠水位线清掉
# ─────────────────────────────────────────────────────────────

def test_requeue_short_chunks_picks_only_vectorized_short_blocks():
    """只挑「还有向量的极短块」所在文件；已经无向量的短块不该再被重排。"""
    db, embed, idx, r, tmp = _env()
    idx.index_file(str(_write(tmp / "a.txt", "1")))          # 短块，A 之后无向量
    idx.index_file(str(_write(tmp / "doc.md", "足够长的正文内容，讲 GNOME 会话管理。\n")))
    assert db.requeue_short_chunks(min_chars=3) == 0, \
        "A 生效后库里没有「带向量的极短块」，不该重排任何文件"

    # 手工造出「存量」状态：给短块补一个向量（模拟 A 之前入的库）
    fid = db.get_file_by_path(str(tmp / "a.txt"))["file_id"]
    db.conn.execute("UPDATE chunks SET vector=? WHERE file_id=?", (b"\x00" * 4096, fid))
    db.conn.execute("UPDATE files SET vectorize_status='completed', content_hash='old' WHERE file_id=?", (fid,))
    db.conn.commit()

    assert db.requeue_short_chunks(min_chars=3) == 1
    row = db.get_file(fid)
    assert row["vectorize_status"] == "queued"
    assert row["content_hash"] == "", "必须同时清 hash，否则重扫会判 skipped（幂等挡路）"


def test_requeue_short_chunks_zero_threshold_is_noop():
    """阈值 0（关闭 A）= 不做任何重排。"""
    db, embed, idx, r, tmp = _env()
    idx.index_file(str(_write(tmp / "a.txt", "1")))
    assert db.requeue_short_chunks(min_chars=0) == 0


def test_chunk_revision_bumped_for_short_chunks():
    """水位线必须 > 1，否则老库的 chunk_revision 已是 1 → 这次重排机会被跳过。"""
    assert CHUNK_REVISION >= 2


def test_reconcile_chunk_revision_runs_once():
    """reconcile 只跑一次：第一次重排、第二次（水位线已记）返回 0。"""
    from bibit_filesearch.settings import reconcile_chunk_revision
    from bibit_filesearch.state import ScanState

    db, embed, idx, r, tmp = _env()
    idx.index_file(str(_write(tmp / "a.txt", "1")))
    fid = db.get_file_by_path(str(tmp / "a.txt"))["file_id"]
    db.conn.execute("UPDATE chunks SET vector=? WHERE file_id=?", (b"\x00" * 4096, fid))
    db.conn.commit()

    state = ScanState(db)
    assert reconcile_chunk_revision(db, state, short_min_chars=3) == 1
    assert reconcile_chunk_revision(db, state, short_min_chars=3) == 0, \
        "水位线已记，第二次不该再重排"
