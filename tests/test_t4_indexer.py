"""向量化 + 文件聚合测试（单文件端到端、hash 变更重算、删除清理）。

embed 用 FakeEmbed（2026-09-04：原连真实 :8082，Qwen3 切换后慢/超时，
且维度由服务决定——测试改用确定性伪 embed，dim=512 自洽，流程断言不变）。
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import pytest


from fake_embed import FakeEmbed  # noqa: E402
from bibit_filesearch.db import Database, _pack_vec, _unpack_vec
from bibit_filesearch.indexer import Indexer  # noqa: E402



@pytest.fixture(scope="module")
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_t4_"))
    db = Database(str(tmp / "test.db"))
    embed = FakeEmbed()
    idx = Indexer(db, embed)
    yield db, embed, idx, tmp
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _write(p: Path, content: str):
    p.write_text(content, encoding="utf-8")
    return p


def test_vector_blob_roundtrip():
    v = [0.1, -0.2, 3.14] * 10
    assert _unpack_vec(_pack_vec(v)) == pytest.approx(v)


def test_index_file_end_to_end(env):
    db, embed, idx, tmp = env
    p = _write(tmp / "周报.md", "# 项目周报\n\n本周完成需求评审，开发进度60%。\n")
    r = idx.index_file(str(p))
    assert r["status"] == "indexed"
    assert r["chunks"] >= 1
    f = db.get_file(r["file_id"])
    assert f["file_type"] == "md"
    # 块已入库且向量维度 512（get_chunks 已解码为 list）
    chunks = db.get_chunks(r["file_id"])
    assert len(chunks) == r["chunks"]
    assert len(chunks[0]["vector"]) == 512


def test_index_skipped_when_hash_unchanged(env):
    db, embed, idx, tmp = env
    p = _write(tmp / "stable.txt", "内容不变的文件。\n")
    r1 = idx.index_file(str(p))
    r2 = idx.index_file(str(p))
    assert r1["status"] == "indexed"
    assert r2["status"] == "skipped"  # 内容 hash 未变 → skip


def test_index_update_on_content_change(env):
    db, embed, idx, tmp = env
    p = _write(tmp / "changing.txt", "旧内容：描述甲。\n")
    r1 = idx.index_file(str(p))
    old_id = r1["file_id"]
    # 修改内容 → 重新向量化
    p.write_text("新内容：描述乙，完全不同。\n", encoding="utf-8")
    r2 = idx.index_file(str(p))
    assert r2["status"] == "updated"
    assert r2["file_id"] == old_id  # 同路径同 id
    # 旧块被替换（块数一致，文本为新内容）
    chunks = db.get_chunks(old_id)
    assert "描述乙" in chunks[0]["text"]


def test_index_unsupported_type_meta_only(env):
    db, embed, idx, tmp = env
    p = _write(tmp / "archive.zip", "PK\x03\x04dummy")
    r = idx.index_file(str(p))
    assert r["status"] == "unsupported"
    f = db.get_file_by_path(str(p))
    assert f is not None and f["file_type"] == "other"
    assert db.get_chunks(f["file_id"]) == []


def test_delete_file_removes_all(env):
    db, embed, idx, tmp = env
    p = _write(tmp / "todelete.md", "要删除的文件。\n")
    r = idx.index_file(str(p))
    fid = r["file_id"]
    assert db.get_chunks(fid) != []
    idx.delete_file(fid)
    assert db.get_file(fid) is None
    assert db.get_chunks(fid) == []


def test_index_code_file(env):
    db, embed, idx, tmp = env
    p = _write(tmp / "calc.py", 'def calculate_total_price(items):\n    """计算总价。"""\n    return sum(i["price"] for i in items)\n')
    r = idx.index_file(str(p))
    assert r["status"] == "indexed"
    chunks = db.get_chunks(r["file_id"])
    joined = "\n".join(c["text"] for c in chunks)
    assert "calculate_total_price" in joined
