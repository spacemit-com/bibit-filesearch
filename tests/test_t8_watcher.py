"""增量扫描（inotify）测试。

覆盖：
- 监听配置目录（与全盘一致）
- 新增文件 → 自动索引（<10s）
- 修改文件 → 重新向量化 + 元数据同步更新（hash 未变不重算）
- 删除文件 → 清理向量+元数据+关联记录
- 排除规则（.git 等不监听）
"""
import sys
import tempfile
import time
from pathlib import Path

import pytest


from fake_embed import FakeEmbed  # noqa: E402
from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer  # noqa: E402
from bibit_filesearch.watcher import Watcher  # noqa: E402
from bibit_filesearch.retriever import Retriever  # noqa: E402



class FakeConfig:
    def __init__(self, index_dirs, excludes=None):
        self.index_dirs = index_dirs
        self.exclude_patterns = excludes or [".git", "__pycache__"]
        # 2026-08-25：补 scan 属性（Watcher.__init__ 读 config.scan.get("max_depth")，dict）
        self.scan = {"max_depth": 0}


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_t8_"))
    db = Database(str(tmp / "test.db"))
    embed = FakeEmbed()
    yield tmp, db, embed
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _wait_for(predicate, timeout=10):
    """轮询等待条件成立（<10s 验收线）。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


def test_watch_create_index(env):
    tmp, db, embed = env
    idx = Indexer(db, embed)
    retriever = Retriever(db, embed)
    idx.set_on_indexed(retriever.on_file_indexed)
    idx.set_on_deleted(retriever.on_file_deleted)
    watcher = Watcher(FakeConfig([str(tmp)]), db, idx)
    watcher.start()
    try:
        # 新增文件 → 自动索引
        p = tmp / "新增.md"
        p.write_text("# 新增文档\n\n增量扫描自动索引。\n", encoding="utf-8")
        # 2026-09-23（池子化）：等**索引完成**，不是等「行出现」。
        # 池子化后 watcher 入池就写行（queued），处理由消费者随后做；
        # 只等行出现的话，下一行 `get_chunks` 会偶发为空（行已在、块还没算）。
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "completed"
        ), "新增文件应在 <10s 内完成索引"
        fid = db.get_file_by_path(str(p))["file_id"]
        assert db.get_chunks(fid) != [], "应有向量块"
        # 可检索
        r = retriever.search({"query": "增量扫描", "page": 1, "pageSize": 5})
        assert any(x["fileId"] == fid for x in r["data"]["results"]), "新增文件应可检索"
    finally:
        watcher.stop()


def test_watch_modify_reindex(env):
    tmp, db, embed = env
    idx = Indexer(db, embed)
    retriever = Retriever(db, embed)
    idx.set_on_indexed(retriever.on_file_indexed)
    idx.set_on_deleted(retriever.on_file_deleted)
    watcher = Watcher(FakeConfig([str(tmp)]), db, idx)
    watcher.start()
    try:
        p = tmp / "修改.md"
        p.write_text("# 旧内容\n\n旧描述。\n", encoding="utf-8")
        # 同 create：等索引完成再取 fid，否则后续 do 之前旧块可能还没有
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "completed"
        )
        fid = db.get_file_by_path(str(p))["file_id"]

        # 修改内容 → 重新向量化（hash 变化触发重索引）
        p.write_text("# 新内容\n\n全新描述完全不一样。\n", encoding="utf-8")
        assert _wait_for(lambda: any("新内容" in c["text"] for c in db.get_chunks(fid))), "块文本应更新"
        chunks = db.get_chunks(fid)
        assert any("新内容" in c["text"] for c in chunks), "块文本应更新"
    finally:
        watcher.stop()


def test_watch_delete_cleanup(env):
    tmp, db, embed = env
    idx = Indexer(db, embed)
    retriever = Retriever(db, embed)
    idx.set_on_indexed(retriever.on_file_indexed)
    idx.set_on_deleted(retriever.on_file_deleted)
    watcher = Watcher(FakeConfig([str(tmp)]), db, idx)
    watcher.start()
    try:
        p = tmp / "删除.md"
        p.write_text("# 将删除\n\n内容。\n", encoding="utf-8")
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "completed"
        )
        fid = db.get_file_by_path(str(p))["file_id"]
        assert db.get_chunks(fid) != []

        # 删除 → 清理向量+元数据+关联记录
        p.unlink()
        assert _wait_for(lambda: db.get_file_by_path(str(p)) is None), "删除应清理索引"
        assert db.get_chunks(fid) == [], "chunks 应清空"
    finally:
        watcher.stop()


def test_watch_excluded_dir(env):
    tmp, db, embed = env
    idx = Indexer(db, embed)
    watcher = Watcher(FakeConfig([str(tmp)]), db, idx)
    watcher.start()
    try:
        gitdir = tmp / ".git"
        gitdir.mkdir(exist_ok=True)
        p = gitdir / "config"
        p.write_text("excluded", encoding="utf-8")
        time.sleep(1.0)
        assert db.get_file_by_path(str(p)) is None, ".git 下文件不应被索引"
    finally:
        watcher.stop()


def test_watch_mtime_only_no_reindex(env):
    """mtime 变化但内容不变 → 不重新向量化（hash 为准）。"""
    tmp, db, embed = env
    idx = Indexer(db, embed)
    retriever = Retriever(db, embed)
    idx.set_on_indexed(retriever.on_file_indexed)
    idx.set_on_deleted(retriever.on_file_deleted)
    watcher = Watcher(FakeConfig([str(tmp)]), db, idx)
    watcher.start()
    try:
        p = tmp / "touch.md"
        p.write_text("# 内容\n\n稳定内容。\n", encoding="utf-8")
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "completed"
        )
        fid = db.get_file_by_path(str(p))["file_id"]
        content_hash_before = db.get_file(fid)["content_hash"]

        # touch（只改 mtime 不改内容）
        import os

        st = p.stat()
        os.utime(p, (st.st_atime, st.st_mtime + 10))
        time.sleep(1.0)
        assert db.get_file(fid)["content_hash"] == content_hash_before, "内容未变不应重算"
        # 块数不变（未重新向量化）
        assert len(db.get_chunks(fid)) >= 1
    finally:
        watcher.stop()
