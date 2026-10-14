"""目录回收的前缀匹配必须是**逐字**的（2026-10-08，审查 R01）。

原实现用 SQL `LIKE` 做目录前缀匹配：

    SELECT path FROM files WHERE path LIKE '/kb/a_b/%'

`LIKE` 的模式串里 `_` 匹配任意单字符、`%` 匹配任意长度，参数化查询不会取消
这两个通配符的含义。于是目录名里含 `_` 或 `%` 时，删除该目录会连带命中兄弟
目录：

    /kb/a_b/f.txt  ← 目标目录
    /kb/axb/f.txt  ← 被 LIKE 一并命中（`_` 匹配 `x`）

修法：`db.files_under_dir()` 用 `substr(path, 1, len(prefix)) = prefix` 逐字比较；
`pool.cancel_tree` 与 `watcher._remove_dir_index` 都改走它。

两条路径都要守：pool 那条在目录被删/移出范围时走，watcher 那条是它的直接替代。
"""
from __future__ import annotations

import pathlib

import pytest

from bibit_filesearch.db import Database
from bibit_filesearch.metadata import extract_metadata


def _db(tmp_path) -> Database:
    return Database(str(tmp_path / "t.db"))


def _put(db: Database, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("内容 content\n", encoding="utf-8")
    db.upsert_file(extract_metadata(str(path)), "h", "", status="completed")


def _names(db: Database) -> set[str]:
    rows = db.conn.execute("SELECT path FROM files").fetchall()
    return {pathlib.Path(r["path"]).parent.name for r in rows}


# ── db.files_under_dir：判据本身 ───────────────────────────────────────────────

@pytest.mark.parametrize("dir_name, sibling", [
    ("a_b", "axb"),        # `_` 匹配任意单字符
    ("a%b", "azb"),        # `%` 匹配任意长度
    ("a_b", "aXb"),        # 大小写无关的干扰项（SQLite LIKE 对 ASCII 不区分大小写）
    ("plain", "axb"),      # 对照组：无通配符字符
])
def test_files_under_dir_matches_literally(tmp_path, dir_name, sibling):
    """逐字前缀：只返回目标目录下的文件，兄弟目录不得被命中。"""
    db = _db(tmp_path)
    _put(db, tmp_path / dir_name / "f.txt")
    _put(db, tmp_path / sibling / "f.txt")

    rows = db.files_under_dir(str(tmp_path / dir_name))
    hit = {pathlib.Path(r["path"]).parent.name for r in rows}

    assert hit == {dir_name}, (
        f"files_under_dir({dir_name!r}) 命中了 {sorted(hit)}——"
        f"兄弟目录 {sibling!r} 不应被包含（前缀必须逐字比较，不能用 SQL LIKE）"
    )


def test_files_under_dir_includes_nested_descendants(tmp_path):
    """子目录下的文件也算「该目录下」——前缀含结尾分隔符即可，无需递归。"""
    db = _db(tmp_path)
    _put(db, tmp_path / "kb" / "a.txt")
    _put(db, tmp_path / "kb" / "sub" / "deep" / "b.txt")
    _put(db, tmp_path / "kb_other" / "c.txt")

    got = {pathlib.Path(r["path"]).name for r in db.files_under_dir(str(tmp_path / "kb"))}
    assert got == {"a.txt", "b.txt"}


def test_files_under_dir_excludes_the_dir_itself(tmp_path):
    """目录自身不是文件；同名文件（不带结尾分隔符）不得被当成后代。"""
    db = _db(tmp_path)
    _put(db, tmp_path / "kb" / "a.txt")
    _put(db, tmp_path / "kb2" / "b.txt")

    rows = db.files_under_dir(str(tmp_path / "kb"))
    assert {pathlib.Path(r["path"]).name for r in rows} == {"a.txt"}


# ── pool.cancel_tree：用户可见的那条路径 ──────────────────────────────────────

class _Idx:
    """最小执行器替身：delete_file 真删，记录调用顺序。"""

    indexable_types = {"txt"}

    def __init__(self, db):
        self.db = db
        self.deleted: list[str] = []

    def delete_file(self, file_id: str):
        row = self.db.conn.execute(
            "SELECT path FROM files WHERE file_id=?", (file_id,)).fetchone()
        self.deleted.append(row["path"])
        self.db.delete_file(file_id)


@pytest.mark.parametrize("dir_name, sibling", [("a_b", "axb"), ("a%b", "azb")])
def test_cancel_tree_does_not_touch_sibling_dir(tmp_path, dir_name, sibling):
    """撤掉 a_b 只清 a_b 的索引；axb 必须原样保留（修复前两者全清）。"""
    from bibit_filesearch.pool import IndexPool

    db = _db(tmp_path)
    idx = _Idx(db)
    pool = IndexPool(db, idx)
    _put(db, tmp_path / dir_name / "f.txt")
    _put(db, tmp_path / sibling / "f.txt")

    n = pool.cancel_tree(str(tmp_path / dir_name))

    assert n == 1, f"应只撤掉 1 条，实际 {n} 条"
    assert _names(db) == {sibling}, (
        f"撤掉 {dir_name!r} 之后库里剩 {sorted(_names(db))}——"
        f"兄弟目录 {sibling!r} 的索引不应被删除"
    )


def test_cancel_tree_still_removes_nested_files(tmp_path):
    """退化保障：正常目录（含子目录）仍被整树撤掉。"""
    from bibit_filesearch.pool import IndexPool

    db = _db(tmp_path)
    pool = IndexPool(db, _Idx(db))
    _put(db, tmp_path / "kb" / "a.txt")
    _put(db, tmp_path / "kb" / "sub" / "b.txt")
    _put(db, tmp_path / "keep" / "c.txt")

    n = pool.cancel_tree(str(tmp_path / "kb"))

    assert n == 2
    assert _names(db) == {"keep"}


# ── watcher._remove_dir_index：同一条判据的另一处调用点 ────────────────────────

def test_watcher_remove_dir_index_does_not_touch_sibling_dir(tmp_path):
    """watcher 的目录清理与 pool.cancel_tree 同判据，不能各自漏一处。"""
    from bibit_filesearch.watcher import Watcher

    db = _db(tmp_path)
    idx = _Idx(db)
    watcher = Watcher.__new__(Watcher)      # 只测这一个方法，不启 inotify
    watcher.db = db
    watcher.indexer = idx
    _put(db, tmp_path / "a_b" / "f.txt")
    _put(db, tmp_path / "axb" / "f.txt")

    watcher._remove_dir_index(str(tmp_path / "a_b"))

    assert _names(db) == {"axb"}, (
        "watcher._remove_dir_index 仍会命中兄弟目录——两处调用点都要走逐字前缀"
    )
