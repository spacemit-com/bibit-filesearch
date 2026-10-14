"""扫描进度计数的完整性（2026-09-21）。

用户报：「扫描中为什么显示 118/118 文件，看上去像扫描完了」。根因不在遍历那行本身
（它显示的是 `scanPhase.walked/total`，遍历确实完成了、值是对的），而在于进度**没有
一个可用的「已处理 / 总数」基准**：`vectorizePhase` 只回 queued/completed/failed，
缺 `vectorizing` 与 `unsupported`，于是

  · `completed + queued + failed != 文件数`（在飞的那个不落任何一格）；
  · 界面只能拿遍历的数字当「进度」→ 遍历一结束就定格 118/118，而向量化才刚起步；
  · 有 unsupported（无文本/纯图）文件的库，进度永远到不了 100%。

本文件锁定补全后的契约：`total` 存在且等于各状态之和，`vectorizing` 与
`unsupported` 各自可见，且上游只用这一组数就能算出单调到 100% 的进度。
"""
import tempfile
from pathlib import Path

import pytest

from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer
from bibit_filesearch.scanner import ScanService


class _Embed:
    dim = 8

    def embed_one(self, text):
        return [1.0] * self.dim

    def embed_batch(self, texts):
        return [[1.0] * self.dim for _ in texts]

    def ping(self):
        return True

    def close(self):
        pass


class _Cfg:
    def __init__(self, dirs):
        self.index_dirs = dirs
        self.exclude_patterns = [".git"]

        class _Scan:
            def get(self, k, default=None):
                return {"io_throttle_ms": 0}.get(k, default)

        self.scan = _Scan()


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_progress_"))
    db = Database(str(tmp / "t.db"))
    work = tmp / "work"
    work.mkdir()
    yield tmp, db, work
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _mk_scanner(db, work):
    indexer = Indexer(db, _Embed())
    return ScanService(_Cfg([str(work)]), db, indexer)


def _ins(db, fid, name, status, chunks=0):
    """直接落一行（含状态），用于精确构造计数场景。"""
    db.conn.execute(
        "INSERT INTO files (file_id, path, name, file_type, content_hash, vectorize_status)"
        " VALUES (?, ?, ?, 'md', 'H', ?)",
        (fid, f"/kb/{name}", name, status),
    )
    for i in range(chunks):
        db.conn.execute(
            "INSERT INTO chunks (file_id, chunk_index, text, vector) VALUES (?, ?, ?, ?)",
            (fid, i, f"t{i}", b"\x00" * 4),
        )
    db.conn.commit()


# ---------------------------------------------------------------- 计数完整性
def test_vectorize_phase_reports_every_state(env):
    """★ 核心断言：五个状态各自可见，且 total == 各状态之和（不缺在飞的那个）。"""
    tmp, db, work = env
    for i, st in enumerate(["queued", "queued", "vectorizing", "completed",
                            "completed", "completed", "failed", "unsupported",
                            "unsupported"]):
        _ins(db, f"f{i}", f"f{i}.md", st)
    scanner = _mk_scanner(db, work)

    vp = scanner.progress()["vectorizePhase"]
    assert vp["queued"] == 2, vp
    assert vp["vectorizing"] == 1, f"在飞的文件必须可见（缺了它三数之和就差 1）：{vp}"
    assert vp["completed"] == 3, vp
    assert vp["failed"] == 1, vp
    assert vp["unsupported"] == 2, f"无文本文件必须可见（否则进度到不了 100%）：{vp}"
    assert vp["total"] == 9, f"total 应为 files 表行数：{vp}"
    assert (vp["queued"] + vp["vectorizing"] + vp["completed"]
            + vp["failed"] + vp["unsupported"]) == vp["total"], f"各状态之和必须闭合：{vp}"


def test_total_counts_rows_not_walked(env):
    """total 取自 DB 行数（不是本轮的 walked）：遍历后新增的文件也计入。"""
    tmp, db, work = env
    _ins(db, "a", "a.md", "completed")
    scanner = _mk_scanner(db, work)
    assert scanner.progress()["vectorizePhase"]["total"] == 1
    # 模拟 watcher 在遍历之后新增一个文件
    _ins(db, "b", "b.md", "queued")
    vp = scanner.progress()["vectorizePhase"]
    assert vp["total"] == 2, f"新增文件应计入 total：{vp}"


# ---------------------------------------------------------------- 上游算进度
def test_progress_reaches_exactly_100_with_unsupported(env):
    """★ 有 unsupported 文件时，上游按 done = completed+failed+unsupported 能算到 100%。

    这正是修复前做不到的事：unsupported 不在字段里 → 分子永远少几格 → 卡在 97%。
    """
    tmp, db, work = env
    _ins(db, "a", "a.md", "completed")
    _ins(db, "b", "b.md", "unsupported")   # 无文本 / 纯图 PDF
    _ins(db, "c", "c.md", "unsupported")
    scanner = _mk_scanner(db, work)

    vp = scanner.progress()["vectorizePhase"]
    assert vp["queued"] == 0 and vp["vectorizing"] == 0, vp
    done = vp["completed"] + vp["failed"] + vp["unsupported"]
    assert done == vp["total"], f"队列排空时 done 必须等于 total：{vp}"
    assert round(done / vp["total"] * 100) == 100


def test_vectorizing_file_is_counted_as_pending(env):
    """在飞的文件算「待处理」，不算「已处理」——进度条不会提前跳到 100%。"""
    tmp, db, work = env
    _ins(db, "a", "a.md", "completed")
    _ins(db, "b", "b.md", "vectorizing")
    scanner = _mk_scanner(db, work)

    vp = scanner.progress()["vectorizePhase"]
    done = vp["completed"] + vp["failed"] + vp["unsupported"]
    pending = vp["queued"] + vp["vectorizing"]
    assert done + pending == vp["total"], f"已处理 + 待处理必须闭合：{vp}"
    assert done / vp["total"] < 1.0, f"还有在飞的就不该显示满进度：{vp}"


# ---------------------------------------------------------------- 遍历阶段信号
def test_walk_phase_exposes_walked_and_total(env):
    """遍历阶段的两个数仍在（上游用它判断「遍历是否走完」）。"""
    tmp, db, work = env
    (work / "a.md").write_text("# 甲\n\n内容。\n", encoding="utf-8")
    (work / "b.md").write_text("# 乙\n\n内容。\n", encoding="utf-8")
    scanner = _mk_scanner(db, work)
    import time

    scanner.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30 and scanner.status != "done":
        time.sleep(0.05)
    sp = scanner.progress()["scanPhase"]
    assert sp["total"] == 2, sp
    assert sp["walked"] == 2, sp
    # 遍历走完的判据（上游用这个决定要不要显示「遍历中」）
    assert sp["walked"] >= sp["total"]
