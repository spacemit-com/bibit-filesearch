"""取消标记必须在**异常出口**也被消费掉（2026-10-08，审查 R08）。

## 现象

`IndexWorker._run_one` 原先只在成功路径调 `pool.commit(path)`。异常路径直接
return，于是 `_cancelled` / `_dirty` 标记留在集合里没人消费。

## 为什么后果比「少删一次」更糟

标记是**跨轮**存活的：取消发生在解析期间 → 本次索引失败 → 标记仍在 → 同路径之后
重新出现（用户又把文件拷回来）并**成功**索引 → `commit()` 读到上一轮的陈旧标记，
把刚写好的行和块删掉。实测：重试成功后 `db.get_file_by_path()` 返回 None。

## 判据

异常出口先做池收尾：被取消 → 结果丢弃（删行）；内容又变了 → 重新排队。
两者都不是失败路径该保留的状态。
"""
from __future__ import annotations

import pathlib

from bibit_filesearch.db import Database
from bibit_filesearch.metadata import extract_metadata
from bibit_filesearch.pool import IndexPool, IndexWorker


class _Indexer:
    """替身执行器：可切换「失败 / 成功」，成功时真写行。"""

    indexable_types = {"txt"}

    def __init__(self, db: Database):
        self.db = db
        self.fail = False
        self.content_hash = "new-hash"
        self.max_chunks_per_file = 64

    def delete_file(self, file_id: str):
        self.db.delete_file(file_id)

    def index_file(self, path: str):
        if self.fail:
            raise RuntimeError("模拟解析失败")
        self.db.upsert_file(extract_metadata(path), self.content_hash, "",
                            status="completed")
        return {"status": "updated", "chunks": 1}


def _setup(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("内容\n", encoding="utf-8")
    db = Database(str(tmp_path / "t.db"))
    idx = _Indexer(db)
    pool = IndexPool(db, idx)
    worker = IndexWorker(pool, idx)
    db.upsert_file(extract_metadata(str(p)), "old-hash", "", status="queued")
    return p, db, idx, pool, worker


def test_failure_after_cancel_does_not_leave_a_stale_marker(tmp_path):
    """取消 + 本次失败 → 标记必须被消费（改前残留 True）。"""
    p, db, idx, pool, worker = _setup(tmp_path)

    row = pool.take()
    assert pool.cancel(str(p)) == "deferred", "在算中的取消应走 deferred"
    idx.fail = True
    worker._run_one(row)

    assert str(p) not in pool._cancelled, (
        "异常出口未消费取消标记——它会跨轮存活，误删下次成功索引的结果"
    )


def test_retry_after_cancelled_failure_survives(tmp_path):
    """取消 + 失败 + 同路径重新出现并成功 → 结果必须留在库里。

    这是用户能看见的那一面：取消标记泄漏会把刚索引好的文件删掉。
    """
    p, db, idx, pool, worker = _setup(tmp_path)

    row = pool.take()
    pool.cancel(str(p))
    idx.fail = True
    worker._run_one(row)                     # 第一次：取消 + 失败
    assert str(p) not in pool._cancelled

    # 文件重新出现（用户拷回来），重新入池并这次成功
    idx.fail = False
    pool.offer(str(p))
    row2 = pool.take()
    if row2 is not None:
        worker._run_one(row2)

    rec = db.get_file_by_path(str(p))
    assert rec is not None, "同路径重试成功后记录被删空了——取消标记泄漏所致"
    assert rec["vectorize_status"] == "completed"


def test_cancelled_failure_discards_the_row(tmp_path):
    """取消是明确的「文件已不在范围内」：结果丢弃，不留痕。"""
    p, db, idx, pool, worker = _setup(tmp_path)
    row = pool.take()
    pool.cancel(str(p))
    idx.fail = True
    worker._run_one(row)
    assert db.get_file_by_path(str(p)) is None, "被取消的文件不该留下任何行"


def test_plain_failure_still_records_an_error(tmp_path):
    """没有取消时的普通失败：仍然记错误明细 + 落终态（别被本次改动吃掉）。"""
    p, db, idx, pool, worker = _setup(tmp_path)
    row = pool.take()
    idx.fail = True
    worker._run_one(row)

    rec = db.get_file_by_path(str(p))
    assert rec is not None
    assert rec["vectorize_status"] == "failed", "普通失败仍应落到 failed"


def test_success_path_unchanged(tmp_path):
    """成功路径行为不变（对照组）。"""
    p, db, idx, pool, worker = _setup(tmp_path)
    row = pool.take()
    worker._run_one(row)
    rec = db.get_file_by_path(str(p))
    assert rec is not None and rec["vectorize_status"] == "completed"


def test_success_after_cancel_still_discards(tmp_path):
    """取消 + 成功 → 仍然丢弃（原语义，防本次改动把它改坏）。"""
    p, db, idx, pool, worker = _setup(tmp_path)
    row = pool.take()
    pool.cancel(str(p))
    worker._run_one(row)
    assert db.get_file_by_path(str(p)) is None
