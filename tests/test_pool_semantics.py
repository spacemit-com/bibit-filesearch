"""索引池重构的语义回归（2026-09-23）。

本文件守的是重构后的**新契约**（旧行为已被替换，旧测试多已同步改过）。每条
用例都对应一处曾经真实出错的场景，不是为覆盖率而写：

1. 「待向量化」精确等于池中未算完的行数，且**不超过库中总行数**
   （旧算法 queued+vectorizing+inflight 稳定多算 1，实测 300 帧中 299 帧）。
2. 池中已有该行时不重复入池 —— 一次 `cp` 会产生多个 inotify 事件。
3. 入池后**有人消费**（旧 worker 池空即退出，实测 8 秒无人处理）。
4. 文件从磁盘删除 → 库中不留 completed 幽灵行（旧实现只在全盘遍历末尾清一次）。
5. 在算期间被取消 → 结果丢弃、库中不留行（删除事件不打断在飞，收尾时清）。
6. 「未纳入索引」计数：被排除规则剪掉的文件数得出来（旧实现四个观测面全静默）。
"""
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer
from bibit_filesearch.pool import ST_COMPLETED, ST_FAILED, ST_QUEUED, ST_UNSUPPORTED, ST_VECTORIZING, IndexPool, IndexWorker


class _Embed:
    """确定性 embed 替身（dim=8，与 _Cfg 无关，池子不校验维度）。"""

    def __init__(self, down=False):
        self.down = down
        self.dim = 8

    def _vec(self, text):
        return [float((hash((text, i)) % 97) + 1) for i in range(self.dim)]

    def embed_one(self, text):
        if self.down:
            raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")
        return self._vec(text)

    def embed_batch(self, texts):
        if self.down:
            raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")
        return [self._vec(t) for t in texts]

    def ping(self):
        return not self.down

    def close(self):
        pass


class _Cfg:
    """ScanService 的最小配置替身。

    `scan` 用 dict 更好：ScanService 只读 `config.scan.get(k, default)`，
    测试按需覆盖 io_throttle_ms / max_depth / max_attempts 即可。
    """

    def __init__(self, dirs, excludes=None, max_depth=0, indexable_types=None):
        self.index_dirs = dirs
        self.exclude_patterns = (
            ["#recycle", "*.tmp"] if excludes is None else list(excludes))
        self.indexable_types = list(indexable_types or ["txt", "md", "pdf"])
        self.scan = {
            "io_throttle_ms": 0,
            "max_depth": max_depth,
            "max_attempts": 3,
        }


def _wait_for(pred, timeout=15):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_pool_"))
    db = Database(str(tmp / "test.db"))
    embed = _Embed()
    yield tmp, db, embed
    embed.close()
    db.close()
    shutil.rmtree(tmp, ignore_errors=True)


def _status(db, p):
    row = db.get_file_by_path(str(p))
    return row["vectorize_status"] if row else None


# ---------- 1. 计数口径 ----------

def test_pending_never_exceeds_total_rows(env):
    """★★★ 「待向量化」不得超过库中总行数（旧的 +1 错法会突破这个上界）。

    实测（隔离环境，8 个 md）：旧算法算出 pendingTotal = 9，而库里只有 8 行。
    根因是 index_file 入口的 `_inflight += 1` 挂在两条路径的公共入口上，scanner
    正在算的文件既有 vectorizing 行、又被计入 inflight。
    """
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    worker = IndexWorker(pool, idx)
    for i in range(8):
        (tmp / f"doc{i}.md").write_text(f"content {i}\n", encoding="utf-8")
    worker.start()
    try:
        for i in range(8):
            pool.offer(str(tmp / f"doc{i}.md"))
        # 采样期间不断核对上界
        bad = []
        for _ in range(300):
            pending = pool.pending_count()
            total = sum(pool.counts().values())
            if pending > total:
                bad.append((pending, total))
            time.sleep(0.005)
        assert not bad, f"「待向量化」超过库中总行数（前 5 次：{bad[:5]}）"
        assert _wait_for(lambda: pool.pending_count() == 0), \
            f"最终应排空（实际 {pool.counts()}）"
        assert pool.counts().get(ST_COMPLETED) == 8, f"8 个文件都应 completed：{pool.counts()}"
    finally:
        worker.stop()


def test_pending_equals_db_unfinished_rows(env):
    """「待向量化」精确等于「queued + vectorizing 的库行数」（不再掺进程内计数）。"""
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    embed.down = True  # 让文件停在失败，便于观察
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp / name).write_text("x\n", encoding="utf-8")
        pool.offer(str(tmp / name))
    # 尚未消费：三行全 queued
    assert pool.pending_count() == 3
    rows = db.conn.execute("SELECT COUNT(*) FROM files WHERE vectorize_status=?", (ST_QUEUED,)).fetchone()[0]
    assert pool.pending_count() == rows == 3
    # 手工取走一行（模拟在算）→ pending 不变（仍算「没算完」）
    row = pool.take()
    assert row is not None
    assert pool.pending_count() == 3, "在算的那个也要算进「待向量化」"


# ---------- 2. 幂等入池 ----------

def test_offer_is_idempotent_for_same_path(env):
    """一次 cp 会产生多个 inotify 事件（CREATE/WRITE/CLOSE_WRITE）→ 只入池一次。"""
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    p = tmp / "one.txt"
    p.write_text("hello\n", encoding="utf-8")
    first = pool.offer(str(p))
    for _ in range(5):
        again = pool.offer(str(p))
        assert again == "noop", f"重复 offer 应幂等，实际 {again}"
    assert first in ("added", "refreshed")
    assert pool.pending_count() == 1, f"库中应只有 1 行待办：{pool.counts()}"


# ---------- 3. 入池后有人消费（缺泵的回归） ----------

def test_offered_file_is_consumed_without_scan(env):
    """★★★ 入池后必须有人消费 —— 不需要任何扫描动作。

    旧实现：`_run_worker` 池空即 `return` 退出线程，重开的唯一入口是
    `start()`/`resume()`。实测：全盘扫描跑完后手工入池一行，**8 秒无人处理**
    （worker 线程存活 = False，状态恒 queued）。池子化后消费者常驻。
    """
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    worker = IndexWorker(pool, idx)
    worker.start()
    try:
        assert _wait_for(lambda: worker.alive), "消费者应已起来"
        p = tmp / "late.txt"
        p.write_text("arrived after scan finished\n", encoding="utf-8")
        pool.offer(str(p))          # 完全模拟 watcher 入池，不做任何扫描
        assert _wait_for(lambda: _status(db, p) == ST_COMPLETED, timeout=10), \
            f"入池后应被消费（实际 {_status(db, p)}）"
        assert worker.alive, "消费完也不能退出（等下一次唤醒）"
    finally:
        worker.stop()


def test_worker_survives_empty_pool(env):
    """池空时消费者不退出（等唤醒 + 超时兜底）。"""
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    worker = IndexWorker(pool, idx)
    worker.start()
    try:
        time.sleep(5.0)  # 远超 wait 超时（2s）
        assert worker.alive, "池空 5 秒后消费者仍须存活"
    finally:
        worker.stop()


# ---------- 4. 删除不留幽灵行 ----------

def test_purge_removes_completed_row(env):
    """★★ 已完成的文件被删除 → 库中不留 completed 幽灵行。

    实测：文件已从磁盘删除，池中 B 行仍 `status=completed chunks=1`；
    旧实现只在**全盘遍历末尾**跑一次 `_prune_deleted`，watcher 关闭时无人回收。
    """
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    worker = IndexWorker(pool, idx)
    worker.start()
    try:
        p = tmp / "gone.txt"
        p.write_text("will be deleted\n", encoding="utf-8")
        pool.offer(str(p))
        assert _wait_for(lambda: _status(db, p) == ST_COMPLETED)
        assert db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0

        p.unlink()
        assert pool.purge(str(p)) != "missing"
        assert db.get_file_by_path(str(p)) is None, "删除后库里不该留行"
        n = db.conn.execute(
            "SELECT COUNT(*) FROM chunks c LEFT JOIN files f ON f.file_id=c.file_id WHERE f.file_id IS NULL"
        ).fetchone()[0]
        assert n == 0, f"不该留孤儿块（{n} 条）"
    finally:
        worker.stop()


# ---------- 5. 取消语义（在飞删除） ----------

def test_cancel_during_index_discards_result(env):
    """在算期间被取消 → 算完丢弃，库中不留行（不打断在飞，收尾时清）。"""
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    p = tmp / "flight.txt"
    p.write_text("being indexed\n", encoding="utf-8")
    pool.offer(str(p))
    row = pool.take()
    assert row is not None, "应能取到一行"
    # 此刻该行 = vectorizing（在飞）
    assert _status(db, p) == ST_VECTORIZING
    res = pool.cancel(str(p))
    assert res == "deferred", f"在飞的取消应延迟结算，实际 {res}"
    # 消费者算完后 commit：结果丢弃
    assert pool.commit(str(p)) is False
    assert db.get_file_by_path(str(p)) is None, "被取消的文件不该留在库里"


def test_cancel_queued_removes_row_immediately(env):
    """未消费（queued）的取消 → 立刻删行（移出知识库路径的主路径）。"""
    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)
    p = tmp / "queued.txt"
    p.write_text("waiting\n", encoding="utf-8")
    pool.offer(str(p))
    assert _status(db, p) == ST_QUEUED
    assert pool.cancel(str(p)) == "cancelled"
    assert db.get_file_by_path(str(p)) is None


# ---------- 6. 未纳入索引（excluded） ----------

def test_excluded_tree_counts_toward_not_indexed(env):
    """被排除规则剪掉的目录 → 计入「未纳入索引」（旧实现四个观测面全静默）。

    用 ScanService 的遍历数计（这是唯一产出该计数的地方）。
    """
    from bibit_filesearch.scanner import ScanService

    tmp, db, embed = env
    kb = tmp / "kb"
    (kb / "#recycle").mkdir(parents=True)
    (kb / "#recycle" / "a.txt").write_text("junk\n", encoding="utf-8")
    (kb / "keep.md").write_text("kept\n", encoding="utf-8")
    idx = Indexer(db, embed)
    scanner = ScanService(_Cfg([str(kb)]), db, idx)
    scanner.start()
    try:
        assert _wait_for(lambda: scanner.progress()["scanPhase"]["excluded"] > 0, timeout=10), \
            "被排除的文件必须被计数（此前全静默）"
        assert scanner.progress()["scanPhase"]["excluded"] >= 1
    finally:
        scanner.stop()


# ---------- 7. 释放触发点：池空不再无条件释放（方案 A，2026-10-10） ----------

def test_settle_does_not_release_unconditionally(env, monkeypatch):
    """★★ 池空 ≠ 释放：空闲不足 TTL 时 `_settle()` 绝不能放掉 worker。

    实测：`_settle()` 里原本还有一条「池空就 `release_idle()`」的**无条件**释放，
    它认的触发条件是「索引池排空」——而这既不代表「不会再用了」（下一张图随时可能来；
    实测两张图隔 15s 到达时，每张都重建一遍三个 worker、各多付 1.5s），也不代表
    「内存贵」（与图片大小无关）。方案 A 删掉它，运行期释放只剩「空闲 ≥
    `ocr.idle_release_secs`」这一条。

    本用例用**带 scanner** 的 IndexWorker（生产形态：scanner 与 watcher 共用池），
    正是曾经放错位置的那条路径。两条断言合起来锁住：① 空闲不足 → 一次都不放；
    ② 空闲满足 → 恰好放一次（时间判据是唯一触发点）。
    """
    from bibit_filesearch import ocr_engine, ocr_ep

    tmp, db, embed = env
    idx = Indexer(db, embed)
    pool = IndexPool(db, idx)

    calls = {"n": 0}
    monkeypatch.setattr(
        ocr_engine, "release_idle",
        lambda: calls.__setitem__("n", calls["n"] + 1) or 3)

    class _ScannerStub:
        def recount_excluded(self, **kw):
            return False

        def mark_idle_if_drained(self):
            return None

    worker = IndexWorker(pool, idx, scanner=_ScannerStub())

    monkeypatch.setattr(ocr_ep, "should_release_when_idle", lambda now=None: False)
    worker._settle()
    assert calls["n"] == 0, "池空但空闲不足 TTL 时不得释放（原「池空即释放」已删）"

    monkeypatch.setattr(ocr_ep, "should_release_when_idle", lambda now=None: True)
    worker._settle()
    assert calls["n"] == 1, "空闲满足 TTL 时应恰好释放一次（时间判据是唯一触发点）"

