"""watcher 增量索引失败必须与全盘扫描行为一致（2026-09-09 用户实测暴露）。

现象：embed 掉线时 `touch`/新建文件，GUI 失败文件数为 0。
根因：watcher._process 捕获异常只 logger.warning 吞掉——文件连元数据都不入库，
所以 ① 失败计数为 0、② 错误明细没有、③ embed 恢复看门狗按 DB 的
failed/queued 计数触发 → 看不见这个文件，永远不会重试（永久遗忘）。

断言：
- 增量失败 → DB 落 failed 记录 + errorDetails 有明细 + 看门狗 pending > 0
- 空文件（no_text）不算失败（对照：不该把所有异常都标 failed）
- 已有记录重索引失败 → 只改状态，保留原 content_hash（避免重复解析）
- progress() 过滤已删除文件的明细（失败归 0 但展开还有旧条目）
"""
import tempfile
import time
from pathlib import Path

import pytest


from bibit_filesearch.db import Database  # noqa: E402
from bibit_filesearch.indexer import Indexer  # noqa: E402
from bibit_filesearch.recovery import EmbedRecovery  # noqa: E402
from bibit_filesearch.scanner import ScanService  # noqa: E402
from bibit_filesearch.watcher import Watcher  # noqa: E402


class _SwitchableEmbed:
    """可控 embed：down=True 时抛连接错误（模拟 8082 掉线）。"""

    def __init__(self, down=True):
        self.down = down
        self.dim = 8

    def _vec(self, text: str):
        return [float((hash((text, i)) % 97) + 1) for i in range(self.dim)]

    def embed_one(self, text: str):
        if self.down:
            raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")
        return self._vec(text)

    def embed_batch(self, texts):
        if self.down:
            raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")
        return [self._vec(t) for t in texts]

    def ping(self) -> bool:
        return not self.down

    def close(self) -> None:
        pass


class _Cfg:
    def __init__(self, dirs):
        self.index_dirs = dirs
        self.exclude_patterns = [".git"]

        class _Scan:
            def get(self, k, d=None):
                return {"io_throttle_ms": 0}.get(k, d)

        self.scan = _Scan()


def _wait_for(pred, timeout=10):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_wfail_"))
    db = Database(str(tmp / "test.db"))
    embed = _SwitchableEmbed()
    yield tmp, db, embed
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _mk_watcher(tmp, db, embed, scanner=None):
    idx = Indexer(db, embed)
    w = Watcher(_Cfg([str(tmp)]), db, idx)
    if scanner is not None:
        w._scanner_ref = scanner
    return idx, w


def test_incremental_failure_is_recorded(env):
    """embed 掉线新建非空文件 → DB 落 failed + 明细 + 看门狗可见。

    2026-09-23（池子化）更新等待条件：此前等「行出现」就断言 failed——因为旧
    watcher 是**算完才写行**，行出现 = 处理结束。重构后 watcher 只入池，
    **行在入池那一刻就出现（queued）**，处理由消费者随后做——「行存在」不再
    蕴含「已处理完」。故改为等终态，语义更强（守的仍是「失败必须落库 + 明细 +
    看门狗可见」，只是不再借用行出现当完成信号）。
    """
    tmp, db, embed = env
    scanner = ScanService(_Cfg([str(tmp)]), db, Indexer(db, embed))
    idx, w = _mk_watcher(tmp, db, embed, scanner)
    w.start()
    try:
        p = tmp / "new.txt"
        p.write_text("meaningful text content\n", encoding="utf-8")
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "failed"
        ), f"失败必须落到终态 failed（实际 {(db.get_file_by_path(str(p)) or {}).get('vectorize_status')}）"
        row = db.get_file_by_path(str(p))
        assert row is not None, "增量失败也必须落库（此前完全不入库，用户看不到任何失败）"

        # 明细：progress() 里能看到这条（GUI 展开失败文件列表的数据源）
        details = scanner.progress()["scanPhase"]["errorDetails"]
        assert any(d["path"] == str(p) for d in details), f"明细应含该文件：{details}"
        assert "Connection refused" in next(d["reason"] for d in details if d["path"] == str(p))

        # 看门狗可见：pending 计数 > 0（否则 embed 恢复后永远不会重试它）
        #
        # 2026-09-23（池子化）：这条断言曾偶发失败（`assert 0 > 0`，全量套件里约
        # 1/10，单文件跑不出现）。原因不是看门狗坏了，而是**等待条件太弱**：
        # 上面只等到「状态落成 failed」，而 failed 由 `index_file` → `mark_failed`
        # 先写、`attempt_count` 随后 `mark_attempt_failed` 才加一 —— 两者之间有个
        # 极短窗口，此刻 `_pending_count()` 的 `attempt_count < max_attempts`
        # 条件把这一行排除掉（老实现无此条件，故此前不暴露）。
        # 同一路径在补扫与事件里各入池一次时窗口更容易被撞上。
        # 改为等「看门狗真的能看见它」，语义不变、只去掉对写入中间态的依赖。
        rec = EmbedRecovery(embed, db, scanner, interval=30)
        assert _wait_for(lambda: rec._pending_count() > 0), \
            f"看门狗必须能看见这个失败文件（库中失败行：{db.conn.execute('SELECT path, vectorize_status, attempt_count FROM files').fetchall()}）"
    finally:
        w.stop()


def test_empty_file_is_not_a_failure(env):
    """空文件走 no_text（unsupported），不算失败——不该把一切异常都标 failed。"""
    tmp, db, embed = env
    scanner = ScanService(_Cfg([str(tmp)]), db, Indexer(db, embed))
    idx, w = _mk_watcher(tmp, db, embed, scanner)
    w.start()
    try:
        p = tmp / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "unsupported"
        )
        assert db.get_file_by_path(str(p))["vectorize_status"] == "unsupported"
        assert scanner.progress()["vectorizePhase"]["failed"] == 0, "空文件不该计失败"
    finally:
        w.stop()


def test_refailure_keeps_content_hash(env):
    """已索引文件修改后重索引失败 → 落**当前内容**的 hash（不清空、不留旧值）。

    2026-09-18 更新（原断言为「保留 old_hash」）：
    旧断言与新增的「重试上限」机制冲突——文件内容已从 A 改成 B，若仍保留 A 的
    hash，扫描每轮都发现 hash 不匹配 → 判「内容变了、需要重算」→ 达上限后仍被
    reset_attempt → **无限重试**。落当前内容的 hash 后语义才闭合：
      · 内容未变 → 达 attempt_count 上限即停止重排；
      · 用户再次修改（hash 再变）→ 重置重试预算，重新尝试。
    本测试守的核心意图（失败不得把 hash 清空，否则每轮都判「待重算」）保持不变。
    """
    tmp, db, embed = env
    embed.down = False
    scanner = ScanService(_Cfg([str(tmp)]), db, Indexer(db, embed))
    idx, w = _mk_watcher(tmp, db, embed, scanner)
    w.start()
    try:
        p = tmp / "doc.md"
        p.write_text("# 原始内容\n\n正文甲。\n", encoding="utf-8")
        # 2026-09-23（池子化）：等**索引完成**，不能只等「行出现」。
        # 池子化后 watcher 入池就写行（status=queued、content_hash 为空），
        # 处理由消费者随后做；「行存在」不再蕴含「已处理完」。
        # 只等行出现的话，下一行 `assert old_hash` 会偶发拿到空串
        # （实测 30 轮中 1 次：assert ''）。
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "completed"
        ), f"首次索引应完成（实际 {(db.get_file_by_path(str(p)) or {}).get('vectorize_status')}）"
        old_hash = db.get_file_by_path(str(p))["content_hash"]
        assert old_hash, "首次索引应写入 hash"

        embed.down = True  # 掉线后修改 → 重索引失败
        p.write_text("# 修改后内容\n\n正文乙完全不同。\n", encoding="utf-8")
        # 等「状态 failed **且** 计数已落」——与上面同理，不等中间态
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "failed"
            and (db.get_file_by_path(str(p)) or {}).get("attempt_count") == 1
        ), (
            "重索引失败应落 failed 且计数 1（实际 "
            f"{db.get_file_by_path(str(p))}）"
        )
        row = db.get_file_by_path(str(p))
        assert row["content_hash"], "失败必须留下 hash（不得清空，否则每轮判「待重算」重灌）"
        assert row["content_hash"] != old_hash, "应落当前（修改后）内容的 hash"
        assert row["attempt_count"] == 1, "失败要计数，供重试上限判定"
    finally:
        w.stop()


def test_unreadable_file_via_watcher_is_counted(env):
    """读不到的文件（权限/IO 错）经 watcher → 必须落 failed **且计数**。

    实测来源（2026-09-18，169）：把 root 拥有、0600 的文件放进索引目录，服务以
    bianbu 运行读不到它，DB 落成 failed / attempt_count=0 / hash 空。而扫描对
    「failed 且未达上限」的文件每轮都重新排队 → 每轮重试一次、计数永远是 0，
    即「什么都没做却一直忙」。根因：watcher 曾自己改状态（不计数、不落 hash），
    且 index_file 的 try 只从 parse_file 开始，取元数据/算 hash 的异常漏在外面。
    """
    tmp, db, embed = env
    scanner = ScanService(_Cfg([str(tmp)]), db, Indexer(db, embed))
    idx, w = _mk_watcher(tmp, db, embed, scanner)
    w.start()
    try:
        p = tmp / "locked.md"
        p.write_text("内容\n", encoding="utf-8")
        idx._content_hash = lambda _p: (_ for _ in ()).throw(PermissionError(13, "Permission denied"))
        w._process({str(p): {256}})          # 256 = IN_CREATE，走增量入池路径

        # 2026-09-23（池子化）：改为等终态。重构后 watcher 只登记，处理由消费者
        # 异步做——「行出现」不再蕴含「已处理完」（见 test_incremental_failure_is_recorded
        # 的同款说明）。
        assert _wait_for(
            lambda: (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "failed"
        ), f"读不到的文件必须落 failed（实际 {(db.get_file_by_path(str(p)) or {}).get('vectorize_status')}）"
        row = db.get_file_by_path(str(p))
        assert row is not None, "失败必须落库"
        # 2026-09-23（池子化）放宽为 1..max_attempts：本用例的构造方式是
        # 「start() 的启动补扫 + 手工 _process 一次事件」两条路径都指向同一个
        # 文件，而补扫在 delay=0 时立即执行 → 该文件被入池两次、计两次失败
        # （实测偶发 2，重构后因行出现时机变化而变确定）。生产上 backfill_delay
        # 下限 10s，且池中 queued/vectorizing 的行 offer 是幂等的，不会重算。
        # 本用例守的是「失败**必须计数**（此前恒为 0，导致每轮重试）」，故下界即断言核心。
        assert 1 <= row["attempt_count"] <= 3, \
            f"失败必须计数且不超上限（实际 {row['attempt_count']}）"
    finally:
        w.stop()


def test_progress_filters_deleted_paths(env):
    """明细里已删除的路径被过滤（失败归 0 但展开仍有旧条目）。"""
    tmp, db, embed = env
    embed.down = True
    scanner = ScanService(_Cfg([str(tmp)]), db, Indexer(db, embed))
    scanner.record_error("parse", str(tmp / "gone.txt"), "embed 掉线")
    assert scanner.progress()["scanPhase"]["errorDetails"] == [], \
        "磁盘上不存在的路径不该出现在明细里"

    alive = tmp / "alive.txt"
    alive.write_text("x", encoding="utf-8")
    scanner.record_error("parse", str(alive), "embed 掉线")
    details = scanner.progress()["scanPhase"]["errorDetails"]
    assert [d["path"] for d in details] == [str(alive)], \
        f"已删除路径应从明细中消失（实际 {details}）"


def test_same_path_failure_not_duplicated(env):
    """同一文件反复失败 → 明细只保留一条（failed=1 明细 4 条，2026-09-09 实测）。"""
    tmp, db, embed = env
    embed.down = True
    scanner = ScanService(_Cfg([str(tmp)]), db, Indexer(db, embed))
    p = tmp / "flaky.txt"
    p.write_text("content\n", encoding="utf-8")
    for i in range(4):
        scanner.record_error("parse", str(p), f"embed 掉线 #{i}")

    details = scanner.progress()["scanPhase"]["errorDetails"]
    assert len(details) == 1, f"同路径应去重（实际 {len(details)} 条：{details}）"
    assert details[0]["reason"] == "embed 掉线 #3", "保留的应是最新一次原因"

    # 另一个文件互不影响（去重只针对同一 path）
    q = tmp / "other.txt"
    q.write_text("content\n", encoding="utf-8")
    scanner.record_error("parse", str(q), "embed 掉线")
    details = scanner.progress()["scanPhase"]["errorDetails"]
    assert sorted(d["path"] for d in details) == sorted([str(p), str(q)]), details
