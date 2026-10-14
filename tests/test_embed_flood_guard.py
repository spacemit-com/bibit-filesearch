"""防 embed 灌爆护栏测试（2026-09-18）。

背景（169 实测）：一棵 40MB 的反编译 JS 树被折算成约 6.8 万个 embed 输入灌给
只有 4 个 slot 的 embed 服务，其中单个 24.8MB 的 minified 文件被整包成一个请求
（83,782 条输入，实测）；失败/待处理文件的 content_hash 为空导致幂等跳过失效，
扫描每轮全量重算；embed 崩溃重启后看门狗又立刻触发一轮 → 「OOM → 重启 → 再灌
一轮」正反馈。

本文件守四条护栏：
  1. embed_batch 按 batch_size 分批（单请求输入数有上界）
  2. ping 是真活探测（半死态可见，不再只看 /health）
  3. 单文件块数上限（超限只索引前 N 块并标 oversized）
  4. 重试计数封顶 + 看门狗冷却（终止每轮全量重灌与自动重建正反馈）
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import pytest

from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer
from bibit_filesearch.metadata import extract_metadata
from bibit_filesearch.model_client import EmbedClient, _InputTooLarge
from bibit_filesearch.recovery import EmbedRecovery
from bibit_filesearch.scanner import ScanService

from fake_embed import DIM as FAKE_DIM
from fake_embed import FakeEmbed


# ---------------------------------------------------------------- 1. 分批

class _SplitProbe(EmbedClient):
    """只测分批逻辑：记录每次实际请求的输入数，不联网。"""

    def __init__(self, batch_size: int):
        super().__init__("http://127.0.0.1:1", dim=4, batch_size=batch_size)
        self.calls: list[int] = []

    def _embed_request(self, texts):
        self.calls.append(len(texts))
        return [[0.0] * 4 for _ in texts]


def test_embed_batch_splits_by_batch_size():
    """10 个输入 / batch_size=3 → 4 次请求，尺寸 3,3,3,1，顺序与数量不变。"""
    c = _SplitProbe(batch_size=3)
    out = c.embed_batch([f"t{i}" for i in range(10)])
    assert len(out) == 10
    assert c.calls == [3, 3, 3, 1]
    c.close()


def test_embed_batch_single_request_when_under_limit():
    """输入数不超上界 → 只发一次（不引入额外往返）。"""
    c = _SplitProbe(batch_size=32)
    c.embed_batch(["a", "b"])
    assert c.calls == [2]
    c.close()


def test_embed_batch_empty():
    c = _SplitProbe(batch_size=3)
    assert c.embed_batch([]) == []
    assert c.calls == []
    c.close()


def test_embed_batch_zero_means_no_split():
    """batch_size=0 → 不分批（保留旧行为，供极端场景回退）。"""
    c = _SplitProbe(batch_size=0)
    c.embed_batch([f"t{i}" for i in range(5)])
    assert c.calls == [5]
    c.close()


# ---------------------------------------------------------------- 2. 真活探测

class _Resp:
    def __init__(self, code: int, payload):
        self.status_code = code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHTTP:
    """记录请求；GET 一律视为违规（ping 不得再走 /health）。"""

    def __init__(self, resp=None, exc=None):
        self.resp = resp
        self.exc = exc
        self.posts: list[tuple[str, dict]] = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if self.exc:
            raise self.exc
        return self.resp

    def get(self, *a, **k):
        raise AssertionError("ping 不得再依赖 GET /health（半死态不可见）")


def _client_with(resp=None, exc=None) -> EmbedClient:
    c = EmbedClient("http://127.0.0.1:1", dim=4)
    c._client = _FakeHTTP(resp=resp, exc=exc)
    return c


def test_ping_true_on_real_embedding_response():
    c = _client_with(_Resp(200, {"data": [{"embedding": [0.1, 0.2]}]}))
    assert c.ping() is True
    assert c._client.posts[0][0] == "/v1/embeddings", "必须走真实业务端点"


def test_ping_false_on_503_even_though_health_would_be_200():
    """核心守卫：服务返回 503（半死）时 ping 必须为 False。

    这正是原实现看不见的故障态——只 GET /health 时它会返回 True，
    看门狗于是认为服务健康，永不触发恢复。
    """
    c = _client_with(_Resp(503, {"error": "no slot available"}))
    assert c.ping() is False
    assert "/v1/embeddings" in c._client.posts[0][0]


def test_ping_false_on_empty_payload():
    c = _client_with(_Resp(200, {"data": []}))
    assert c.ping() is False


def test_ping_false_on_embedding_missing():
    c = _client_with(_Resp(200, {"data": [{"index": 0}]}))
    assert c.ping() is False


def test_ping_false_on_connection_error():
    c = _client_with(exc=ConnectionError("refused"))
    assert c.ping() is False


def test_ping_does_not_raise():
    """探活永远不得抛异常（调用方在看门狗线程里）。"""
    c = _client_with(exc=RuntimeError("boom"))
    assert c.ping() is False


# ------------------------------------------- 2b. 物理批次超限的二分降级

class _FlakyServer(EmbedClient):
    """模拟 embed 服务：单条输入按字符数折算 token，超限即拒收。

    最坏密度（1 字符 ≈ 1 token）——正是压缩 JS 的实测行为（600 字符 = 548~616 token）。
    """

    def __init__(self, limit_tokens: int, batch_size: int = 32):
        super().__init__("http://127.0.0.1:1", dim=4, batch_size=batch_size)
        self.limit = limit_tokens
        self.calls: list[int] = []          # 每次请求的条数
        self.sizes: list[list[int]] = []    # 每次请求各条的字符数

    def _embed_request(self, texts):
        self.calls.append(len(texts))
        self.sizes.append([len(t) for t in texts])
        for t in texts:
            if len(t) > self.limit:
                raise _InputTooLarge(f"input ({len(t)} tokens) is too large to process")
        return [[0.5] * 4 for _ in texts]


def test_no_split_when_all_inputs_within_limit():
    """正常内容一次通过，不引入额外请求。"""
    c = _FlakyServer(limit_tokens=512)
    out = c.embed_batch(["x" * 100, "y" * 200])
    assert len(out) == 2
    assert c.calls == [2], "不该触发二分"


def test_single_oversized_input_is_split_by_chars():
    """单条超限 → 按字符二分，仍返回 1 条向量（保持 1:1，不破坏 indexer 断言）。"""
    c = _FlakyServer(limit_tokens=512)
    out = c.embed_batch(["z" * 900])
    assert len(out) == 1, "向量数必须等于输入数"
    assert len(out[0]) == 4
    # 900 → 450+450，都在限内 → 共 3 次调用（原始 + 两次二分）
    assert c.calls == [1, 1, 1]
    assert c.sizes[1] == [450] and c.sizes[2] == [450]


def test_oversized_input_split_recursively():
    """二分后仍超 → 继续二分，直到每段在限内（递归收敛）。"""
    c = _FlakyServer(limit_tokens=200)
    out = c.embed_batch(["q" * 1600])
    assert len(out) == 1, "最终仍须返回 1 条向量"
    assert len(out[0]) == 4
    flat = [s for sizes in c.sizes for s in sizes]
    assert flat[0] == 1600, "首个请求应是原始长度"
    assert min(flat) <= 200, f"应递归切到限内，实际最小 {min(flat)}"


def test_oversized_list_is_narrowed_by_queue_split():
    """整批被拒时先按队列二分，定位到超限那条。"""
    c = _FlakyServer(limit_tokens=512, batch_size=32)
    texts = ["a" * 100, "b" * 100, "c" * 900, "d" * 100]
    out = c.embed_batch(texts)
    assert len(out) == 4, "4 条输入必须得 4 条向量"
    assert all(len(v) == 4 for v in out)


def test_too_large_is_not_retried():
    """超限错误不得走重试（重试结果相同，纯浪费）。"""
    c = _FlakyServer(limit_tokens=512)
    c.batch_size = 0  # 不分批，直接走 _embed_resilient
    c.embed_batch(["z" * 900])
    # 第一次 900（被拒）→ 450 + 450（成功）：3 次调用，没有重复同样的尺寸
    assert c.calls.count(1) == 3
    assert c.sizes[0] == [900], "首个请求应是原始长度"


# ------------------------------------------- 2c. 失败留痕（下沉到 indexer）

class _AlwaysFailEmbed:
    """每次 embed 都失败的替身。"""

    def __init__(self):
        self.dim = 4

    def embed_batch(self, texts):
        raise RuntimeError("embed 请求失败（重试 2 次）: 500")

    def embed_one(self, text):
        raise RuntimeError("boom")

    def ping(self):
        return False

    def probe_dim(self):
        return 4

    def close(self):
        pass


def test_index_failure_records_hash_and_attempt(env):
    """index_file 失败必须落 content_hash + attempt_count（缺口 B 的守卫）。

    这条守的是「watcher 补扫路径也覆盖」——失败留痕下沉到 index_file 后，
    任何调用方（scanner worker / watcher 补扫 / backfill）都自动生效。
    """
    tmp, db, _ = env
    p = tmp / "will_fail.md"
    p.write_text("内容会被 embed 拒绝", encoding="utf-8")
    idx = Indexer(db, _AlwaysFailEmbed())

    with pytest.raises(RuntimeError):
        idx.index_file(str(p))

    row = db.get_file_by_path(str(p))
    assert row is not None, "失败也要有记录，否则下轮无法识别"
    assert row["vectorize_status"] == "failed"
    assert row["attempt_count"] == 1, f"应记一次失败（实际 {row['attempt_count']}）"
    assert row["content_hash"], "失败必须落 hash——否则每轮扫描都判「待重算」重灌"


def test_unreadable_file_is_recorded_failed_not_left_vectorizing(env):
    """读不到文件（权限/IO 错）也必须落失败，不能停在 vectorizing。

    实测来源：把 root 拥有、0600 的文件放进索引目录 → worker 先把状态置为
    vectorizing，随后取元数据/算 hash 抛 PermissionError。修复前该 try 只从
    parse_file 开始，异常在 try 之外直接抛给调用方：状态永远停在 vectorizing、
    attempt_count 不增 —— UI 看不到失败，也永不重试。
    """
    tmp, db, embed = env
    p = tmp / "unreadable.md"
    p.write_text("内容", encoding="utf-8")
    # 模拟「扫描已把它排入队列」，再让读文件阶段失败
    db.upsert_file(extract_metadata(str(p)), "", "", status="queued")
    idx = Indexer(db, embed)

    def boom(_p):
        raise PermissionError(13, "Permission denied")

    idx._content_hash = boom          # 失败发生在解析/embed 之前
    with pytest.raises(PermissionError):
        idx.index_file(str(p))

    row = db.get_file_by_path(str(p))
    assert row is not None
    assert row["vectorize_status"] == "failed", "不得停在 vectorizing"
    assert row["attempt_count"] == 1, "必须记一次失败，否则永不重试"


def test_repeated_failures_reach_limit_and_stop_requeue(env):
    """连续失败累积到上限后，扫描不再重排（终止每轮全量重灌）。"""
    tmp, db, _ = env
    p = tmp / "always_fail.md"
    p.write_text("内容", encoding="utf-8")
    idx = Indexer(db, _AlwaysFailEmbed())

    for _ in range(3):
        with pytest.raises(RuntimeError):
            idx.index_file(str(p))

    row = db.get_file_by_path(str(p))
    assert row["attempt_count"] == 3

    sc = ScanService(_Cfg([str(tmp)], {"max_attempts": 3}), db, idx)
    sc._enqueue_file(str(p))
    after = db.get_file_by_path(str(p))
    assert after["vectorize_status"] == "failed", "达上限后不得再排入队列"
    assert after["attempt_count"] == 3, "不应再加计数"


def test_success_resets_attempt_count_via_index(env):
    """失败过但后来成功 → attempt_count 清零（用户修好文件后能继续）。"""
    tmp, db, embed = env
    p = tmp / "later_ok.md"
    p.write_text("内容", encoding="utf-8")

    idx_bad = Indexer(db, _AlwaysFailEmbed())
    with pytest.raises(RuntimeError):
        idx_bad.index_file(str(p))
    assert db.get_file_by_path(str(p))["attempt_count"] == 1

    idx_ok = Indexer(db, embed)
    r = idx_ok.index_file(str(p))
    assert r["status"] in ("indexed", "updated")
    assert db.get_file_by_path(str(p))["attempt_count"] == 0


# ---------------------------------------------------------------- 3. 单文件块数上限

class _Cfg:
    def __init__(self, dirs, scan: dict | None = None):
        self.index_dirs = dirs
        self.exclude_patterns = []
        self._scan = {"io_throttle_ms": 0, "max_depth": 0, **(scan or {})}

        class _Scan:
            def __init__(self, d):
                self._d = d

            def get(self, k, default=None):
                return self._d.get(k, default)

        self.scan = _Scan(self._scan)


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_flood_"))
    db = Database(str(tmp / "test.db"))
    embed = FakeEmbed()
    yield tmp, db, embed
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def test_oversized_file_indexes_first_n_chunks_only(env):
    """超出 max_chunks_per_file → 只索引前 N 块，并回报 oversized/total_chunks。"""
    tmp, db, embed = env
    p = tmp / "big.txt"
    p.write_text("".join(f"段落{i}内容。" for i in range(400)), encoding="utf-8")
    idx = Indexer(db, embed, chunk_size=20, chunk_overlap=0, max_chunks_per_file=3)

    r = idx.index_file(str(p))
    assert r["status"] == "indexed"
    assert r["chunks"] == 3
    assert r["oversized"] is True
    assert r["total_chunks"] > 3

    fid = r["file_id"]
    assert db.get_file(fid)["vectorize_status"] == "completed", "截断后仍须可检索"
    assert len(db.get_chunks(fid)) == 3, "只应落前 N 块的向量"


def test_file_under_limit_not_flagged(env):
    tmp, db, embed = env
    p = tmp / "small.txt"
    p.write_text("短文件", encoding="utf-8")
    idx = Indexer(db, embed, chunk_size=500, chunk_overlap=0, max_chunks_per_file=32)
    r = idx.index_file(str(p))
    assert r["oversized"] is False
    assert r["chunks"] == r["total_chunks"] == 1


def test_zero_max_chunks_means_unlimited(env):
    tmp, db, embed = env
    p = tmp / "big2.txt"
    p.write_text("x" * 500, encoding="utf-8")
    idx = Indexer(db, embed, chunk_size=50, chunk_overlap=0, max_chunks_per_file=0)
    r = idx.index_file(str(p))
    assert r["oversized"] is False
    assert r["chunks"] == 10


# ---------------------------------------------------------------- 4. 重试封顶

def _add_failed(db, path: Path, attempts: int, content_hash: str = ""):
    db.upsert_file(
        {"fileId": "", "filePath": str(path), "fileName": path.name, "fileType": "md",
         "fileSize": path.stat().st_size, "createdAt": "", "modifiedAt": "",
         "accessedAt": "", "author": "", "directory": str(path.parent)},
        content_hash, "", status="failed",
    )
    f = db.get_file_by_path(str(path))
    for _ in range(attempts):
        db.mark_attempt_failed(f["file_id"], content_hash)
    return f["file_id"]


def test_mark_attempt_failed_persists_hash_and_counts(env):
    """失败必须落 hash——否则幂等判据永远不中，导致每轮全量重算。"""
    tmp, db, embed = env
    p = tmp / "a.md"
    p.write_text("x", encoding="utf-8")
    fid = _add_failed(db, p, attempts=0, content_hash="")
    n = db.mark_attempt_failed(fid, "deadbeef")
    assert n == 1
    row = db.get_file(fid)
    assert row["content_hash"] == "deadbeef"
    assert row["attempt_count"] == 1


def test_completed_resets_attempt_count(env):
    tmp, db, embed = env
    p = tmp / "b.md"
    p.write_text("x", encoding="utf-8")
    fid = _add_failed(db, p, attempts=2)
    assert db.get_file(fid)["attempt_count"] == 2
    db.set_vectorize_status(fid, "completed")
    assert db.get_file(fid)["attempt_count"] == 0


def test_exhausted_failed_file_is_not_requeued(env):
    """达重试上限且内容未变 → 不重排（否则每轮重灌整个失败集）。"""
    tmp, db, embed = env
    p = tmp / "c.md"
    p.write_text("内容", encoding="utf-8")
    idx = Indexer(db, embed)
    h = idx._content_hash(p)
    _add_failed(db, p, attempts=3, content_hash=h)

    sc = ScanService(_Cfg([str(tmp)], {"max_attempts": 3}), db, idx)
    sc._enqueue_file(str(p))
    assert db.get_file_by_path(str(p))["vectorize_status"] == "failed", \
        "达上限后不得改回 queued"


def test_exhausted_file_with_changed_content_is_requeued(env):
    """用户改了文件（hash 变）→ 重置重试预算并重新排队。"""
    tmp, db, embed = env
    p = tmp / "d.md"
    p.write_text("原内容", encoding="utf-8")
    idx = Indexer(db, embed)
    _add_failed(db, p, attempts=3, content_hash=idx._content_hash(p))
    p.write_text("改过的内容，长度不同", encoding="utf-8")

    sc = ScanService(_Cfg([str(tmp)], {"max_attempts": 3}), db, idx)
    sc._enqueue_file(str(p))
    row = db.get_file_by_path(str(p))
    assert row["vectorize_status"] == "queued"
    assert row["attempt_count"] == 0


def test_failed_file_under_limit_is_still_requeued(env):
    """未达上限 → 照常重排（保留重试能力）。"""
    tmp, db, embed = env
    p = tmp / "e.md"
    p.write_text("内容", encoding="utf-8")
    idx = Indexer(db, embed)
    _add_failed(db, p, attempts=1, content_hash=idx._content_hash(p))

    sc = ScanService(_Cfg([str(tmp)], {"max_attempts": 3}), db, idx)
    sc._enqueue_file(str(p))
    assert db.get_file_by_path(str(p))["vectorize_status"] == "queued"


# ---------------------------------------------------------------- 5. 看门狗

class _SwitchableEmbed:
    def __init__(self):
        self.down = True

    def ping(self) -> bool:
        return not self.down

    def close(self):
        pass


class _SpyScanner:
    def __init__(self, status="done"):
        self.status = status
        self.starts = 0

    def start(self):
        self.starts += 1
        return {"status": "scanning"}


def _pending(db, tmp, status="failed", attempts=0):
    p = tmp / f"p{status}{attempts}.md"
    p.write_text("x", encoding="utf-8")
    db.upsert_file(
        {"fileId": "", "filePath": str(p), "fileName": p.name, "fileType": "md",
         "fileSize": 1, "createdAt": "", "modifiedAt": "", "accessedAt": "",
         "author": "", "directory": str(tmp)},
        "", "", status=status,
    )
    if attempts:
        fid = db.get_file_by_path(str(p))["file_id"]
        for _ in range(attempts):
            db.mark_attempt_failed(fid, "")
    return p


def _wait(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not cond():
        time.sleep(0.01)


def test_cooldown_blocks_second_rebuild(env):
    """冷却期内第二次 down→up 不得再触发重建（打断 OOM 正反馈）。"""
    tmp, db, _fake = env
    _pending(db, tmp, "failed")
    embed = _SwitchableEmbed()   # 可切换 down/up 的替身
    scanner = _SpyScanner()
    rec = EmbedRecovery(embed, db, scanner, interval=0.05, max_attempts=3, cooldown=60.0)

    embed.down = True
    rec.start()
    _wait(lambda: rec.probes >= 2)
    embed.down = False
    _wait(lambda: scanner.starts >= 1)
    assert scanner.starts == 1

    # 再次掉线→恢复，但仍在冷却期内
    embed.down = True
    _wait(lambda: rec.probes >= 6)
    embed.down = False
    _wait(lambda: rec.probes >= 12)
    rec.stop()
    assert scanner.starts == 1, f"冷却期内不得再次重建（实际 {scanner.starts}）"


def test_rebuild_allowed_after_cooldown_elapsed(env):
    """冷却=0 → 第二次 down→up 仍可重建（确认冷却不是把功能关死）。"""
    tmp, db, _fake = env
    _pending(db, tmp, "failed")
    embed = _SwitchableEmbed()
    scanner = _SpyScanner()
    rec = EmbedRecovery(embed, db, scanner, interval=0.05, max_attempts=3, cooldown=0.0)

    embed.down = True
    rec.start()
    _wait(lambda: rec.probes >= 2)
    embed.down = False
    _wait(lambda: scanner.starts >= 1)
    embed.down = True
    _wait(lambda: rec.probes >= 8)
    embed.down = False
    _wait(lambda: scanner.starts >= 2)
    rec.stop()
    assert scanner.starts >= 2


def test_no_trigger_when_all_pending_exhausted(env):
    """待重跑文件全部耗尽重试预算 → 视作无待重跑，不触发。"""
    tmp, db, _fake = env
    _pending(db, tmp, "failed", attempts=3)
    embed = _SwitchableEmbed()
    scanner = _SpyScanner()
    rec = EmbedRecovery(embed, db, scanner, interval=0.05, max_attempts=3, cooldown=0.0)
    assert rec._pending_count() == 0, "达上限的永久失败文件不应计入待重跑"

    embed.down = True
    rec.start()
    _wait(lambda: rec.probes >= 2)
    embed.down = False
    _wait(lambda: rec.probes >= 6)
    rec.stop()
    assert scanner.starts == 0


def test_pending_counts_files_with_budget(env):
    tmp, db, _fake = env
    _pending(db, tmp, "failed", attempts=1)
    _pending(db, tmp, "queued", attempts=0)
    _pending(db, tmp, "failed", attempts=9)   # 超上限，不计
    rec = EmbedRecovery(_SwitchableEmbed(), db, _SpyScanner(), max_attempts=3)
    assert rec._pending_count() == 2
    rec.stop()
