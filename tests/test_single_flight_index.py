"""同路径单飞（F8）：两条索引路径撞车时，同一个文件只能被解析一次（2026-09-22）。

真机实测的 bug：
    cp -r /home/user/pdf /home/user/test/     ← 拷目录进来
    14:41:38,753  bibit.indexer: 文件过大…  ← watcher 路径第 1 次 parse
    14:41:39,541  bibit.scanner: 全盘扫描：共 13 个文件待处理   ← 用户点了「保存并索引」
    14:41:42,378  bibit.indexer: 文件过大…  ← scanner 路径第 2 次 parse（4 秒后）
    端到端 50s → 120s（vectorize_parallelism=1，同一份块被算两遍）
    「文件过大」这行**每次 parse 恰好一条**，所以它天然是解析计数器。

根因：两条路径都在 index 开始之前查库问「这文件在处理吗」，而库里那行是在
`embed_batch` **之后**才写的（`upsert_file`）→ 整个 embed 窗口里文件在库中
**不存在** → 另一条路径的 hash 快路径（要求 status=='completed'）判不出「在算」。

本文件锁定的契约（Indexer.index_file）：
  ① 同一路径在飞期间，第二条请求**不解析**；
  ② 它**等赢家收尾并复用赢家的返回值**（不是空手返回）——否则「跳过重复」会
     退化成「跳过更新」：赢家算的是旧内容，落库旧内容而变更事件已被丢弃；
  ③ 赢家异常时把异常**重新抛给等待者**（调用方看到的行为与单独调用一致）；
  ④ 单飞只在**同路径**生效，不同路径互不影响（各自的并发度不变）；
  ⑤ 计数与占位同生共死：窗口内 inflight 计 1（不是 2），出窗口归零。
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_embed import FakeEmbed  # noqa: E402
from bibit_filesearch.db import Database  # noqa: E402
from bibit_filesearch.indexer import Indexer  # noqa: E402


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_singleflight_"))
    db = Database(str(tmp / "test.db"))
    yield tmp, db
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


class CountingEmbed(FakeEmbed):
    """记录 embed_batch 的入参块数，可闸门停住——用来把索引窗口撑开。"""

    def __init__(self, gate: threading.Event | None = None):
        super().__init__()
        self.gate = gate
        self.entered = threading.Semaphore(0)
        self.calls: list[int] = []          # 每次 embed_batch 的输入块数
        self._lock = threading.Lock()

    def embed_batch(self, texts):
        with self._lock:
            self.calls.append(len(texts))
        self.entered.release()
        if self.gate is not None:
            self.gate.wait(timeout=30)
        return super().embed_batch(texts)


class ParseCounter:
    """给 Indexer 的解析函数打桩，数「这个文件被解析了几次」。

    为什么按 parse 计数而不是按 embed：embed 是两条路径的公共下游，而 parse 是
    「解析一遍」这件事本身；真机上的判据也正是它（「文件过大」那条日志在
    parse 之后、embed 之前打出）。
    """

    def __init__(self, monkeypatch, indexer):
        self.calls: list[str] = []
        self._lock = threading.Lock()
        real = indexer._index_file.__func__  # 不替换，只在外面包一层计数

        def spy(path):
            with self._lock:
                self.calls.append(path)
            return real(indexer, path)

        monkeypatch.setattr(indexer, "_index_file", spy)
        self.indexer = indexer


def _big_file(root: Path, name: str = "big.md", n: int = 4000) -> str:
    p = root / name
    p.write_text("# 大文档\n\n" + ("内容。" * n) + "\n", encoding="utf-8")
    return str(p)


# ------------------------------------------------------------ 核心：两条路径撞车

def test_second_request_does_not_parse_again(env, monkeypatch):
    """★ 核心回归：同一路径在飞时，第二条请求必须复用结果而不是再解析一遍。

    这正是真机上「同一 PDF 被 parse 两次、耗时翻倍」的那条。
    """
    tmp, db = env
    f = _big_file(tmp)

    gate = threading.Event()
    embed = CountingEmbed(gate=gate)
    indexer = Indexer(db, embed)
    spy = ParseCounter(monkeypatch, indexer)

    first: list[dict] = []
    t1 = threading.Thread(target=lambda: first.append(indexer.index_file(f)), daemon=True)
    t1.start()
    assert embed.entered.acquire(timeout=10), "第一次应已进入 embed"

    second: list[dict] = []
    t2 = threading.Thread(target=lambda: second.append(indexer.index_file(f)), daemon=True)
    t2.start()
    # 给它足够时间"如果它想解析的话"就该走到 embed —— 断言它没走
    time.sleep(0.5)
    assert len(embed.calls) == 1, \
        f"★ 回归：第二条请求不应再次 embed（实测 bug 就是这里被算了第二遍）：{embed.calls}"
    assert len(spy.calls) == 1, f"★ 回归：同一路径只该被解析一次，实际 {spy.calls}"

    gate.set()
    t1.join(timeout=60)
    t2.join(timeout=60)
    assert not t1.is_alive() and not t2.is_alive()

    # ② 等待者拿到的是赢家的结果（不是空手返回、不是自己再算一遍）
    assert first and second
    assert second[0]["status"] == first[0]["status"], \
        f"等待者应复用赢家的结果：first={first[0]} second={second[0]}"
    assert second[0].get("file_id") == first[0].get("file_id")
    assert len(embed.calls) == 1, f"全程只该 embed 一次：{embed.calls}"

    # ⑤ 计数与占位同生共死
    assert indexer.inflight_indexes == 0
    assert len(indexer._active_paths) == 0


def test_waiter_sees_winner_result_not_stale_skip(env, monkeypatch):
    """★ 关键语义：等待者拿到的必须是**赢家算出来的**结果。

    若实现改成「输的一方直接返回 skipped（=内容没变）」，等待者会把「另一条路径
    正在算」误报成「已是最新」；而赢家可能刚发现内容变了。本用例钉住这一点。
    """
    tmp, db = env
    f = tmp / "doc.md"
    f.write_text("# 旧\n\n旧内容。\n", encoding="utf-8")
    indexer = Indexer(db, FakeEmbed())
    indexer.index_file(str(f))                       # 先正常索引一次 → completed

    f.write_text("# 新\n\n新内容，变了。\n", encoding="utf-8")   # 内容变更

    gate = threading.Event()
    indexer.embed = CountingEmbed(gate=gate)
    results: list[dict] = []
    threads = [threading.Thread(target=lambda: results.append(indexer.index_file(str(f))), daemon=True)
               for _ in range(2)]
    for t in threads:
        t.start()
    assert indexer.embed.entered.acquire(timeout=10)
    time.sleep(0.3)
    gate.set()
    for t in threads:
        t.join(timeout=60)

    assert len(results) == 2
    statuses = {r["status"] for r in results}
    assert statuses == {"updated"}, \
        f"两条路径的结论必须一致且反映真实变更，实际 {[r['status'] for r in results]}"
    row = db.get_file_by_path(str(f))
    assert row["vectorize_status"] == "completed"

    # 库里必须是**新**内容：若等待者提前返回 skipped、赢家又读到旧内容，
    # 这里就会看到旧正文（「跳过重复」退化成「跳过更新」）
    chunks = db.get_chunks(row["file_id"])
    assert any("新内容" in c["text"] for c in chunks), "库里必须落新内容"
    assert not any(c["text"].strip() == "旧内容。" for c in chunks), "不该残留旧块"


def test_winner_exception_is_re_raised_to_waiter(env, monkeypatch):
    """③ 赢家抛异常 → 等待者也要看到异常（调用方行为与单独调用一致）。"""
    tmp, db = env
    f = _big_file(tmp)

    gate = threading.Event()
    embed = CountingEmbed(gate=gate)
    indexer = Indexer(db, embed)

    boom = RuntimeError("embed 炸了")
    real_embed_batch = embed.embed_batch

    def exploding(texts):
        embed.entered.release()
        gate.wait(timeout=30)
        raise boom

    embed.embed_batch = exploding

    errs: list[BaseException] = []

    def call():
        try:
            indexer.index_file(f)
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    t1 = threading.Thread(target=call, daemon=True)
    t1.start()
    assert embed.entered.acquire(timeout=10)
    t2 = threading.Thread(target=call, daemon=True)
    t2.start()
    time.sleep(0.3)
    gate.set()
    t1.join(timeout=60)
    t2.join(timeout=60)

    assert len(errs) == 2, f"两条路径都应看到异常，实际 {errs}"
    assert all(isinstance(e, RuntimeError) and "embed 炸了" in str(e) for e in errs), errs
    assert indexer.inflight_indexes == 0, "异常后计数必须归零"
    assert len(indexer._active_paths) == 0, "异常后占位必须摘掉"
    del real_embed_batch, boom


# ------------------------------------------------------------ ④ 不同路径互不影响

def test_different_paths_are_not_serialized(env):
    """单飞只作用于**同一路径**：不同文件仍可并发（不引入全局锁）。"""
    tmp, db = env
    files = [_big_file(tmp, f"f{i}.md", 500) for i in range(4)]

    gate = threading.Event()
    embed = CountingEmbed(gate=gate)
    indexer = Indexer(db, embed)

    threads = [threading.Thread(target=lambda p=p: indexer.index_file(p), daemon=True)
               for p in files]
    for t in threads:
        t.start()
    for _ in range(4):
        assert embed.entered.acquire(timeout=10), "4 个不同文件都应各自进入 embed"
    # 峰值在飞应为 4（若单飞被写成全局串行，这里只能是 1）
    peak = 0
    for _ in range(20):
        peak = max(peak, indexer.inflight_indexes)
        if peak >= 4:
            break
        time.sleep(0.02)
    assert peak >= 4, f"不同路径不应互相串行化：peak={peak}"

    gate.set()
    for t in threads:
        t.join(timeout=60)
    assert indexer.inflight_indexes == 0
    assert len(indexer._active_paths) == 0


# ------------------------------------------------------------ ⑤ 计数语义

def test_inflight_counts_one_during_duplicate_window(env):
    """窗口内 inflight 计 1 而不是 2 —— 否则「待向量化」格与状态派生都会被算大。"""
    tmp, db = env
    f = _big_file(tmp)

    gate = threading.Event()
    embed = CountingEmbed(gate=gate)
    indexer = Indexer(db, embed)

    t1 = threading.Thread(target=lambda: indexer.index_file(f), daemon=True)
    t1.start()
    assert embed.entered.acquire(timeout=10)
    t2 = threading.Thread(target=lambda: indexer.index_file(f), daemon=True)
    t2.start()
    time.sleep(0.3)
    assert indexer.inflight_indexes == 1, \
        f"同一文件被两条路径请求，在飞数仍应为 1，实际 {indexer.inflight_indexes}"
    gate.set()
    t1.join(timeout=60)
    t2.join(timeout=60)
    assert indexer.inflight_indexes == 0


def test_wait_timeout_falls_back_to_inflight_status(env):
    """等待上限兜底：赢家永不收尾时，等待者返回 inflight 而不是永久挂住。"""
    tmp, db = env
    f = _big_file(tmp)

    gate = threading.Event()          # 永不 set
    embed = CountingEmbed(gate=gate)
    indexer = Indexer(db, embed)
    indexer.duplicate_wait_seconds = 1  # 缩短，便于测

    t1 = threading.Thread(target=lambda: indexer.index_file(f), daemon=True)
    t1.start()
    assert embed.entered.acquire(timeout=10)

    r = indexer.index_file(f)          # 本线程即等待者
    assert r["status"] == "inflight" and r["reason"] == "waited_too_long", r

    gate.set()
    t1.join(timeout=60)
    assert indexer.inflight_indexes == 0


def test_late_request_after_release_is_not_served_stale(env, monkeypatch):
    """★ 顺序契约：**先摘牌、再放行**（用「迟到请求是否被旧结果顶替」来量）。

    含义：赢家收尾时若先 `done.set()` 再把它从在飞集合摘掉，那么在 set 与摘牌之间
    到达的新请求会看到「牌还在 + 事件已置位」→ wait() 立刻返回 → **拿到上一轮的
    结果**，而那个结果对应的是旧内容。于是「跳过重复解析」退化成「跳过更新」：
    用户改了文件、事件也到了，但库里仍是旧内容。

    构造（确定性）：把 `Event.set()` 打桩 —— 在**真正置位之后**从**另一个线程**发一次
    新请求（文件在这之间已被改写）。正确顺序下，这个新请求会发现集合里没有它 →
    自己解析一遍；错误顺序下它会复用赢家的（过期）结果、连解析都不发生。

    判据：`parse_file` 调用次数。正确顺序 = 2（赢家 1 次 + 迟到请求 1 次）；
    错误顺序 = 1（迟到请求被旧结果顶替）。**这是行为判据，不是对实现细节的断言。**
    """
    tmp, db = env
    f = tmp / "doc.md"
    f.write_text("# 第一版\n\n第一版内容。\n", encoding="utf-8")

    indexer = Indexer(db, FakeEmbed())
    indexer.index_file(str(f))                    # 先建好记录 → 之后才是 updated 而非 indexed
    spy = ParseCounter(monkeypatch, indexer)
    f.write_text("# 改过\n\n改过的内容。\n", encoding="utf-8")

    import bibit_filesearch.indexer as indexer_mod

    real_event_cls = indexer_mod.threading.Event
    state: dict = {"fired": False, "late_status": None, "err": None}

    class HookedEvent(real_event_cls):
        """set() 置位之后，从另一线程发一次迟到请求（此刻文件已被改写）。"""

        def set(self):
            super().set()
            if state["fired"]:
                return
            state["fired"] = True
            f.write_text("# 第二版\n\n第二版内容，换了。\n", encoding="utf-8")
            def late():
                try:
                    state["late_status"] = indexer.index_file(str(f))["status"]
                except Exception as e:  # noqa: BLE001
                    state["err"] = e

            t = threading.Thread(target=late, daemon=True)
            t.start()
            t.join(timeout=30)
            assert not t.is_alive(), "迟到请求不得挂住"

    monkeypatch.setattr(indexer_mod.threading, "Event", HookedEvent)

    gate = real_event_cls()
    indexer.embed = CountingEmbed(gate=gate)

    t = threading.Thread(target=lambda: indexer.index_file(str(f)), daemon=True)
    t.start()
    assert indexer.embed.entered.acquire(timeout=10)
    gate.set()                                    # 赢家收尾 → 触发 set() → 触发迟到请求
    t.join(timeout=60)
    assert not t.is_alive()

    assert state["fired"], "钩子未触发 —— 这个顺序契约根本没被验证到"
    assert state["err"] is None, f"迟到请求不该抛异常：{state['err']}"
    assert len(spy.calls) == 2, (
        f"★ 顺序契约：放行等待者之前必须先摘牌。实测 parse 次数={len(spy.calls)}"
        f"（期望 2：赢家 1 + 迟到请求 1）。若为 1，说明迟到请求被上一轮的过期结果顶替了。"
    )
    assert state["late_status"] == "updated", \
        f"迟到请求必须拿到反映新内容的结果，实际 {state['late_status']}"
    row = db.get_file_by_path(str(f))
    chunks = db.get_chunks(row["file_id"])
    assert any("第二版内容" in c["text"] for c in chunks), "库里必须是新内容"
    assert indexer.inflight_indexes == 0
    assert len(indexer._active_paths) == 0


# --------------------------------------------------- 兜底分支：scanner worker 还原状态

def test_worker_restores_status_when_single_flight_returns_no_result(env):
    """★ 兜底分支（变异 ⑥）：worker 拿到 "inflight" 时必须**还原状态**。

    正常路径下 worker 会等到赢家的结果（不会走这里）；只有「等超时 / 同线程重入」才会。
    若不还原，该行会被永久钉在 `vectorizing` —— 既不重试也不修正，界面「待向量化」
    永远显示 1，直到下次启动 `_recover_interrupted` 才被复位。
    """
    from fake_embed import FakeEmbed as _FakeEmbed
    from bibit_filesearch.scanner import ScanService

    class FakeConfig:
        index_dirs = None
        exclude_patterns = [".git", "__pycache__"]
        max_depth = 0

        class _Scan:
            def get(self, k, d=None):
                return {"io_throttle_ms": 0, "max_attempts": 3}.get(k, d)

        scan = _Scan()

    tmp, db = env
    work = tmp / "w"
    work.mkdir()
    f = work / "a.md"
    f.write_text("# 甲\n\n内容甲。\n", encoding="utf-8")

    indexer = Indexer(db, _FakeEmbed())
    cfg = FakeConfig()
    cfg.index_dirs = [str(work)]
    scanner = ScanService(cfg, db, indexer)

    real = indexer.index_file
    calls = {"n": 0}
    seen_status: list[str] = []

    def flaky_first(path):
        calls["n"] += 1
        if calls["n"] == 1:
            # 第一次：模拟"等超时"——worker 必须自己把状态还原，否则该行钉死在 vectorizing
            seen_status.append(db.get_file_by_path(path)["vectorize_status"])
            return {"status": "inflight", "reason": "waited_too_long", "path": path}
        return real(path)

    indexer.index_file = flaky_first
    scanner.start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row = db.get_file_by_path(str(f))
        if row and row["vectorize_status"] == "completed":
            break
        time.sleep(0.05)

    row = db.get_file_by_path(str(f))
    assert row is not None
    assert row["vectorize_status"] != "vectorizing", \
        "★ 变异 ⑥：worker 拿到 inflight 后必须还原状态，否则该行永久卡 vectorizing"
    assert row["vectorize_status"] == "completed", f"兜底后应能重新排队并算完：{row['vectorize_status']}"
    assert calls["n"] >= 2, "还原成 queued 后必须被重新取出处理"
    chunks = db.get_chunks(row["file_id"])
    assert chunks and any("内容甲" in c["text"] for c in chunks), "最终必须真的入库"
