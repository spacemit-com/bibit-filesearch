"""扫描状态契约：**对外 status 必须回答「还有没有活在干」**（2026-09-22）。

用户实测的 bug（复现两次）：

    cp -r /home/user/pdf /home/user/test/      ← 拷一个目录进来
    11:59:04  watcher: 目录补扫完成: /home/user/test/pdf
    11:59:07  indexer: 文件过大，仅索引前 64 块（共 2690 块）: Theartofmulticore.pdf
    11:59:52  watcher: 增量索引 Theartofmulticore.pdf: indexed   ← 49 秒，真的在算
    全程 scan_status = "done"（2 秒粒度采样 45 次，一次都没变过）

界面于是显示「已完成 / 已索引文件 9」，用户以为没扫，必须手点一次「保存并索引」
才翻成「扫描中」。根因：索引有**两条路径**，只有一条写状态——

  scanner 路径：_enqueue_file 写 files.vectorize_status='queued' → worker 消费 → 写 scan_status
  watcher 路径：_backfill_dir / _handle 直调 indexer.index_file()，**不写 queued，不写 scan_status**

而 done 的判据是「队列空 + 遍历线程结束」——watcher 干活时队列本来就是空的 → 判成 done。

本文件锁定契约（docs/scan-status-contract.md §3.2、§5）：

  ① index_file 是三条路径的唯一公共入口 → 一个在飞计数器覆盖全部索引来源；
  ② 计数器必须在**所有**出口归零（含异常）——漏减会让状态永久卡在 scanning；
  ③ progress().status 在 done 之上派生 scanning，当且仅当 queued>0 或有在飞；
  ④ paused / error / idle 不被派生覆盖。
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
from bibit_filesearch.scanner import ScanService, STATUS_DONE, STATUS_SCANNING, STATUS_PAUSED  # noqa: E402
from bibit_filesearch.watcher import Watcher  # noqa: E402


class FakeConfig:
    def __init__(self, index_dirs, excludes=None):
        self.index_dirs = index_dirs
        self.exclude_patterns = excludes or [".git", "__pycache__"]

        class _Scan:
            def get(self, k, d=None):
                return {"io_throttle_ms": 0}.get(k, d)

        self.scan = _Scan()


class SlowEmbed(FakeEmbed):
    """可控制时长的 embed 替身：用来把「正在算」这个窗口撑开，便于观测状态。

    `hold` 是每次 embed_batch 的睡眠秒数。`gate` 为 Event 时改用 gate.wait()
    ——测试可以对计算窗口做精确同步，不靠 sleep 猜时长。
    """

    def __init__(self, hold: float = 0.0, gate: threading.Event | None = None):
        super().__init__()
        self.hold = hold
        self.gate = gate
        self.entered = threading.Semaphore(0)
        self.batches = 0

    def embed_batch(self, texts):
        self.batches += 1
        self.entered.release()
        if self.gate is not None:
            self.gate.wait(timeout=30)
        elif self.hold:
            time.sleep(self.hold)
        return super().embed_batch(texts)


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_status_"))
    db = Database(str(tmp / "test.db"))
    yield tmp, db
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _wait_status(scanner, target, timeout=30):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if scanner.status == target:
            return True
        time.sleep(0.05)
    return scanner.status == target


# ---------------------------------------------------------------- 计数语义

def test_inflight_zero_when_nothing_running(env):
    _, db = env
    indexer = Indexer(db, FakeEmbed())
    assert indexer.inflight_indexes == 0


def test_inflight_counts_during_indexing_and_returns_to_zero(env):
    """在算期间计数为 1，算完归零。"""
    tmp, db = env
    work = tmp / "w"
    work.mkdir()
    f = work / "a.md"
    f.write_text("# 甲\n\n内容甲。\n", encoding="utf-8")

    gate = threading.Event()
    embed = SlowEmbed(gate=gate)
    indexer = Indexer(db, embed)

    t = threading.Thread(target=lambda: indexer.index_file(str(f)), daemon=True)
    t.start()
    assert embed.entered.acquire(timeout=10), "embed 应已被调用（即已进入在飞区间）"
    assert indexer.inflight_indexes == 1, "在算期间必须计到 1"

    gate.set()
    t.join(timeout=30)
    assert not t.is_alive()
    assert indexer.inflight_indexes == 0, "算完必须归零（否则状态永久卡 scanning）"


@pytest.mark.parametrize(
    "make, kind",
    [
        (lambda w: str(w / "不存在.md"), "路径不存在"),
        (lambda w: str(w / "b.unknownext"), "不可索引类型"),
    ],
)
def test_inflight_zero_on_early_return_paths(env, make, kind):
    """not_a_file / unsupported 这类提前 return 也必须归零。"""
    tmp, db = env
    work = tmp / "w"
    work.mkdir()
    (work / "b.unknownext").write_bytes(b"\x00\x01\x02")
    indexer = Indexer(db, FakeEmbed())
    indexer.index_file(make(work))          # 不该抛
    assert indexer.inflight_indexes == 0, f"{kind} 路径漏减"


def test_inflight_zero_on_skipped_path(env):
    """hash 未变的快路径也要归零。"""
    tmp, db = env
    work = tmp / "w"
    work.mkdir()
    f = work / "a.md"
    f.write_text("# 甲\n\n内容甲。\n", encoding="utf-8")
    indexer = Indexer(db, FakeEmbed())
    indexer.index_file(str(f))
    r = indexer.index_file(str(f))
    assert r["status"] == "skipped", f"第二次应走 hash 快路径，实际 {r}"
    assert indexer.inflight_indexes == 0


def test_inflight_zero_on_exception(env):
    """异常路径（读不了的文件）必须归零——这是 finally 的全部意义。"""
    tmp, db = env
    work = tmp / "w"
    work.mkdir()
    f = work / "a.md"
    f.write_text("# 甲\n\n内容甲。\n", encoding="utf-8")
    embed = FakeEmbed()

    def boom(texts):
        raise RuntimeError("embed 炸了")

    embed.embed_batch = boom
    indexer = Indexer(db, embed)
    with pytest.raises(RuntimeError):
        indexer.index_file(str(f))
    assert indexer.inflight_indexes == 0, "异常路径漏减 → 状态会永久卡 scanning"


def test_inflight_counts_are_correct_under_concurrency(env):
    """并发调用后计数必须归零，且期间确实观测到多个在飞。

    ⚠️ 诚实说明：**本用例不构成对 `_inflight_lock` 的变异保护**。CPython 的 GIL 让
    `self._inflight += 1` 在绝大多数调度下表现为原子（字节码之间的切换窗口极窄），
    去掉锁后本用例仍会通过（实测：变异 ④ 不变红）。锁是**防御性**的——真要复现丢计数
    需要精确的线程时序，测不稳。这里守的是别的东西：
      · 并发结束后计数**精确归零**（漏减的最坏后果：状态永久卡 scanning）；
      · 期间峰值 ≥ 2（若计数被写成「布尔标志」或「只在 scanner 路径自增」，峰值会塌到 1）。
    """
    tmp, db = env
    work = tmp / "w"
    work.mkdir()
    files = []
    for i in range(40):
        p = work / f"f{i:02d}.md"
        p.write_text(f"# 文档{i}\n\n内容{i}。\n", encoding="utf-8")
        files.append(str(p))

    gate = threading.Event()
    embed = SlowEmbed(gate=gate)
    indexer = Indexer(db, embed)

    peak = 0
    peak_lock = threading.Lock()

    def watch():
        nonlocal peak
        while not gate.is_set():
            n = indexer.inflight_indexes
            with peak_lock:
                peak = max(peak, n)
            time.sleep(0.001)

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    threads = [threading.Thread(target=lambda p=p: indexer.index_file(p), daemon=True)
               for p in files]
    for t in threads:
        t.start()
    # 等第一个 embed 进来说明真的在跑，再放行
    assert embed.entered.acquire(timeout=10)
    time.sleep(0.3)
    gate.set()
    for t in threads:
        t.join(timeout=60)
    w.join(timeout=5)

    assert indexer.inflight_indexes == 0, "全部结束后必须归零"
    assert peak >= 2, f"并发期间应观测到多个在飞（否则计数不准）：peak={peak}"


# ---------------------------------------------------------------- 状态派生（核心回归）

def test_done_then_watcher_indexing_reports_scanning(env):
    """★ 用户实测的那个 bug：watcher 补扫在算，progress().status 必须是 scanning。

    这是本修复的核心不变式——`done` 必须保证"真的没有活在干"。
    """
    tmp, db = env
    work = tmp / "work"
    work.mkdir()
    (work / "a.md").write_text("# 文档甲\n\n内容甲。\n", encoding="utf-8")

    embed = SlowEmbed()
    indexer = Indexer(db, embed)
    scanner = ScanService(FakeConfig([str(work)]), db, indexer)

    scanner.start()
    assert _wait_status(scanner, "done", timeout=30), \
        f"基线应到 done，实际 {scanner.status}"
    assert scanner.progress()["status"] == "done", "无活在干时就是 done"

    # ——— 模拟用户操作：拷一个新文件进来，由 watcher 路径索引（不经 scanner 队列）———
    big = work / "big.md"
    big.write_text("# 大文档\n\n" + ("内容。" * 4000) + "\n", encoding="utf-8")

    gate = threading.Event()
    indexer.embed = SlowEmbed(gate=gate)      # 换慢 embed，撑开计算窗口
    watcher = Watcher(FakeConfig([str(work)]), db, indexer)

    t = threading.Thread(target=lambda: watcher._backfill_dir(str(work)), daemon=True)
    t.start()
    assert indexer.embed.entered.acquire(timeout=10), "补扫应已进入 embed"

    prog = scanner.progress()
    assert prog["status"] == STATUS_SCANNING, (
        "★ 回归：watcher 增量索引期间必须报 scanning，"
        f"否则界面显示「已完成」而实际在算（实测 49 秒）。实际 {prog['status']}"
    )
    # 该路径不写 queued —— 这正是旧判据失效的原因，用例里也钉住这个前提
    assert prog["vectorizePhase"]["queued"] == 0, \
        "watcher 路径不写 queued（若写了说明实现变了，本用例的前提需重新审视）"
    # 前端据这个字段决定「不画进度」（增量期间没有可信分母）
    assert prog["vectorizePhase"]["inflight"] >= 1, \
        "progress().vectorizePhase.inflight 必须暴露在飞数（前端靠它遮蔽假进度）"
    assert indexer.inflight_indexes >= 1

    gate.set()
    t.join(timeout=60)
    assert not t.is_alive()

    # 算完 → 回 done（前端 2s 轮询内能看到）
    assert _wait_status(scanner, "done", timeout=15) or True
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if scanner.progress()["status"] == "done":
            break
        time.sleep(0.05)
    assert scanner.progress()["status"] == "done", "算完必须自动回 done，不能钉死 scanning"


def test_queued_alone_forces_scanning_even_when_status_done(env):
    """queued>0 也要报 scanning——覆盖 scanner 路径的启动窗口（worker 还没跑到）。"""
    tmp, db = env
    work = tmp / "work"
    work.mkdir()
    f = work / "a.md"
    f.write_text("# 甲\n\n内容甲。\n", encoding="utf-8")

    indexer = Indexer(db, FakeEmbed())
    scanner = ScanService(FakeConfig([str(work)]), db, indexer)
    scanner.start()
    assert _wait_status(scanner, "done", timeout=30)

    # 直接把库里改回 queued，模拟「已入队但 worker 尚未消费」
    fid = db.get_file_by_path(str(f))["file_id"]
    db.set_vectorize_status(fid, "queued")
    assert scanner.progress()["vectorizePhase"]["queued"] == 1
    assert scanner.progress()["status"] == STATUS_SCANNING, \
        "库里有排队文件就不该报 done"


def test_paused_is_not_overridden_by_inflight(env):
    """暂停优先：**有在飞时**也必须保持 paused（派生只作用于 done）。

    构造方式：起一次扫描，等某个文件真的进了 embed（此时 inflight ≥ 1），
    再按暂停 —— 于是"有活在干（inflight>0）"与"用户暂停"同时成立。
    变异点：把派生判据从 `status == DONE` 放宽成「非 idle 即派生」，本用例会变红。
    """
    tmp, db = env
    work = tmp / "work"
    work.mkdir()
    for i in range(30):
        (work / f"f{i:02d}.md").write_text(f"# 文档{i}\n\n内容{i}。\n", encoding="utf-8")

    gate = threading.Event()
    embed = SlowEmbed(gate=gate)
    indexer = Indexer(db, embed)
    scanner = ScanService(FakeConfig([str(work)]), db, indexer)
    scanner.start()
    assert embed.entered.acquire(timeout=30), "扫描应进入 embed（把在飞窗口撑开）"
    r = scanner.pause()
    assert r["status"] == STATUS_PAUSED, f"暂停应生效，实际 {r}"
    assert indexer.inflight_indexes >= 1, "此刻确实有在飞（才有资格验证覆盖）"
    try:
        assert scanner.progress()["status"] == STATUS_PAUSED, \
            "派生不得覆盖 paused（用户按了暂停，界面就不该显示扫描中）"
    finally:
        gate.set()
        # 等 worker 把在飞清干净，避免影响后续（本用例独立 db，仅求干净退出）
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and indexer.inflight_indexes > 0:
            time.sleep(0.05)


def test_idle_is_not_derived_to_scanning(env):
    """服务刚起、尚未扫过：status=idle，库里可能有 queued（重启续跑），但 idle 语义是
    「还没扫过」，派生只作用于 done，不该把 idle 变成 scanning（保持既有行为）。"""
    tmp, db = env
    indexer = Indexer(db, FakeEmbed())
    scanner = ScanService(FakeConfig([str(tmp / "nothing")]), db, indexer)
    assert scanner.progress()["status"] == "idle"


def test_error_status_is_not_overridden(env):
    """error 不复位为 scanning（只派生 done）。"""
    tmp, db = env
    indexer = Indexer(db, FakeEmbed())
    scanner = ScanService(FakeConfig([str(tmp)]), db, indexer)
    with scanner._lock:
        scanner._status = "error"
    assert scanner.progress()["status"] == "error"
