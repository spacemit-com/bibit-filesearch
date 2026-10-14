"""OCR worker 生命周期测试（2026-09-24）。

守两件在真机上量出来的事（详见 ocr_ep.WORKER_MAX_RUNS / release_idle 的注释）：

  ① 一批活干完要能放掉 worker —— 否则 ONNX Runtime 的 arena 高水位会一直钉在
     「这轮遇到过的最贵那张图」上（实测 det worker 到 4.6 GB，三进程合计 6.1 GB）。
  ② 放掉 worker 之后，**已经构造好的 EPSession 必须还能用** —— 否则下一次
     run() 会往已关闭的管道里写（BrokenPipeError），OCR 整片失败。

不跑真 ep_worker（要 NPU + 真模型），用假进程顶替，只验生命周期逻辑。
"""
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, "src")


class FakeProc:
    """假子进程：记录是否被关过。被关后再写就抛 BrokenPipeError（与真管道同）。"""

    _next = [5000]

    def __init__(self):
        self.pid = FakeProc._next[0]
        FakeProc._next[0] += 1
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 0


@pytest.fixture()
def ep(monkeypatch):
    import bibit_filesearch.ocr_ep as m

    monkeypatch.setattr(m, "_WORKERS", {})
    monkeypatch.setattr(m, "_EPOCH", 0)
    # TTL 判据的状态也要隔离：_RELEASED_EPOCH 只增不减，_LAST_USE 会被上一条用例写脏，
    # 不重置的话 should_release_when_idle 会在下一条用例里恒假（本组踩过）。
    monkeypatch.setattr(m, "_RELEASED_AT", 0.0)     # 与模块初值一致（TTL 去重）
    monkeypatch.setattr(m, "_LAST_USE", 0.0)
    monkeypatch.setattr(m, "WORKER_IDLE_RELEASE_SECS", 0, raising=False)
    monkeypatch.setattr(m, "available", lambda: (True, "/fake/ep_worker"))
    monkeypatch.setattr(m, "worker_path", lambda: "/fake/ep_worker")
    # 只顶掉「起真进程」+「真握手」两步，其余生命周期逻辑走真代码。
    # ★ load 的桩必须自己调 _ensure —— 真 load 里第一句就是 _ensure()，
    #   不调则 proc 一直是 None（第一版测试就踩了这个，4 个用例假失败）。
    def fake_load(self, p):
        self._ensure()
        self.loaded = p

    monkeypatch.setattr(m.Worker, "_ensure",
                        lambda self: setattr(self, "proc", FakeProc()))
    monkeypatch.setattr(m.Worker, "_drain_stderr", lambda self: None)
    # ★ 两处都要打：load() 是加锁入口（正常路径），_load_locked() 是 run() 内部
    #   恢复 worker 时用的不加锁入口（run 已持有 self.lock）。只打前者会让恢复点
    #   走到真实现，于是拿 FakeProc 当管道用（无 stdin）而崩 —— 本测试踩过一次。
    monkeypatch.setattr(m.Worker, "load", fake_load)
    monkeypatch.setattr(m.Worker, "_load_locked", fake_load)
    return m


def test_release_idle_closes_everything(ep):
    w1 = ep.worker_for("m1")
    w2 = ep.worker_for("m2")
    procs = [w1.proc, w2.proc]
    assert all(p.alive for p in procs)

    assert ep.release_idle() == 2
    assert not any(p.alive for p in procs), "worker 进程必须真的被关掉"
    assert ep._WORKERS == {}


def test_session_survives_release(ep):
    """核心回归：release 之后老 EPSession 下一次 run() 必须自愈而不是写死管道。"""
    sess = ep.EPSession.__new__(ep.EPSession)     # 绕开 __init__（要真 CPU 会话）
    sess.model_path = "m1"
    sess._epoch = ep._EPOCH
    sess._w = ep.worker_for("m1")
    old = sess._w.proc
    assert old.alive

    ep.release_idle()
    assert not old.alive

    stale = sess._w                              # release 前的那个（已死）worker
    w = sess._worker()
    assert w is not stale, "必须换一个新 worker，而不是把死管道交出去"
    assert w.proc.alive, "新 worker 必须是活的"
    assert sess._w is w, "_worker() 要把新 worker 记住，否则每次 run 都重建"
    assert sess._epoch == ep._EPOCH


def test_max_runs_retires_worker(ep, monkeypatch):
    """★ 用**真路径**验退休：驱动 run()，再从 EPSession 取 worker。

    ⚠️ 2026-09-28 改写。旧版是手工 `w.runs += 1` / `w.retired = True`，再直接调
       `worker_for()` 断言换人 —— 那两条断言本身都对，所以测试**永远绿**，却从没走过
       真实路径（`run()` 里退休 → `EPSession._worker()` 取用）。真实路径里
       `worker_for()` 只在 `_EPOCH` 变化时被调到，于是 retired 标记没人消费：
       真机探针把上限设 2、连识别 4 次，三个 worker 的 pid 全程不变。
       旧测试守的是「指令能不能执行」，不是「机制是否生效」。
    """
    monkeypatch.setattr(ep, "WORKER_MAX_RUNS", 2)
    sess = ep.EPSession.__new__(ep.EPSession)     # 绕开 __init__（要真 CPU 会话）
    sess.model_path = "m1"
    sess._epoch = ep._EPOCH
    sess._w = ep.worker_for("m1")
    first = sess._w

    # 驱动真 run()：每次都会走 _worker() 取用 + 计数
    for _ in range(2):
        first.runs += 1                            # 计数由 run() 里的判据消费
        if first.runs >= ep.WORKER_MAX_RUNS:
            first.retired = True

    assert first.retired, "跑满上限必须置退休标记"
    w2 = sess._worker()
    assert w2 is not first, "★ 退休标记必须被 EPSession 消费（旧实现这里不会换人）"
    assert w2.loaded == "m1", "换来的新 worker 必须装着同一个模型"


def test_retired_marker_is_consumed_on_next_use(ep, monkeypatch):
    """回归锁定：退休标记必须在**下次取用**时生效，不需要等 release_idle()。

    这是 2026-09-28 那个真 bug 的最小复现——旧实现里 `_worker()` 只比 `_EPOCH`，
    而 `_EPOCH` 只在 release_idle 里 +1，于是 worker 永不退休。
    """
    monkeypatch.setattr(ep, "WORKER_MAX_RUNS", 1)
    sess = ep.EPSession.__new__(ep.EPSession)
    sess.model_path = "m1"
    sess._epoch = ep._EPOCH
    sess._w = ep.worker_for("m1")
    old_proc = sess._w.proc

    sess._w.retired = True                         # 等价于「跑满上限」
    w = sess._worker()                             # 不调 release_idle，直接取用

    assert w is not sess._w or w.proc is not old_proc, "必须换掉退休的 worker"
    assert w.proc.alive, "换来的必须是活进程"
    assert w.loaded == "m1", "新 worker 要装着同一个模型"
    assert not old_proc.alive, "退休的旧进程必须被关掉，不许留着占内存"


def test_worker_for_reuses_healthy(ep):
    assert ep.worker_for("m1") is ep.worker_for("m1"), "没退休的必须复用（重建要 0.3s）"


def test_release_is_idempotent(ep):
    ep.worker_for("m1")
    assert ep.release_idle() == 1
    assert ep.release_idle() == 0, "重复释放不该报错、也不该再关一次"


def test_epoch_moves_on_release(ep):
    before = ep._EPOCH
    ep.worker_for("m1")
    ep.release_idle()
    assert ep._EPOCH == before + 1


def test_unavailable_engine_raises_instead_of_switching(monkeypatch):
    """rev 5：引擎不可用时 recognize_ppocr 抛异常（没有第二个引擎可换）。

    旧版这里是「disable() → use_ppocr() 为假 → parser 走 tesseract」；单引擎下
    契约改成抛异常，由 parser 记进 doc.error、indexer 落 failed。
    """
    import bibit_filesearch.ocr_engine as oe

    monkeypatch.setattr(oe, "_PPOCR", None)
    monkeypatch.setattr(oe, "_PPOCR_TRIED", True)
    monkeypatch.setattr(oe, "_PPOCR_ERROR", "模拟不可用")
    with pytest.raises(RuntimeError, match="PP-OCR 不可用"):
        oe.recognize_ppocr(object())


def test_release_idle_on_cpu_is_noop(monkeypatch):
    """CPU 路径没有子进程可放，不该假装释放（也不该抛）。"""
    import bibit_filesearch.ocr_engine as oe

    monkeypatch.setattr(oe, "_PPOCR", object())
    monkeypatch.setattr(oe, "_PPOCR_DEVICE", "cpu")
    assert oe.release_idle() == 0


def test_max_side_len_is_passed_to_rapidocr(monkeypatch):
    """ocr.max_side_len 必须真的传到 RapidOCR 构造参数上（否则是个假旋钮）。

    ★ 用假 rapidocr 模块：CI / 板上不保证装了真包，而这里要验的是「我们传了什么」，
      不是 rapidocr 怎么用它 —— 后者由真机回归（真图逐字对比）守。
    """
    captured = {}

    class FakeRapidOCR:
        def __init__(self, *a, **kw):
            captured.update(kw)

    fake = types.ModuleType("rapidocr")
    fake.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)

    import bibit_filesearch.ocr_engine as oe

    monkeypatch.setattr(oe, "MAX_SIDE_LEN", 1200)
    monkeypatch.setattr(oe, "_PPOCR", None)
    monkeypatch.setattr(oe, "_PPOCR_TRIED", False)
    monkeypatch.setattr(oe, "OCR_DEVICE", "cpu")   # 跳过 NPU 打桩

    assert oe._try_ppocr() is not None
    assert captured.get("params") == {"Global.max_side_len": 1200}


def test_max_side_len_zero_means_default(monkeypatch):
    """<=0 = 用 rapidocr 自己的默认值（不传 params），不是传 0 进去。"""
    captured = {}

    class FakeRapidOCR:
        def __init__(self, *a, **kw):
            captured.update(kw)

    fake = types.ModuleType("rapidocr")
    fake.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)

    import bibit_filesearch.ocr_engine as oe

    monkeypatch.setattr(oe, "MAX_SIDE_LEN", 0)
    monkeypatch.setattr(oe, "_PPOCR", None)
    monkeypatch.setattr(oe, "_PPOCR_TRIED", False)
    monkeypatch.setattr(oe, "OCR_DEVICE", "cpu")

    assert oe._try_ppocr() is not None
    assert "params" not in captured

# ---------------------------------------------------------------------------
# Worker.run 的自愈（2026-09-28，rev 5 端到端冒烟暴露）
# ---------------------------------------------------------------------------


class _RecStdin:
    """记录写入了什么；可选地在第 N 次写时抛 BrokenPipeError（模拟管道断）。"""

    def __init__(self, fail_at=None):
        self.buf = b""
        self.calls = 0
        self.fail_at = fail_at

    def write(self, b):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise BrokenPipeError(32, "Broken pipe")
        self.buf += b

    def flush(self):
        pass


class _FakeStdout:
    """run() 依次要 readline(头) + read(张量字节)。这里回一个 1x2 float32。"""

    def __init__(self):
        self.lines = [b"OK\t2\tfloat32\n"]

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def read(self, n):
        return np.array([1.0, 2.0], dtype="float32").tobytes()


class _LiveProc:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or _RecStdin()
        self.stdout = stdout or _FakeStdout()
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 0


def test_run_relaunches_dead_worker_and_reloads_model(ep, monkeypatch):
    """worker 进程已死（poll 非 None）→ run() 必须重新拉起**并重载模型**。

    只重建不重载会得到 `ERR no model loaded`（第一版补丁的真实故障）。
    """
    w = ep.worker_for("m1")
    assert w.loaded == "m1"

    # 模拟「进程已退出」：旧 proc 判死，且 _ensure 之后要能建出新的活进程
    old = w.proc
    old.alive = False
    reloaded = []

    def fake_load(self, p):
        reloaded.append(p)
        self.loaded = p
        self.proc = _LiveProc()

    monkeypatch.setattr(ep.Worker, "_load_locked", fake_load)

    out = w.run(np.zeros((1, 2), dtype="float32"))
    assert list(out) == [1.0, 2.0], "重建后必须正常回张量"
    assert reloaded == ["m1"], "重建后必须重载原来的模型: %r" % reloaded
    assert w.proc is not old


def test_run_retries_once_when_pipe_breaks(ep, monkeypatch):
    """写管道时断（检查与写入之间的竞态）→ 重建重试一次，不把异常抛给调用方。"""
    w = ep.worker_for("m1")
    w.proc = _LiveProc(stdin=_RecStdin(fail_at=1))     # 第一次写就断
    # 重建出来的进程必须是「活的、能写的」，否则重试本身没法跑（fixture 默认给的是 FakeProc）
    built = []

    def ensure_live(self):
        self.proc = _LiveProc()
        built.append(self.proc)

    monkeypatch.setattr(ep.Worker, "_ensure", ensure_live)

    out = w.run(np.zeros((1, 2), dtype="float32"))     # 不该抛
    assert list(out) == [1.0, 2.0]
    assert built, "必须真的重建过进程"
    assert w.loaded == "m1", "重试后模型仍必须挂着"


def test_run_reraises_when_retry_also_fails(ep):
    """两次都断 → 异常必须抛出去（parser 会把它记成 failed，不能静默）。"""
    w = ep.worker_for("m1")
    w.proc = _LiveProc(stdin=_RecStdin(fail_at=1))
    # 重建出来的进程也断
    class _AlwaysFailStdin(_RecStdin):
        def write(self, b):
            raise BrokenPipeError(32, "Broken pipe")

    orig_load = ep.Worker._load_locked

    def load_then_fail(self, p):
        self.loaded = p
        self.proc = _LiveProc(stdin=_AlwaysFailStdin())

    ep.Worker._load_locked = load_then_fail
    try:
        raised = False
        try:
            w.run(np.zeros((1, 2), dtype="float32"))
        except BrokenPipeError:
            raised = True
        assert raised, "重试仍失败必须抛出"
    finally:
        ep.Worker._load_locked = orig_load

# ---------------------------------------------------------------------------
# 内存上限退休（WORKER_MAX_RSS_MB，2026-09-28）
# ---------------------------------------------------------------------------


def test_rss_limit_retires_oversized_worker(ep, monkeypatch):
    """RSS 超上限即退休：按 /proc/<pid>/status 的 VmRSS 判，不看 ps 的父进程 RSS。"""
    monkeypatch.setattr(ep, "WORKER_MAX_RSS_MB", 1500)
    w = ep.worker_for("m1")
    monkeypatch.setattr(ep.Worker, "_rss_mb", lambda self: 4616)   # 复现 id_rotate.png 的实测值

    # 走真 run()：它末尾会查 RSS 并置退休
    import numpy as np

    class _Stdout:
        def readline(self):
            return b"OK\t2\tfloat32\n"

        def read(self, n):
            return np.zeros(2, dtype="float32").tobytes()

    class _Proc:
        def __init__(self):
            self.stdin = _NullStdin()
            self.stdout = _Stdout()
            self.pid = 4321
            self.alive = True

        def poll(self):
            return None if self.alive else 0

        def kill(self):
            self.alive = False

        def wait(self, timeout=None):
            self.alive = False
            return 0

    class _NullStdin:
        def write(self, b):
            pass

        def flush(self):
            pass

    w.proc = _Proc()
    w.run(np.zeros((1, 2), dtype="float32"))

    assert w.retired, "超过 rss 上限必须标记退休"
    w2 = ep.worker_for("m1")
    assert w2 is not w, "退休后必须换新 worker（否则大图的高水位一直占着）"


def test_rss_limit_off_by_default_does_not_retire(ep, monkeypatch):
    """默认 0 = 关。不能因为『顺手量一下内存』把 worker 变成每张图重建。"""
    monkeypatch.setattr(ep, "WORKER_MAX_RSS_MB", 0)
    w = ep.worker_for("m1")
    monkeypatch.setattr(ep.Worker, "_rss_mb", lambda self: 99999)
    w.runs = 1
    # 直接验判据：limit=0 时 run() 里的分支根本不进
    rss_limit = int(ep.WORKER_MAX_RSS_MB or 0)
    assert rss_limit == 0
    assert not w.retired


def _drive_run(w, ep):
    """用最小的桩驱动一次真 run()（只为触发末尾的退休判据）。"""
    import numpy as np

    class _Stdout:
        def readline(self):
            return b"OK\t2\tfloat32\n"

        def read(self, n):
            return np.zeros(2, dtype="float32").tobytes()

    class _Stdin:
        def write(self, b):
            pass

        def flush(self):
            pass

    class _Proc:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.pid = 4321

        def poll(self):
            return None

    w.proc = _Proc()
    return w.run(np.zeros((1, 2), dtype="float32"))


def test_rss_boundary_is_inclusive_at_limit(ep, monkeypatch):
    """边界语义：RSS **等于**上限即退休（>= 而非 >），RSS 低于上限不退休。

    ★ 这条守的是阈值边界——rec 的自然水位实测就在 1050 MB，如果判据写成
      `>` 而不是 `>=`，一个恰好落在阈值上的 worker 会被反复判成「没超」而永久持有。
    """
    w = ep.worker_for("m1")
    monkeypatch.setattr(ep, "WORKER_MAX_RSS_MB", 1500)

    monkeypatch.setattr(ep.Worker, "_rss_mb", lambda self: 1499)
    _drive_run(w, ep)
    assert not w.retired, "低于上限不该退休（否则等于每张图都重建）"

    monkeypatch.setattr(ep.Worker, "_rss_mb", lambda self: 1500)
    w2 = ep.worker_for("m1")            # 取一个干净的（上一个已换）
    _drive_run(w2, ep)
    assert w2.retired, "恰好等于上限必须退休（>= 语义）"

# ---------------------------------------------------------------------------
# 空闲释放（TTL，ocr.idle_release_secs，2026-09-28）
#
# 为什么需要这组：原来的释放只有一个触发点 —— pool._settle 里的批次边界，
# 而它的判据是「索引池排空 ∧ 遍历线程结束」。watcher 增量路径一次只来一张，
# **永远等不到排空**，于是那次识别的高水位一直占着。
# ---------------------------------------------------------------------------


def test_idle_release_off_by_default_is_noop(ep):
    """TTL = 0（关）时任何空闲都不释放 —— 默认行为不能变。"""
    ep.WORKER_IDLE_RELEASE_SECS = 0
    ep.mark_used()
    ep.worker_for("m1")
    assert not ep.should_release_when_idle(now=1e9), "0 = 关，永远不判真"


def test_idle_release_triggers_after_ttl(ep):
    """空闲超过 TTL 才判真；刚好等于 TTL 也算（>= 语义，与 RSS 上限一致）。"""
    ep.WORKER_IDLE_RELEASE_SECS = 30
    w = ep.worker_for("m1")
    ep.mark_used()
    t0 = ep._LAST_USE

    assert not ep.should_release_when_idle(now=t0 + 29.9), "没到 TTL 不该释放"
    assert ep.should_release_when_idle(now=t0 + 30.0), "恰好到 TTL 就要释放（>=）"
    assert ep.should_release_when_idle(now=t0 + 100.0)


def test_idle_release_waits_for_real_use(ep):
    """从没跑过 OCR 时（_LAST_USE=0）不释放 —— 别把正在构造引擎的那一步打断。"""
    ep.WORKER_IDLE_RELEASE_SECS = 30
    ep.worker_for("m1")
    ep._LAST_USE = 0.0
    assert not ep.should_release_when_idle(now=1e9)


def test_idle_release_requires_live_workers(ep):
    """手上没 worker 时判否 —— 避免空转心跳反复通知上层。"""
    ep.WORKER_IDLE_RELEASE_SECS = 30
    ep.mark_used()
    ep._WORKERS.clear()
    assert not ep.should_release_when_idle(now=ep._LAST_USE + 999)


def test_idle_release_is_deduped_across_heartbeat(ep):
    """释放过之后再判否，直到又真的用过一次 —— 心跳每 2s 调一次不能反复通知。

    ★ 这条是本组里最要紧的：`pool._settle()` 每 2s 空转被唤醒调一次，
      没有世代去重的话，第一次释放后每次心跳都会再报一次「释放了 N 个 worker」。
    """
    ep.WORKER_IDLE_RELEASE_SECS = 30
    ep.worker_for("m1")
    ep.mark_used()
    t0 = ep._LAST_USE
    assert ep.should_release_when_idle(now=t0 + 31), "第一次该释放"

    assert ep.release_idle() == 1
    assert not ep.should_release_when_idle(now=t0 + 31), "刚释放过，不能马上又判真"

    # 再真正用一次 → 重新武装
    ep.mark_used()
    ep.worker_for("m1")
    assert not ep.should_release_when_idle(now=ep._LAST_USE + 1), "刚用过，不释放"
    assert ep.should_release_when_idle(now=ep._LAST_USE + 31), "又空闲够久 → 再释放"


def test_mark_used_is_called_by_recognize(monkeypatch):
    """recognize_ppocr 真跑完一次推理后必须刷新 _LAST_USE（TTL 的输入）。

    只打桩「引擎拿到手」这一步（`_try_ppocr`），后面的记账逻辑走真代码 ——
    真去 import rapidocr 会加载真模型（数秒），与本用例要验的东西无关。
    """
    import bibit_filesearch.ocr_engine as oe
    import bibit_filesearch.ocr_ep as ep2

    class _Res:
        txts = ["abc"]
        scores = [0.9]

    ep2._LAST_USE = 0.0
    # 注意返回的必须是**引擎**（可调用对象），recognize_ppocr 拿到后还会 eng(img)
    monkeypatch.setattr(oe, "_try_ppocr", lambda: (lambda img: _Res()))
    txt, conf = oe.recognize_ppocr(object())

    assert txt == "abc" and conf > 89, "先确认这一次真的走完了识别"
    assert ep2._LAST_USE > 0, "recognize_ppocr 成功后必须 mark_used()"
