"""服务停止时是否真的显式释放了 ep_worker（2026-09-28）。

守两件事：
  ① lifespan 的 shutdown 会调 release_idle()（uvicorn 优雅关闭的路径）；
  ② 幂等 —— 信号兜底与 lifespan 可能都触发，第二次不能重复关/重复报。

不跑真 ep_worker：只验「有没有把释放这件事做出去」，真进程的行为由
ocr_ep / ocr_engine 自己的用例守。用 `_release_on_shutdown` 而非整个 app，
是为了不把 FastAPI 起停拖进这个用例。
"""
import sys

sys.path.insert(0, "src")


def test_shutdown_releases_ocr_workers(monkeypatch):
    import bibit_filesearch.main as m
    import bibit_filesearch.ocr_engine as oe

    calls = []
    monkeypatch.setattr(m, "_SHUTDOWN_RELEASED", False)
    monkeypatch.setattr(oe, "release_idle", lambda: calls.append(1) or 3)

    m._release_on_shutdown()

    assert calls == [1], "服务停止必须调一次 release_idle()"
    assert m._SHUTDOWN_RELEASED is True, "要留标记，供幂等判断"


def test_shutdown_release_is_idempotent(monkeypatch):
    """第二次（信号兜底 + lifespan 都触发时）不能再调一次。"""
    import bibit_filesearch.main as m
    import bibit_filesearch.ocr_engine as oe

    calls = []
    monkeypatch.setattr(m, "_SHUTDOWN_RELEASED", False)
    monkeypatch.setattr(oe, "release_idle", lambda: calls.append(1) or 0)

    m._release_on_shutdown()
    m._release_on_shutdown()
    m._release_on_shutdown()

    assert calls == [1], "幂等：多次触发只释放一次"


def test_shutdown_release_survives_engine_error(monkeypatch):
    """释放抛异常不能把关闭流程带崩 —— 记一条 warning 就够。"""
    import bibit_filesearch.main as m
    import bibit_filesearch.ocr_engine as oe

    def boom():
        raise RuntimeError("模拟引擎异常")

    monkeypatch.setattr(m, "_SHUTDOWN_RELEASED", False)
    monkeypatch.setattr(oe, "release_idle", boom)

    m._release_on_shutdown()          # 不抛即通过
    assert m._SHUTDOWN_RELEASED is True, "失败也要置标记，避免反复重试拖慢关闭"


def test_lifespan_calls_release_on_shutdown(monkeypatch):
    """lifespan 的形状：yield 之后必须调 _release_on_shutdown()（uvicorn 靠它）。"""
    import bibit_filesearch.main as m

    calls = []
    monkeypatch.setattr(m, "_release_on_shutdown", lambda: calls.append(1))

    gen = m._lifespan(object())
    next(gen)                          # 启动阶段：这里不该释放
    assert calls == [], "启动阶段不能释放 worker"

    try:
        next(gen)
    except StopIteration:
        pass
    assert calls == [1], "停止阶段必须释放"


def test_app_uses_lifespan():
    """FastAPI 必须真的挂上我们的 lifespan —— 挂了但没接上就等于没做。"""
    import inspect

    import bibit_filesearch.main as m

    src = inspect.getsource(m.create_app)
    assert "lifespan=_lifespan" in src, "create_app 里必须 FastAPI(..., lifespan=_lifespan)"
