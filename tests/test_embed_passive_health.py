"""/health/deep 的**被动**健康判据（2026-09-21 新增，回归「扫描中抖动」）。

背景（用户实测）：知识库指向 118 文件 / 40MB 目录时，界面状态在「扫描中」与
「索引服务不可用 / 向量化服务未启动」之间来回跳。

根因：探活 ping() 发的是一个**真实 embeddings 请求**，而该服务串行处理——
扫描期间批次一个接一个（单批 32 块实测 25~46s），探针只能排在队尾，等待时间 =
当前批次剩余时间 ≫ probe_timeout(5s) → 必然超时 → 判 unreachable → 前端把
embedUp=false 的分支排在 scanning 之前 → 顶掉「扫描中」。批次间隙又跳回去，
5s 轮询撞 25~46s 批次，抖动是稳定复现的。

修法：加**被动**判据 is_healthy()——最近 grace 秒内真的成功产出过向量即为健康，
不发任何请求。本文件锁定该语义与两个调用点（/health/deep、看门狗 _probe）。
"""
import os
import time

import pytest
import yaml


# ---------------------------------------------------------------- 替身
class _BusyEmbed:
    """「忙但健康」的 embed：ping 假超时（返回 False），但被动信号说健康。

    这正是生产故障的形态：ping 失败**不代表**后端坏了，只代表现在没空。
    """

    def __init__(self, healthy: bool = True, ping_ok: bool = False, url: str = "http://127.0.0.1:18790"):
        self._healthy = healthy
        self._ping_ok = ping_ok
        self.url = url
        self.ping_calls = 0

    def is_healthy(self) -> bool:
        return self._healthy

    def ping(self) -> bool:
        self.ping_calls += 1
        return self._ping_ok

    def close(self) -> None:
        pass


class _Resp:
    """httpx 响应的最小替身（EmbedClient._post_embeddings 的返回）。"""

    def __init__(self, payload, status: int = 200, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _client(**kw):
    import httpx
    from bibit_filesearch.model_client import EmbedClient

    c = EmbedClient("http://127.0.0.1:18790", dim=4, model="qwen3-embedding-0.6b-q4_0",
                    batch_size=32, probe_timeout=0.05, **kw)
    c._client = httpx.Client(timeout=httpx.Timeout(1.0, connect=1.0))
    return c


def _ok_payload(n: int, dim: int = 4) -> dict:
    return {"data": [{"index": i, "embedding": [0.1] * dim} for i in range(n)]}


# ---------------------------------------------------------------- 1. 被动信号语义
def test_not_healthy_before_any_success():
    """从未成功过 → 不健康（必须回落真探针，否则「从没起来过」会被判健康）。"""
    c = _client()
    try:
        assert c.is_healthy() is False
    finally:
        c.close()


def test_healthy_after_real_success(monkeypatch):
    """真实推理成功一次 → 被动判据即为健康，且不额外发请求。"""
    c = _client()
    try:
        monkeypatch.setattr(c, "_post_embeddings", lambda texts, timeout=None: _Resp(_ok_payload(len(texts))))
        assert c.is_healthy() is False
        vecs = c.embed_batch(["a", "b"])
        assert len(vecs) == 2
        assert c.is_healthy() is True
    finally:
        c.close()


def test_not_healthy_after_grace_expires(monkeypatch):
    """成功之后久未再成功（超过 grace）→ 回落不健康，交给真探针。"""
    c = _client(health_grace=0.2)
    try:
        monkeypatch.setattr(c, "_post_embeddings", lambda texts, timeout=None: _Resp(_ok_payload(len(texts))))
        c.embed_batch(["a"])
        assert c.is_healthy() is True
        time.sleep(0.3)
        assert c.is_healthy() is False
    finally:
        c.close()


def test_grace_defaults_from_timeout():
    """grace 未显式给定时从 timeout 推导：max(2×timeout, 120)，保证覆盖一个批次。"""
    c60 = _client(timeout=60.0, health_grace=0.0)
    c300 = _client(timeout=300.0, health_grace=0.0)
    try:
        assert c60.health_grace == 120.0      # 2×60=120 与下限 120 相等
        assert c300.health_grace == 600.0     # 长超时场景按 2× 放大
    finally:
        c60.close()
        c300.close()


def test_explicit_grace_wins():
    c = _client(timeout=60.0, health_grace=7.5)
    try:
        assert c.health_grace == 7.5
    finally:
        c.close()


def test_failed_request_does_not_mark_healthy(monkeypatch):
    """请求失败不得刷新被动信号（否则失败会自证健康）。"""
    c = _client(retries=0, health_grace=0.2)
    try:
        monkeypatch.setattr(c, "_post_embeddings", lambda texts, timeout=None: _Resp(_ok_payload(len(texts))))
        c.embed_batch(["a"])
        assert c.is_healthy() is True
        time.sleep(0.3)
        # 之后连续失败
        monkeypatch.setattr(c, "_post_embeddings",
                            lambda texts, timeout=None: _Resp({}, status=500, text="boom"))
        with pytest.raises(Exception):
            c.embed_batch(["b"])
        assert c.is_healthy() is False
    finally:
        c.close()


def test_probe_does_not_refresh_passive_signal(monkeypatch):
    """探针（ping）成功**不**刷新被动信号。

    否则「每 grace 秒探一次」会自我延续成「永远不用再探」，空闲期真掉线被掩盖。
    被动信号只反映业务负载的健康。
    """
    c = _client(health_grace=0.2)
    try:
        monkeypatch.setattr(c, "_post_embeddings", lambda texts, timeout=None: _Resp(_ok_payload(len(texts))))
        assert c.ping() is True
        # ping 成功但从未有业务成功 → 仍然不健康
        assert c.is_healthy() is False
    finally:
        c.close()


# ---------------------------------------------------------------- 2. /health/deep
@pytest.fixture()
def app_ctx(tmp_path):
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    cfg = {
        "storage": {"db_path": str(tmp / "ph.db")},
        "server": {"port": 18097, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "top_k": 50,
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    from fastapi.testclient import TestClient
    from bibit_filesearch.main import create_app

    app = create_app(str(cfg_path))
    yield app, TestClient(app)
    if hasattr(app.state, "embed"):
        app.state.embed.close()


def test_health_deep_ok_when_busy_ping_times_out(app_ctx):
    """★ 回归：扫描中 ping 假超时，但被动信号健康 → 必须报 ok，不得 degraded。

    修复前这里是 embed=unreachable（这就是界面抖动的后端原因）。
    """
    app, client = app_ctx
    fake = _BusyEmbed(healthy=True, ping_ok=False)
    app.state.embed = fake

    d = client.get("/health/deep").json()["data"]
    assert d["embed"] == "ok"
    assert d["status"] == "ok"
    # 被动信号已足够，**不应**再去发那个注定超时的探针
    assert fake.ping_calls == 0


def test_health_deep_probes_when_idle(app_ctx):
    """空闲（被动信号过期）→ 仍然真探；探针失败照旧报 unreachable（不回归旧 bug）。"""
    app, client = app_ctx
    fake = _BusyEmbed(healthy=False, ping_ok=False)
    app.state.embed = fake

    d = client.get("/health/deep").json()["data"]
    assert d["embed"] == "unreachable"
    assert d["status"] == "degraded"
    assert fake.ping_calls == 1


def test_health_deep_ok_when_idle_and_ping_ok(app_ctx):
    """空闲 + 探针成功 → ok（对照组）。"""
    app, client = app_ctx
    fake = _BusyEmbed(healthy=False, ping_ok=True)
    app.state.embed = fake

    d = client.get("/health/deep").json()["data"]
    assert d["embed"] == "ok"
    assert d["status"] == "ok"
    assert fake.ping_calls == 1


# ---------------------------------------------------------------- 3. 看门狗
def test_watchdog_probe_uses_passive_signal(tmp_path):
    """★ 回归：看门狗在忙时不得把好后端判 down（假跳变会凭空触发自动重建）。"""
    from bibit_filesearch.recovery import EmbedRecovery

    class _DB:
        conn = None

    rec = EmbedRecovery(_BusyEmbed(healthy=True, ping_ok=False), _DB(), None, interval=0.05)
    assert rec._probe() is True
    assert rec.embed.ping_calls == 0


def test_watchdog_probe_falls_back_to_ping_when_idle(tmp_path):
    """空闲时仍真探（探针失败 → down，保持原有语义）。"""
    from bibit_filesearch.recovery import EmbedRecovery

    class _DB:
        conn = None

    rec = EmbedRecovery(_BusyEmbed(healthy=False, ping_ok=False), _DB(), None, interval=0.05)
    assert rec._probe() is False
    assert rec.embed.ping_calls == 1


def test_watchdog_tolerates_stub_without_is_healthy():
    """测试桩/旧实现只实现 ping() 时不得因缺方法而误判 down。"""
    from bibit_filesearch.recovery import EmbedRecovery

    class _DB:
        conn = None

    class _Stub:
        url = ""

        def ping(self):
            return True

        def close(self):
            pass

    rec = EmbedRecovery(_Stub(), _DB(), None, interval=0.05)
    assert rec._probe() is True
