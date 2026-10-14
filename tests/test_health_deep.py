"""/health/deep 契约测试（2026-09-09 新增，回归 bug）。

背景：health_deep 原实现 `embed.ping()` 丢弃返回值——ping() 自身吞掉所有异常
并返回 bool，故 embed 掉线时仍报 embed=ok / status=ok（2026-09-09 实测 :8082
无监听却 ok）。回归断言：ping() 返回 False 时必须 embed=unreachable + degraded。

用 TestClient（不启真实端口），embed 用假对象直接替换 app.state.embed，
避免依赖真实 embed 服务。
"""
import os

import pytest
import yaml


class _PingEmbed:
    """最小 embed 替身：只实现 health_deep 用到的 ping() / url。"""

    def __init__(self, ok: bool, url: str = ""):
        self._ok = ok
        self.url = url

    def ping(self) -> bool:
        return self._ok

    def close(self) -> None:
        pass


@pytest.fixture()
def app_ctx(tmp_path):
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    cfg = {
        "storage": {"db_path": str(tmp / "health.db")},
        "server": {"port": 18098, "cors_origins": []},
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


def test_health_deep_degraded_when_embed_down(app_ctx):
    """embed.ping() 返回 False → embed=unreachable、status=degraded。"""
    app, client = app_ctx
    app.state.embed = _PingEmbed(ok=False, url="http://127.0.0.1:8082")

    r = client.get("/health/deep")
    assert r.status_code == 200
    j = r.json()
    assert j["code"] == 0
    d = j["data"]
    assert d["embed"] == "unreachable"
    assert d["status"] == "degraded"
    # 前端降级横幅据此显示 embed 服务地址
    assert d["embedUrl"] == "http://127.0.0.1:8082"


def test_health_deep_ok_when_embed_up(app_ctx):
    """embed.ping() 返回 True → embed=ok、status=ok（对照组）。"""
    app, client = app_ctx
    app.state.embed = _PingEmbed(ok=True, url="http://127.0.0.1:8082")

    r = client.get("/health/deep")
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["embed"] == "ok"
    assert d["status"] == "ok"
