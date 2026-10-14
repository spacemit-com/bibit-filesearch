"""/api/meta 契约测试（实例归属元信息，2026-09-08 新增）。

覆盖：
- GET /api/meta → code 0，data 含 uid / userName / homeDir / port 四键
- uid == os.getuid()；userName == pwd.getpwuid(uid).pw_name
- homeDir == str(Path.home())（测试以 HOME=/root 运行即 /root）
- port == main._port_for_uid(uid)（serve 默认按 uid 派生，非 config server.port）

用 TestClient（不启真实端口）；配置指向临时目录，避免真实扫描干扰。
"""
import os
import pwd
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def app_ctx(tmp_path):
    """临时配置 + create_app → (client, cfg_path, tmp)。"""
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    cfg = {
        "storage": {"db_path": str(tmp / "meta.db")},
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
    yield TestClient(app), cfg_path, tmp
    if hasattr(app.state, "embed"):
        app.state.embed.close()


def test_meta_contract(app_ctx):
    """GET /api/meta：code 0 + 键齐全（rev 5 起多了 ocr 段）。"""
    client, cfg_path, tmp = app_ctx
    r = client.get("/api/meta")
    assert r.status_code == 200
    j = r.json()
    assert j["code"] == 0
    assert j["message"] == "success"
    d = j["data"]
    assert set(d) == {"uid", "userName", "homeDir", "port", "ocr"}


def test_meta_exposes_ocr_status(app_ctx):
    """rev 5：/api/meta 必须报出「OCR 到底在跑什么」——引擎唯一化之后，
    这是判断「引擎没起来」与「参数不对」的唯一在线入口（旧版只能翻日志，
    于是出现过 postinst 报就绪、实际跑不动的谎报）。"""
    client, cfg_path, tmp = app_ctx
    ocr = client.get("/api/meta").json()["data"]["ocr"]
    assert ocr["engine"] == "ppocr", "引擎只剩 PP-OCR"
    assert "device" in ocr and "ready" in ocr
    assert ocr["device"] in ("npu", "cpu", "?")
    # 2026-09-28：TTL 空闲释放的三格 —— 「worker 到底放没放掉」的在线观测点。
    # 少了它们，troubleshooting 里那句「查 worker_count」就是空头承诺。
    assert "idle_release_secs" in ocr, "必须报出 TTL 阈值"
    assert "worker_count" in ocr, "必须报出当前活着的 worker 数"
    assert "idle_age_secs" in ocr, "必须报出距上次使用多久"


def test_meta_uid_matches_running_process(app_ctx):
    """实例归属 = 运行身份：uid/userName/homeDir/port 与本进程一致。

    核心断言（多用户支持）：插件据此确认 :18700 段实例属于哪个用户。
    """
    client, cfg_path, tmp = app_ctx
    d = client.get("/api/meta").json()["data"]

    uid = os.getuid()
    from bibit_filesearch.main import _port_for_uid

    assert d["uid"] == uid, "uid 必须等于服务进程实际 uid"
    assert d["userName"] == pwd.getpwuid(uid).pw_name
    assert d["homeDir"] == str(Path.home())
    assert d["port"] == _port_for_uid(uid), "port 必须与 serve 默认的 uid 派生端口一致"
