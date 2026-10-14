"""恢复出厂默认（POST /api/settings/reset）语义回归测试（2026-09-09）。

背景：稀疏用户文件语义下（用户文件只存「与默认不同的键」），「恢复默认」=
清空用户文件里的自定义键 → 生效值回落代码默认。此前只有手改文件一条路，
GUI 无入口（用户原话「确实需要现在恢复默认只能手改文件」）。

覆盖：
- SettingsService.reset()：清空用户键、保留 config_version、生效值回落默认
- reset 幂等：连续调用两次结果一致，不报错
- 无用户文件时 reset 不炸（文件本就不存在）
- POST /api/settings/reset：契约、data 为完整生效设置、写盘内容只剩版本号
- reset 无条件重应用（不因「看起来没变化」早退）——scanner 被 restart
- reset 与 PUT 的语义差异：PUT 只删「等于默认」的键，reset 清空全部
- index_dirs 回落动态默认（~/Documents，存在才有）
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from bibit_filesearch.settings import CONFIG_VERSION, SettingsService


def _svc(tmp_path, cfg: dict | None = None) -> tuple[SettingsService, pathlib.Path]:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg or {}, allow_unicode=True), encoding="utf-8")
    return SettingsService(config_path=p), p


def test_reset_clears_user_keys_keeps_version(tmp_path):
    """reset 清空所有用户键，文件只剩 config_version。"""
    svc, p = _svc(tmp_path, {
        "index_dirs": ["/home/u/Docs"],
        "top_k": 50,
        "max_depth": 5,
        "indexable_types": ["md"],
        "config_version": CONFIG_VERSION,
    })
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["top_k"] == 50

    svc.reset()
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after == {"config_version": CONFIG_VERSION}, \
        f"reset 后文件应只剩版本号，实际: {after}"


def test_reset_effective_values_fall_back_to_defaults(tmp_path, monkeypatch):
    """reset 后生效值全部回落代码默认。"""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home" / "Documents").mkdir(parents=True)
    svc, _ = _svc(tmp_path, {
        "index_dirs": ["/home/u/Docs"], "top_k": 50, "max_depth": 5,
        "auto_index": False, "indexable_types": ["md"],
        "config_version": CONFIG_VERSION,
    })
    d = svc.reset()
    assert d["top_k"] == 5, "top_k 应回落默认 5"
    assert d["max_depth"] == 3, "max_depth 应回落默认 3"
    assert d["auto_index"] is True, "auto_index 应回落默认 True"
    assert len(d["indexable_types"]) == 20, "indexable_types 应回落全 20 类（含 png/jpg）"
    # index_dirs 走动态默认（仅存在者）
    assert d["index_dirs"] == [str(tmp_path / "home" / "Documents")], \
        "index_dirs 应回落动态默认 ~/Documents"


def test_reset_is_idempotent(tmp_path):
    """连续 reset 两次结果一致，且不抛异常。"""
    svc, p = _svc(tmp_path, {"top_k": 50, "config_version": CONFIG_VERSION})
    first = svc.reset()
    body1 = p.read_text(encoding="utf-8")
    second = svc.reset()
    assert first == second, "两次 reset 的生效值应一致"
    assert p.read_text(encoding="utf-8") == body1, "第二次 reset 不应再改文件内容"


def test_reset_without_user_file_does_not_crash(tmp_path):
    """用户文件不存在时 reset 正常工作（不创建文件、不抛异常）。"""
    p = tmp_path / "config.yaml"
    svc = SettingsService(config_path=p)  # 文件不存在
    assert not p.exists()
    d = svc.reset()
    assert d["top_k"] == 5
    # 语义选择：文件本不存在 → reset 不无中生有地创建它（保持「首启不写文件」）
    assert not p.exists(), "reset 不应在文件不存在时创建文件"


def test_reset_differs_from_put(tmp_path):
    """reset 清空全部用户键；PUT 把值改回默认只是删那一个键。"""
    svc, p = _svc(tmp_path, {
        "top_k": 50, "max_depth": 5, "config_version": CONFIG_VERSION,
    })
    svc.update({"top_k": 5})  # PUT 改回默认 → 只删 top_k
    after_put = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "top_k" not in after_put
    assert after_put["max_depth"] == 5, "PUT 不应动其他键"

    svc.reset()
    after_reset = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after_reset == {"config_version": CONFIG_VERSION}, "reset 应清空全部"
    assert svc.get("max_depth") == 3, "max_depth 回落默认"


# ---------- HTTP 端点 ----------

@pytest.fixture()
def app_ctx(tmp_path, monkeypatch):
    """临时配置 + create_app → (client, cfg_path)。"""
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "storage": {"db_path": str(tmp / "reset.db")},
        "server": {"port": 18098, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "top_k": 50,
    }, allow_unicode=True), encoding="utf-8")

    monkeypatch.setenv("BIBIT_CONFIG", str(cfg_path))
    from fastapi.testclient import TestClient
    from bibit_filesearch.main import create_app

    app = create_app(str(cfg_path))
    return TestClient(app), cfg_path, app


def test_reset_endpoint_contract(app_ctx):
    """POST /api/settings/reset：契约 + data 为完整生效设置。"""
    client, cfg_path, _ = app_ctx
    r = client.post("/api/settings/reset")
    j = r.json()
    assert r.status_code == 200
    assert j["code"] == 0 and j["message"] == "success"
    d = j["data"]
    for k in ("index_dirs", "auto_index", "indexable_types",
              "exclude_patterns", "max_depth", "top_k"):
        assert k in d, f"data 缺 key: {k}"
    assert d["top_k"] == 5, "应回落代码默认 5（非 yaml 里的 50）"
    assert d["max_depth"] == 3
    assert "changed" not in d, "reset 语义不涉及 changed 字段"


def test_reset_endpoint_clears_file_and_get_agrees(app_ctx):
    """reset 后落盘只剩版本号，GET 返回同一份生效值。"""
    client, cfg_path, _ = app_ctx
    client.post("/api/settings/reset")
    after = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert after == {"config_version": CONFIG_VERSION}, f"落盘应为稀疏空壳，实际 {after}"
    d = client.get("/api/settings").json()["data"]
    assert d["top_k"] == 5
    assert d["max_depth"] == 3


def test_reset_endpoint_reapplies_unconditionally(app_ctx):
    """reset 必须无条件重应用（scanner.restart 被调用），不能因「没变化」早退。

    这是与 PUT 的关键差异：PUT 幂等早退（changed=false）时不再重扫；
    用户点「恢复默认」的意图是让默认值重新生效并重扫，必须每次都触发。
    """
    client, cfg_path, app = app_ctx
    scanner = app.state.scanner
    calls = {"n": 0}
    orig = scanner.restart

    def _spy():
        calls["n"] += 1
        return orig()

    scanner.restart = _spy
    client.post("/api/settings/reset")
    assert calls["n"] == 1, "reset 应调用 scanner.restart 一次"
    # 再点一次（此时配置已经是默认，没有任何变化）——仍应重扫
    client.post("/api/settings/reset")
    assert calls["n"] == 2, "reset 不得因「配置未变化」而跳过重扫"


def test_reset_endpoint_then_put_still_works(app_ctx):
    """reset 之后 PUT 仍能正常写入（版本号未丢、迁移不会重跑）。"""
    client, cfg_path, _ = app_ctx
    client.post("/api/settings/reset")
    r = client.put("/api/settings", json={"top_k": 33})
    assert r.json()["code"] == 0
    after = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert after["top_k"] == 33
    assert after["config_version"] == CONFIG_VERSION
