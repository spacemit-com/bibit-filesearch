"""配置路径只解析一处（2026-10-08，审查 R06）。

## R06：两套环境变量各管一半

    load_config()     读 BIBIT_CONFIG
    SettingsService   读 BIBIT_FILESEARCH_CONFIG

只设 `BIBIT_CONFIG` 时，组件配置来自那个文件、`/api/settings` 读写的却是
`~/.config/...`。实测：组件层 top_k=17、设置层 top_k=5 —— 用户「保存成功」，
运行行为不跟。

修法：`main.resolve_config_path` 把「配置从哪读」解析成**一个**路径，create_app
把它同时交给 SettingsService 与 load_config。（R05 的 CLI `--config` 接线见
`test_cli_config_flag.py`。）
优先级：显式参数 > BIBIT_CONFIG > BIBIT_FILESEARCH_CONFIG > None（各自默认）。
"""
from __future__ import annotations

import pathlib

import yaml

from bibit_filesearch.main import resolve_config_path


# ── resolve_config_path 的优先级 ──────────────────────────────────────────────

def test_explicit_path_wins(monkeypatch):
    monkeypatch.setenv("BIBIT_CONFIG", "/from/env.yaml")
    monkeypatch.setenv("BIBIT_FILESEARCH_CONFIG", "/from/other.yaml")
    assert resolve_config_path("/explicit.yaml") == "/explicit.yaml"


def test_bibit_config_env_beats_the_other_env(monkeypatch):
    """两个 env 同时存在时以 BIBIT_CONFIG 为准（与原先 load_config 的取舍一致）。"""
    monkeypatch.setenv("BIBIT_CONFIG", "/a.yaml")
    monkeypatch.setenv("BIBIT_FILESEARCH_CONFIG", "/b.yaml")
    assert resolve_config_path(None) == "/a.yaml"


def test_falls_back_to_settings_env_then_none(monkeypatch):
    """只设 BIBIT_FILESEARCH_CONFIG 也要能被识别（它原先只对设置层生效）。"""
    monkeypatch.delenv("BIBIT_CONFIG", raising=False)
    monkeypatch.setenv("BIBIT_FILESEARCH_CONFIG", "/only/settings.yaml")
    assert resolve_config_path(None) == "/only/settings.yaml"

    monkeypatch.delenv("BIBIT_FILESEARCH_CONFIG", raising=False)
    assert resolve_config_path(None) is None


# ── R06：两个组件必须读同一个文件 ──────────────────────────────────────────────

def _write_user_config(home: pathlib.Path, top_k: int) -> pathlib.Path:
    p = home / ".config/bibit-filesearch/config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"top_k": top_k}), encoding="utf-8")
    return p


def test_single_env_keeps_settings_and_config_in_sync(tmp_path, monkeypatch):
    """只设 BIBIT_CONFIG：设置层与组件层读**同一个**文件（修复前两者分属两处）。"""
    from bibit_filesearch.config import load_config
    from bibit_filesearch.settings import SettingsService

    home = tmp_path / "home"
    _write_user_config(home, top_k=5)                 # 干扰项：~/.config 里的旧值
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({"top_k": 17}), encoding="utf-8")

    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.delenv("BIBIT_FILESEARCH_CONFIG", raising=False)
    monkeypatch.setenv("BIBIT_CONFIG", str(custom))

    resolved = resolve_config_path(None)
    svc = SettingsService(config_path=resolved)
    cfg = load_config(resolved)

    assert svc.get_all()["top_k"] == 17, "设置层应读 BIBIT_CONFIG 指定的文件"
    assert cfg.top_k == 17, "组件层应读同一个文件"
    assert str(svc.config_path()) == str(custom)


def test_user_config_still_used_when_no_env(tmp_path, monkeypatch):
    """没设任何 env：两边都落到 ~/.config 用户配置（行为与原先一致）。"""
    from bibit_filesearch.config import load_config
    from bibit_filesearch.settings import SettingsService

    home = tmp_path / "home"
    _write_user_config(home, top_k=9)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.delenv("BIBIT_CONFIG", raising=False)
    monkeypatch.delenv("BIBIT_FILESEARCH_CONFIG", raising=False)

    resolved = resolve_config_path(None)
    assert resolved is None, "没有指定时不该编造路径，各组件按默认走"
    assert SettingsService().get_all()["top_k"] == 9
    assert load_config().top_k == 9
