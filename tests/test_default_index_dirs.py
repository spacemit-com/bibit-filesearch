"""守护 index_dirs 的「动态默认」（2026-10-14 收敛）。

守护语义：index_dirs 的「动态默认」= 当前用户 ~/Documents，仅当目录真实
存在才出现，不硬编码用户名。

沿革：2026-09-09 曾把 ~/Desktop 也纳入默认（顺序 Desktop 在前）；
2026-10-14 按要求收敛回只索引 ~/Documents。该默认只在 yaml 里 index_dirs
为空/缺失时生效，已写过配置的用户不受影响（get/update 均走 yaml 优先）。
"""
from __future__ import annotations

import pathlib

from bibit_filesearch.settings import SettingsService


def _svc(tmp_path, monkeypatch, home):
    """构造 SettingsService，config 指向空 index_dirs 的 yaml。"""
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("index_dirs: []\n", encoding="utf-8")
    return SettingsService(config_path=cfg)


def test_default_index_dirs_is_documents(tmp_path, monkeypatch):
    """~/Documents 存在 → [~/Documents]（不再包含 ~/Desktop）。"""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    (home / "Desktop").mkdir(parents=True)          # 存在也不纳入
    svc = _svc(tmp_path, monkeypatch, home)
    assert svc.get("index_dirs") == [str(home / "Documents")]


def test_default_index_dirs_empty_when_documents_missing(tmp_path, monkeypatch):
    """~/Documents 不存在 → 空列表（不猜路径；~/Desktop 在也不算）。"""
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    svc = _svc(tmp_path, monkeypatch, home)
    assert svc.get("index_dirs") == []


def test_default_does_not_override_user_config(tmp_path, monkeypatch):
    """用户已写 yaml → 动态默认不得覆盖（配置单一来源，升级不劫持老用户）。"""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    custom = tmp_path / "custom_kb"
    custom.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"index_dirs:\n- {custom}\n", encoding="utf-8")
    svc = SettingsService(config_path=cfg)
    assert svc.get("index_dirs") == [str(custom)]
    assert svc.get_all()["index_dirs"] == [str(custom)]


def test_default_respects_existing_desktop_only_config(tmp_path, monkeypatch):
    """用户配置里仍写着 ~/Desktop → 原样保留（收敛默认不劫持既有配置）。"""
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"index_dirs:\n- {home / 'Desktop'}\n", encoding="utf-8")
    svc = SettingsService(config_path=cfg)
    assert svc.get("index_dirs") == [str(home / "Desktop")]
