"""config.load_config 路径解析回归测试（2026-09-07 方案2）。

守护语义：默认路径优先级 = BIBIT_CONFIG env > ~/.config/bibit-filesearch/config.yaml
> 包内默认。此前 load_config 不查用户层，systemd 自启（无 env）落到包内默认
（旧 512 维 config.yaml），与存量 Qwen3 1024 索引不符启动即崩。
"""
import os

import pytest

from bibit_filesearch.config import load_config

# 顶层绝对 import（tests/ 直接跑时仓库根在 sys.path；此处显式补一手防 flake）
try:
    from bibit_filesearch.config import load_config as _lc  # noqa
except Exception:  # pragma: no cover
    pass


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_load_config_prefers_bibit_config_env(monkeypatch, tmp_path):
    """① BIBIT_CONFIG env 最高优先。"""
    env_cfg = _write(tmp_path, "env.yaml", "top_k: 7\n")
    home = tmp_path / "home"
    _write(home, ".config/bibit-filesearch/config.yaml", "top_k: 9\n")
    monkeypatch.setenv("BIBIT_CONFIG", str(env_cfg))
    monkeypatch.setenv("HOME", str(home))
    cfg = load_config()
    assert cfg.get("top_k") == 7


def test_load_config_falls_back_to_user_config(monkeypatch, tmp_path):
    """② 无 env → 读 ~/.config/bibit-filesearch/config.yaml（用户层单一来源）。"""
    monkeypatch.delenv("BIBIT_CONFIG", raising=False)
    home = tmp_path / "home"
    _write(home, ".config/bibit-filesearch/config.yaml", "top_k: 5\n")
    monkeypatch.setenv("HOME", str(home))
    cfg = load_config()
    assert cfg.get("top_k") == 5, "无 env 时应读 ~/.config 用户配置"


def test_load_config_ignores_missing_user_config(monkeypatch, tmp_path):
    """③ 用户层不存在 → 落到包内默认（源码旁 config.yaml），不崩。"""
    monkeypatch.delenv("BIBIT_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "nonexistent-home"))
    cfg = load_config()
    # DEFAULTS 兜底能加载（顶层 index_dirs 默认空列表），不崩即可
    assert cfg.get("index_dirs") == []


def test_load_config_explicit_path_wins(monkeypatch, tmp_path):
    """④ 显式传参 > env（create_app(cfg) 等调用不受影响）。"""
    explicit = _write(tmp_path, "explicit.yaml", "top_k: 3\n")
    env_cfg = _write(tmp_path, "env.yaml", "top_k: 7\n")
    monkeypatch.setenv("BIBIT_CONFIG", str(env_cfg))
    cfg = load_config(str(explicit))
    assert cfg.get("top_k") == 3
