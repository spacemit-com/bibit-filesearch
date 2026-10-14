"""候选池的零值语义（2026-10-08，审查 R07）。

`search.candidate_pool` 的约定（见 config.py 该键的注释与 retriever._pool_size）：

    正数 = 最多取相似度最高的这么多块参与文件级聚合
    <= 0 = 不截断

但 `main.create_app` 原先写的是

    candidate_pool=int(config.search.get("candidate_pool", 1000) or 1000)

`0 or 1000` = 1000 —— **显式 0 被默认值顶掉**，与文档写的「不截断」相反。
实测：配置 0 → Retriever 1000；配置 -1 → -1（负数反而正常）。

本文件守住：显式 0 保留、缺省用 1000、正数照传。
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from bibit_filesearch.settings import CONFIG_VERSION


def _app_with_pool(tmp_path, pool_value, monkeypatch, *, key_present=True):
    """用隔离配置起一个 app，返回 app.state.retriever.candidate_pool。"""
    from bibit_filesearch.main import create_app

    root = tmp_path / "data"
    root.mkdir()
    kb = root / "kb"
    kb.mkdir()

    cfg: dict = {
        "config_version": CONFIG_VERSION,
        "index_dirs": [str(kb)],
        "auto_index": False,
    }
    if key_present:
        cfg["search"] = {"candidate_pool": pool_value}
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    monkeypatch.setenv("BIBIT_CONFIG", str(cfg_path))
    monkeypatch.setenv("BIBIT_FILESEARCH_DATA_ROOT", str(root))

    with patch("bibit_filesearch.scanner.ScanService.start"), \
         patch("bibit_filesearch.recovery.EmbedRecovery.start"), \
         patch("bibit_filesearch.retriever.Retriever.warmup"), \
         patch("bibit_filesearch.model_client.EmbedClient.probe_dim", return_value=1024):
        app = create_app(str(cfg_path))
    try:
        return app.state.retriever.candidate_pool, app.state.retriever._pool_size()
    finally:
        app.state.watcher.stop()
        if hasattr(app.state, "scanner"):
            app.state.scanner._status = "idle"
        app.state.embed.close()


def test_explicit_zero_is_kept_and_means_no_truncation(tmp_path, monkeypatch):
    """显式 0 必须原样传到 Retriever —— 改前被 `or 1000` 顶成 1000。"""
    pool, effective = _app_with_pool(tmp_path, 0, monkeypatch)
    assert pool == 0, (
        f"candidate_pool 配 0，Retriever 实为 {pool}——0 是「不截断」的明确写法，"
        "不能被默认值 1000 覆盖"
    )
    assert effective >= 10 ** 8, "0 应解成不截断（_pool_size 归一大整数）"


def test_negative_also_means_no_truncation(tmp_path, monkeypatch):
    """负数同样是不截断（对照组：改前就能过，防回归）。"""
    pool, effective = _app_with_pool(tmp_path, -1, monkeypatch)
    assert pool == -1
    assert effective >= 10 ** 8


def test_positive_value_passes_through(tmp_path, monkeypatch):
    pool, effective = _app_with_pool(tmp_path, 250, monkeypatch)
    assert pool == 250
    assert effective == 250


def test_missing_key_uses_default_1000(tmp_path, monkeypatch):
    """没配这个键时用 1000（默认值仍然生效）。"""
    pool, effective = _app_with_pool(tmp_path, None, monkeypatch, key_present=False)
    assert pool == 1000
    assert effective == 1000


def test_missing_whole_search_section_uses_default(tmp_path, monkeypatch):
    """连 search 段都没有时也要落到 1000，而不是崩或传 None。"""
    pool, _ = _app_with_pool(tmp_path, None, monkeypatch, key_present=False)
    assert pool == 1000
