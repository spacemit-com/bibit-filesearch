"""CLI `serve --config` 必须真的生效（2026-10-08，审查 R05）。

CLI 解析了 `args.config`，但 `uvicorn.run()` 只收到字符串工厂名
`bibit_filesearch.main:create_app`，而 factory 调用 `create_app()` 不传参 ——
`args.config` 从未被使用（`grep args.config` 零命中）。

后果：`serve --config /tmp/x.yaml` 静默落到真实用户配置与真实数据目录。开发时
做隔离实验踩过这个坑（以为在用临时配置，其实写脏了本机库）。

修法：改用闭包工厂，把解析出的路径带进 `create_app`。

★ 夹具注意：应用工厂必须在 **patch 仍生效时**调用（uvicorn 启动时就是这么做的）；
出了 `with` 再调会打到真的 `create_app`（本文件第一版就这么错过一次）。
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import yaml

from bibit_filesearch import main as main_mod


def _run_main(argv, tmp_path, monkeypatch):
    """跑一遍 main()，并在 patch 生效期间触发一次应用工厂。

    返回 captured：
      create_app_arg  工厂传给 create_app 的路径（未传则 None）
      app_is_fake     工厂返回值是否就是替身 app
      kw              uvicorn.Config 收到的关键字参数
    """
    captured: dict = {}
    fake_app = object()

    class _FakeServer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            captured["ran"] = True

    def _fake_config(app, **kw):
        captured["app"] = app
        captured["kw"] = kw
        return {"app": app, "kw": kw}

    def _fake_create_app(config_path=None):
        captured["create_app_arg"] = config_path
        return fake_app

    monkeypatch.setenv("BIBIT_FILESEARCH_DATA_ROOT", str(tmp_path / "data"))
    with patch.object(sys, "argv", argv), \
         patch.object(main_mod, "ocr_selfcheck"), \
         patch.object(main_mod, "create_app", _fake_create_app), \
         patch.object(main_mod.uvicorn, "run", lambda *a, **k: None), \
         patch.object(main_mod.uvicorn, "Config", _fake_config), \
         patch.object(main_mod.uvicorn, "Server", _FakeServer):
        main_mod.main()
        # 在 patch 作用域内触发工厂 —— 与 uvicorn 真正起服务时的调用等价。
        # ⚠️ 旧实现走 `uvicorn.run("main:create_app", factory=True)`：那种情况下
        #    uvicorn.Config 根本不会被调用，captured 里没有 "app"（本用例据此判红）。
        app_factory = captured.get("app")
        captured["app_is_callable"] = callable(app_factory)
        captured["app_is_fake"] = bool(callable(app_factory) and app_factory() is fake_app)
    return captured


def test_cli_forwards_config_path_to_the_app_factory(tmp_path, monkeypatch):
    """`serve --config X` → 起服务用的 app 必须由 create_app(X) 构造。"""
    cfg = tmp_path / "custom.yaml"
    cfg.write_text(yaml.safe_dump({"top_k": 17}), encoding="utf-8")

    captured = _run_main(["x", "serve", "--config", str(cfg)], tmp_path, monkeypatch)

    assert captured["app_is_callable"], \
        "uvicorn.Config 应收到可调用的应用工厂，不是字符串（改前是 'main:create_app'）"
    assert captured["app_is_fake"], "工厂应调用 create_app（替身），而不是别的构造路径"
    assert captured["create_app_arg"] == str(cfg), (
        f"create_app 收到 {captured.get('create_app_arg')!r}，"
        f"应为 {str(cfg)!r}——`--config` 不能被丢弃（改前它根本没被读取）"
    )


def test_cli_without_config_passes_none(tmp_path, monkeypatch):
    """不传 --config 时不额外指定路径（回到 env → 用户配置 → 内置默认那条链）。"""
    monkeypatch.delenv("BIBIT_CONFIG", raising=False)
    monkeypatch.delenv("BIBIT_FILESEARCH_CONFIG", raising=False)

    captured = _run_main(["x", "serve"], tmp_path, monkeypatch)

    assert captured["create_app_arg"] is None, "未指定配置时不应编造路径"


def test_cli_port_and_host_still_reach_uvicorn(tmp_path, monkeypatch):
    """改造启动方式时别把 host/port 弄丢（原先 uvicorn.run 的参数）。"""
    captured = _run_main(["x", "serve", "--port", "19999", "--host", "0.0.0.0"],
                         tmp_path, monkeypatch)
    kw = captured["kw"]
    assert kw["port"] == 19999
    assert kw["host"] == "0.0.0.0"
    assert kw["log_level"] == "info"
