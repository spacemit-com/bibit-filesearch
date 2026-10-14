"""auto_index=false 必须在服务启动时生效（2026-09-09 审计暴露）。

现象：PUT {"auto_index": false} 生效（watcher 停），但重启服务后 journal 仍打
「inotify 监听启动」——开关只在 PUT 路径生效，启动路径被忽略。
根因：main.create_app 无条件 watcher.start()。

断言：auto_index=false 启动 → watcher 未启动（无 inotify fd、无读线程）；
auto_index=true 启动 → 正常启动（对照组，防止「永不启动」假修复）。
"""
import os

import pytest
import yaml


def _make_app(tmp_path, auto_index: bool):
    tmp = tmp_path / f"data_{auto_index}"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 甲\n\n内容。\n", encoding="utf-8")
    cfg = {
        "storage": {"db_path": str(tmp / "search.db")},
        "server": {"port": 18098, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "auto_index": auto_index,
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    os.environ["BIBIT_CONFIG"] = str(cfg_path)

    from bibit_filesearch.main import create_app

    return create_app(str(cfg_path))


@pytest.mark.parametrize("auto_index,expected", [(False, False), (True, True)])
def test_watcher_start_respects_auto_index(tmp_path, auto_index, expected):
    app = _make_app(tmp_path, auto_index)
    w = app.state.watcher
    started = bool(w._thread and w._thread.is_alive())
    assert started is expected, (
        f"auto_index={auto_index} 时 watcher 启动状态应为 {expected}（实际 {started}）——"
        "启动路径必须读 auto_index，不能无条件 start()"
    )
    if expected:
        w.stop()
