"""设置变更时「停 watcher → 回收文件 → 再放开 watcher」的次序必须成立（2026-09-23）。

## 这条守的是什么

`PUT /api/settings` 会按新配置回收不再属于索引范围的文件（改了排除规则/目录/深度/
格式）。回收本身是删库行；而 watcher 的补扫会**把库行加回来**。

补扫线程拿着的是它启动那一刻的 index_dirs / exclude_patterns。如果 PUT 的次序是
「先回收、后 stop()」，一个正在跑的补扫就会用**旧配置**继续 offer：

  · 该被排除的文件 → 旧 excludes 下它不该被排除 → 被加回库（用户：加了排除规则，
    文件还是搜得到）
  · 不该被排除的文件 → 也可能在「加回来 → 再回收」的来回中被删掉

实测（未改动的基线代码同样中招，非池子化引入）：同一份
`tests/test_t12_settings.py` 连跑 20 次失败 2 次，分别是 `app.log` 该回收没回收、
`top.txt`/`keep.txt` 不该删却被删；单独跑该文件 19/19 通过（单跑时补扫通常已结束）。

## 本文件的两条断言

1. **次序**：PUT 里 `watcher.stop()` 必须发生在第一个 `purge` 之前。
2. **补扫会看停止位**：`_stop` 已置时，`_backfill_dir` 不再往池里 offer。
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def app_ctx(monkeypatch):
    """临时配置 + create_app（照 t12 的做法：EmbedClient 换 FakeEmbed）。"""
    tmp = Path(tempfile.mkdtemp(prefix="bibit_reclaim_")) / "data"
    tmp.mkdir(parents=True)
    kb = tmp / "kb"
    kb.mkdir()
    from fake_embed import DIM as FAKE_DIM
    from fake_embed import FakeEmbed

    cfg = {
        "storage": {"db_path": str(tmp / "settings.db")},
        "server": {"port": 18096, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "top_k": 50,
        "models": {"embed": {"dim": FAKE_DIM}},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("BIBIT_CONFIG", str(cfg_path))

    from fastapi.testclient import TestClient

    from bibit_filesearch import model_client as mc
    from bibit_filesearch.main import create_app

    monkeypatch.setattr(mc, "EmbedClient", FakeEmbed)
    app = create_app(str(cfg_path))
    yield app, TestClient(app), kb
    try:
        app.state.scanner.stop()
        app.state.watcher.stop()
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(tmp.parent, ignore_errors=True)


def _insert_row(db, path: Path):
    """直接插一条「已入库 completed」记录（模拟此前扫描已完成）。"""
    import uuid

    meta = {
        "fileId": "f-" + uuid.uuid4().hex[:12],
        "filePath": str(path),
        "fileName": path.name,
        "fileType": path.suffix.lstrip(".") or "txt",
        "fileSize": 1,
        "createdAt": "", "modifiedAt": "", "accessedAt": "", "author": "",
        "directory": str(path.parent),
    }
    return db.upsert_file(meta, "hash-%s" % path.name, "", status="completed")


def test_watcher_stops_before_reclaim_on_settings_change(app_ctx, monkeypatch):
    """★ PUT settings：`watcher.stop()` 必须早于第一个 `purge`。

    反了就会让「还在跑的补扫用旧配置把刚回收的文件加回来」——这正是 t12 那几条
    用例在全量套件里偶发失败的原因（单跑绿、合跑红）。
    """
    app, client, kb = app_ctx
    (kb / "skipme").mkdir()
    (kb / "keep.txt").write_text("keep", encoding="utf-8")
    (kb / "skipme" / "no-index.txt").write_text("excluded", encoding="utf-8")
    for p in (kb / "keep.txt", kb / "skipme" / "no-index.txt"):
        _insert_row(app.state.db, p)

    events: list[str] = []

    real_stop = app.state.watcher.stop

    def _stop():
        events.append("watcher.stop")
        return real_stop()

    monkeypatch.setattr(app.state.watcher, "stop", _stop)

    pool = app.state.pool
    real_purge = pool.purge

    def _purge(path):
        events.append("purge")
        return real_purge(path)

    monkeypatch.setattr(pool, "purge", _purge)

    r = client.put("/api/settings", json={"exclude_patterns": ["skipme"]})
    assert r.json()["code"] == 0, r.text

    assert "purge" in events, f"该回收的文件必须走池子回收：{events}"
    assert "watcher.stop" in events, f"PUT 必须停 watcher：{events}"
    assert events.index("watcher.stop") < events.index("purge"), (
        "次序反了：回收发生在 watcher 停止之前 → 正在跑的补扫会用旧配置把文件加回来。"
        f"实际次序：{events}"
    )
    # 结果侧：该回收的没了、不该回收的还在
    assert app.state.db.get_file_by_path(str(kb / "skipme" / "no-index.txt")) is None
    assert app.state.db.get_file_by_path(str(kb / "keep.txt")) is not None


def test_backfill_stops_offering_after_stop(monkeypatch):
    """★ 停止位已置 → `_backfill_dir` 一个都不再 offer。

    这是上一条能成立的前提：光有次序还不够，补扫自身也得在看得到停止位时收手。
    """
    from bibit_filesearch.db import Database
    from bibit_filesearch.indexer import Indexer
    from bibit_filesearch.watcher import Watcher

    tmp = Path(tempfile.mkdtemp(prefix="bibit_bf_"))
    try:
        for i in range(30):
            (tmp / f"f{i}.txt").write_text(f"content {i}\n", encoding="utf-8")

        class _E:
            dim = 8

            def embed_one(self, t):
                return [1.0] * 8

            def embed_batch(self, ts):
                return [[1.0] * 8 for _ in ts]

            def ping(self):
                return True

            def close(self):
                pass

        class _Cfg:
            def __init__(self, dirs):
                self.index_dirs = dirs
                self.exclude_patterns = []
                self.max_depth = 0

                class _Scan:
                    def get(self, k, d=None):
                        return {"io_throttle_ms": 0}.get(k, d)

                self.scan = _Scan()

        db = Database(str(tmp / "t.db"))
        idx = Indexer(db, _E())
        w = Watcher(_Cfg([str(tmp)]), db, idx)
        w._stop.set()          # 已停止
        w._backfill_dir(str(tmp))
        assert db.count_files() == 0, (
            f"停止后补扫不得再入池，实际库里 {db.count_files()} 行："
            f"{[r['path'] for r in db.all_files()][:5]}"
        )
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
