"""知识库路径重复项的归一化（2026-09-20）。

守护语义：同一个目录在 index_dirs 里出现多次 = 同一个意图，保存时静默去重
（保留首次出现的顺序），并且**运行时读取侧也去重** —— 用户手改 config.yaml
写重复项同样不能放大扫描计数。

背景（用户报告）：知识库路径允许填相同的路径（例如两个 test2），保存并索引后
「扫描中的文件数目」随重复份数**线性增加**：真实文件 2 个时，[A] 报 2、
[A,A] 报 4、[A,A,A] 报 6。

链条三段：
  ① 前端 textarea 按行 split，不去重；
  ② 后端 `_normalize_index_dirs` 只 strip/展开 ~/拒相对路径，不去重；
  ③ `scanner._count_files` 与 `_run_traverse` 与 `for base in self.index_dirs`
     累加/遍历 → 重复项各算一遍、各走一遍（白扫磁盘）。

连带害处：`put_settings` 的幂等判断 `_same` 因重复项失真 → 反复触发
reconcile + 全量重扫。

不覆盖（另一类判据，待定）：嵌套重叠（[/kb, /kb/sub]）——两处判据是
「路径前缀包含」而非「完全相同」，与本次不同。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from bibit_filesearch.settings import SettingsService


# ---------------------------------------------------------------- 单元：规范化

def test_normalize_dedups_duplicates_keeps_first_seen_order():
    """[A, B, A] → [A, B]：重复项丢弃，首次出现的顺序保留。"""
    from bibit_filesearch.api.settings import _normalize_index_dirs

    norm, errors = _normalize_index_dirs(["/kb/a", "/kb/b", "/kb/a"])
    assert errors == []
    assert norm == ["/kb/a", "/kb/b"]


def test_normalize_dedups_after_tilde_expansion(monkeypatch):
    """~/x 与展开后的绝对路径是同一个目录 → 去重（规范化后再判重）。"""
    from bibit_filesearch.api.settings import _normalize_index_dirs

    home = Path("/home/u")
    monkeypatch.setattr(Path, "home", lambda: home)
    norm, errors = _normalize_index_dirs(["/home/u/kb", "~/kb"])
    assert errors == []
    assert norm == ["/home/u/kb"]


def test_normalize_keeps_distinct_dirs_untouched():
    """不同目录不受影响，顺序原样保留（不排序）。"""
    from bibit_filesearch.api.settings import _normalize_index_dirs

    norm, errors = _normalize_index_dirs(["/kb/b", "/kb/a"])
    assert errors == []
    assert norm == ["/kb/b", "/kb/a"]


def test_normalize_still_rejects_bad_entries_with_dups():
    """去重不得掩盖错误：相对路径仍要报错。"""
    from bibit_filesearch.api.settings import _normalize_index_dirs

    norm, errors = _normalize_index_dirs(["/kb/a", "relative/x", "/kb/a"])
    assert errors, "相对路径必须报错"
    assert norm == ["/kb/a"]


# ---------------------------------------------------------------- 读取侧去重

def _svc_from(tmp_path, dirs):
    """写一份含指定 index_dirs 的用户配置并构造 SettingsService。"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"index_dirs": dirs, "config_version": 4}, allow_unicode=True),
        encoding="utf-8",
    )
    return SettingsService(config_path=cfg), cfg


def test_get_all_dedups_hand_edited_config(tmp_path):
    """用户手改 config.yaml 写重复项 → 运行时生效值已去重。"""
    svc, _ = _svc_from(tmp_path, ["/kb/a", "/kb/a"])
    assert svc.get_all()["index_dirs"] == ["/kb/a"]


def test_get_dedups(tmp_path):
    """get("index_dirs") 与 get_all 口径一致。"""
    svc, _ = _svc_from(tmp_path, ["/kb/a", "/kb/b", "/kb/a", "/kb/b"])
    assert svc.get("index_dirs") == ["/kb/a", "/kb/b"]


# ---------------------------------------------------------------- PUT 落盘

@pytest.fixture()
def client_ctx(tmp_path, monkeypatch):
    """临时配置 + create_app → (client, cfg_path, kb)。EmbedClient 打替身。"""
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    (kb / "one.txt").write_text("hello", encoding="utf-8")
    (kb / "two.txt").write_text("world", encoding="utf-8")

    from fake_embed import DIM as FAKE_DIM
    from fake_embed import FakeEmbed

    cfg = {
        "storage": {"db_path": str(tmp / "dedup.db")},
        "server": {"port": 18096, "cors_origins": []},
        "index_dirs": [str(kb)],
        "exclude_patterns": [],
        "models": {"embed": {"dim": FAKE_DIM}},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    from fastapi.testclient import TestClient

    from bibit_filesearch import model_client as mc
    from bibit_filesearch.main import create_app

    monkeypatch.setattr(mc, "EmbedClient", FakeEmbed)
    app = create_app(str(cfg_path))
    with TestClient(app) as client:
        yield client, cfg_path, kb


def test_put_dedups_and_persists_single_entry(client_ctx):
    """PUT [A, A] → 返回与落盘都只有一份 A。"""
    client, cfg_path, kb = client_ctx
    r = client.put("/api/settings", json={"index_dirs": [str(kb), str(kb)]})
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0, body
    assert body["data"]["index_dirs"] == [str(kb)]

    on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert on_disk["index_dirs"] == [str(kb)], "重复项不得落盘"


def test_put_same_dir_twice_is_idempotent(client_ctx):
    """当前已是 [A] 时，PUT [A, A] 不算变化（幂等）——否则反复重扫。"""
    client, _, kb = client_ctx
    client.put("/api/settings", json={"index_dirs": [str(kb)]})
    r = client.put("/api/settings", json={"index_dirs": [str(kb), str(kb)]})
    assert r.json()["data"]["changed"] is False, r.json()


def test_put_duplicate_via_get_all_reflects_dedup(client_ctx):
    """GET /api/settings 也返回去重后的列表。"""
    client, _, kb = client_ctx
    client.put("/api/settings", json={"index_dirs": [str(kb), str(kb)]})
    got = client.get("/api/settings").json()["data"]["index_dirs"]
    assert got == [str(kb)]


# ---------------------------------------------------------------- 端到端：计数

def test_scan_total_not_amplified_by_duplicate_dirs(client_ctx):
    """核心回归：重复路径不得放大扫描总数（真实文件 2 → 报 2，不是 4）。"""
    client, _, kb = client_ctx
    a = str(kb)
    client.put("/api/settings", json={"index_dirs": [a, a]})

    client.post("/api/scan")
    prog = client.get("/api/scan/progress").json()["data"]
    total = prog["scanPhase"]["total"]
    assert total == 2, f"重复路径放大了扫描总数：真实 2 个文件，报 {total}"


def test_scan_total_distinct_dirs_unchanged(tmp_path, monkeypatch):
    """两个不同目录（各 2 个文件）→ 报 4，去重不误伤正常多目录。"""
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb1 = tmp / "kb1"
    kb2 = tmp / "kb2"
    for d in (kb1, kb2):
        d.mkdir()
        (d / "x.txt").write_text("x", encoding="utf-8")
        (d / "y.txt").write_text("y", encoding="utf-8")

    from fake_embed import DIM as FAKE_DIM
    from fake_embed import FakeEmbed

    cfg = {
        "storage": {"db_path": str(tmp / "multi.db")},
        "server": {"port": 18095, "cors_origins": []},
        "index_dirs": [str(kb1), str(kb2)],
        "exclude_patterns": [],
        "models": {"embed": {"dim": FAKE_DIM}},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    from fastapi.testclient import TestClient

    from bibit_filesearch import model_client as mc
    from bibit_filesearch.main import create_app

    monkeypatch.setattr(mc, "EmbedClient", FakeEmbed)
    app = create_app(str(cfg_path))
    with TestClient(app) as client:
        client.post("/api/scan")
        prog = client.get("/api/scan/progress").json()["data"]
        assert prog["scanPhase"]["total"] == 4, prog["scanPhase"]
