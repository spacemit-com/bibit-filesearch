"""知识库管理设置测试（运行时配置层 + Top-K 后端侧）。

覆盖：
- GET /api/settings 契约（code/data 含全部 6 个 key：index_dirs/auto_index/indexable_types/
  exclude_patterns/max_depth/top_k）
- PUT /api/settings 立即生效：top_k → retriever.top_k；max_depth → scanner.max_depth
- PUT 持久化：写 DB，重建 app（同 db）后 GET 仍返回新值（重启保留）
- 索引深度剪枝行为：scanner._walk 按 max_depth 剪枝（0=无限）

用 TestClient（不启真实端口）；index_dirs 指向临时目录，避免真实扫描干扰。
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest


import yaml  # noqa: E402


@pytest.fixture()
def app_ctx(tmp_path, monkeypatch):
    """临时配置 + create_app → (app, client, cfg_path, tmp)。

    2026-09-18：把 EmbedClient 换成 FakeEmbed（与 t4/t5/t7/t8 同做法）。
    此前本文件不打替身，create_app 会真连 127.0.0.1:8082 —— 启动期的
    probe_dim / 看门狗探活都变成真实网络往返，耗时随线上服务负载漂移，
    于是「reconcile 回收 → watcher 补扫又加回」这类时序竞态会随机翻红
    （实测同一用例单独跑 3/3 失败、整文件跑 19/19 通过）。
    cfg 里同步把 models.embed.dim 设为 FakeEmbed.DIM，避免探测维度与配置
    不符触发 reconcile_embed_dim 清库。
    """
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    from fake_embed import DIM as FAKE_DIM  # noqa: E402
    from fake_embed import FakeEmbed  # noqa: E402

    cfg = {
        "storage": {"db_path": str(tmp / "settings.db")},
        "server": {"port": 18097, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "top_k": 50,
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
    return app, TestClient(app), cfg_path, tmp, kb


def test_settings_get_contract(app_ctx):
    """GET /api/settings：契约 + 默认值来自 yaml。"""
    app, client, cfg_path, tmp, kb = app_ctx
    r = client.get("/api/settings")
    j = r.json()
    assert j["code"] == 0
    d = j["data"]
    assert d["index_dirs"] == [str(kb)], "默认 index_dirs 来自 yaml"
    assert d["auto_index"] is True
    assert d["max_depth"] == 0
    assert d["top_k"] == 50
    assert isinstance(d["indexable_types"], list) and "md" in d["indexable_types"]
    assert isinstance(d["exclude_patterns"], list)


def test_settings_put_top_k_effective(app_ctx):
    """PUT top_k → 立即生效：app.state.top_k + retriever 返回条数上限更新。

    2026-09-15：**候选池必须不受影响**。此前这里把 retriever.candidate_pool（当时叫
    top_candidates）也设成 top_k，于是「改显示条数」顺带改了检索深度：Top-K 从 10 调到
    5，块级窗口缩到前 5 块，某文件排第 8 的最佳块被切掉、分数消失、排名掉出视野。
    """
    app, client, cfg_path, tmp, kb = app_ctx
    pool_before = app.state.retriever.candidate_pool
    r = client.put("/api/settings", json={"top_k": 5})
    assert r.json()["code"] == 0
    assert r.json()["data"]["top_k"] == 5
    assert app.state.top_k == 5
    assert app.state.retriever.top_k == 5, "返回条数上限也须同步（否则 PUT 后仍不生效）"
    assert app.state.retriever.candidate_pool == pool_before, \
        "候选池（检索深度）不得随 top_k 变化——否则改显示条数会改排序"


def test_settings_top_k_effective_after_restart(app_ctx):
    """回归 B9：重启后用户设置的 top_k 仍须生效。

    修复前 retriever 的召回池取 `config.search.top_candidates`（嵌套常量 20），
    重启后用户改过的 top_k 只写进 app.state.top_k（无读取方）→ 设置被静默忽略。
    """
    app, client, cfg_path, tmp, kb = app_ctx
    pool_before = app.state.retriever.candidate_pool
    assert client.put("/api/settings", json={"top_k": 3}).json()["code"] == 0

    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    from bibit_filesearch.main import create_app

    app2 = create_app(str(cfg_path))
    try:
        assert app2.state.retriever.top_k == 3, "重启后 top_k 应生效（不再回退常量）"
        assert app2.state.retriever.candidate_pool == pool_before, \
            "重启后候选池仍应等于出厂配置值，与用户 top_k 无关"
    finally:
        app2.state.embed.close() if hasattr(app2.state, "embed") else None


def test_settings_put_max_depth_effective(app_ctx):
    """PUT max_depth → scanner.max_depth 更新（深度剪枝开关生效）。"""
    app, client, cfg_path, tmp, kb = app_ctx
    r = client.put("/api/settings", json={"max_depth": 2})
    assert r.json()["code"] == 0
    assert app.state.scanner.max_depth == 2
    # watcher 同步更新
    assert app.state.watcher.max_depth == 2


def test_settings_put_max_depth_zero_is_unlimited(app_ctx):
    """回归：PUT max_depth=0 不得被 `or 3` 吞成 3（0=无限深，全链路保留）。

    曾有两处 `int(x or 3)` 把显式 0 归一为默认 3（main.py 启动 / api PUT 热更），
    与引擎 0=无限递归语义冲突。修复后：GET/落盘/scanner/watcher 均保留 0。
    """
    app, client, cfg_path, tmp, kb = app_ctx
    # fixture yaml 显式 max_depth: 0 → 启动即无限
    assert app.state.scanner.max_depth == 0, "yaml 显式 0 启动应保留（0=无限深）"
    # 先改成 2，再显式 PUT 0：0 必须保留而非回退默认 3
    r = client.put("/api/settings", json={"max_depth": 2})
    assert r.json()["data"]["max_depth"] == 2
    r = client.put("/api/settings", json={"max_depth": 0})
    j = r.json()
    assert j["code"] == 0
    assert j["data"]["max_depth"] == 0, "显式 0 必须保留（0=无限深）"
    assert app.state.scanner.max_depth == 0
    assert app.state.watcher.max_depth == 0
    with open(cfg_path, encoding="utf-8") as f:
        assert yaml.safe_load(f)["max_depth"] == 0, "yaml 落盘保留 0"
    # 幂等：再次 PUT 0 应 changed=false（0 与当前一致）
    r = client.put("/api/settings", json={"max_depth": 0})
    assert r.json()["data"]["changed"] is False


def test_settings_put_max_depth_negative_normalized(app_ctx):
    """PUT max_depth=-1 → 归一为 0（负数不允许，0=无限深）。"""
    app, client, cfg_path, tmp, kb = app_ctx
    r = client.put("/api/settings", json={"max_depth": -1})
    j = r.json()
    assert j["code"] == 0
    assert j["data"]["max_depth"] == 0, "负数应归一为 0（无限深）"
    assert app.state.scanner.max_depth == 0
    with open(cfg_path, encoding="utf-8") as f:
        assert yaml.safe_load(f)["max_depth"] == 0, "yaml 落盘为归一后的 0"


def test_settings_put_persists_across_restart(app_ctx):
    """PUT 持久化：DB 覆盖 → 重建 app（同 db）→ GET 仍返回新值。"""
    app, client, cfg_path, tmp, kb = app_ctx
    new_dirs = [str(tmp / "kb2")]
    (tmp / "kb2").mkdir()
    r = client.put("/api/settings", json={"index_dirs": new_dirs, "top_k": 3})
    assert r.json()["code"] == 0

    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    from bibit_filesearch.main import create_app

    app2 = create_app(str(cfg_path))
    try:
        from fastapi.testclient import TestClient

        c2 = TestClient(app2)
        j = c2.get("/api/settings").json()
        assert j["data"]["index_dirs"] == new_dirs, "重启后 index_dirs 应保留 DB 覆盖"
        assert j["data"]["top_k"] == 3, "重启后 top_k 应保留 DB 覆盖"
    finally:
        app2.state.embed.close() if hasattr(app2.state, "embed") else None


def test_settings_put_invalid_body(app_ctx):
    """PUT 非对象 → FastAPI 契约层 422（dict 类型校验）。"""
    app, client, cfg_path, tmp, kb = app_ctx
    r = client.put("/api/settings", json=[1, 2])
    assert r.status_code == 422, "非对象请求体应由 FastAPI 拒绝（422）"


def test_scan_walk_max_depth_prune(app_ctx):
    """深度剪枝行为（1-based）：scanner._walk 按 max_depth 剪枝。

    0=无限；1=仅 base 目录本身（不含任何子目录文件）；N=base+下 N-1 层子目录。
    """
    app, client, cfg_path, tmp, kb = app_ctx
    scanner = app.state.scanner
    # 造目录树：kb（base）内 root.txt（d=0）；a/ 第 1 层子目录 → a/top.txt（d=1）；
    # a/b/c/ 第 3 层 → deep.txt（d=3）；a/b/c/d/ 第 4 层 → x.txt（d=4），
    # 用于区分 0=无限 与 1-based 的 N 层语义（第 4 层 > 3 会被剪，测试可抓回归）
    (kb / "root.txt").write_text("root", encoding="utf-8")
    (kb / "a" / "b" / "c" / "d").mkdir(parents=True)
    (kb / "a" / "b" / "c" / "deep.txt").write_text("deep", encoding="utf-8")
    (kb / "a" / "b" / "c" / "d" / "x.txt").write_text("x", encoding="utf-8")
    (kb / "a" / "top.txt").write_text("top", encoding="utf-8")

    def files_of():
        # _walk 期望 str base（内部 base.rstrip("/")）；传 str(kb)
        return {str(Path(r) / f) for r, _, fs in scanner._walk(str(kb)) for f in fs}

    assert scanner.max_depth == 0, "fixture yaml 显式 0 → 启动即无限"
    files = files_of()
    assert str(kb / "root.txt") in files
    assert str(kb / "a" / "top.txt") in files, "第 1 层子目录文件应被遍历（max_depth=0 无限）"
    assert str(kb / "a" / "b" / "c" / "deep.txt") in files
    assert str(kb / "a" / "b" / "c" / "d" / "x.txt") in files, \
        "第 4 层文件也应被遍历（0=无限，远超默认 3 也须遍历）"

    scanner.max_depth = 1
    files1 = files_of()
    assert str(kb / "root.txt") in files1, "max_depth=1 应索引 base 内文件"
    assert str(kb / "a" / "top.txt") not in files1, "1=仅 base 本身：子目录文件不索引"
    assert str(kb / "a" / "b" / "c" / "deep.txt") not in files1

    scanner.max_depth = 2
    files2 = files_of()
    assert str(kb / "root.txt") in files2
    assert str(kb / "a" / "top.txt") in files2, "max_depth=2 = base+1 层子目录"
    assert str(kb / "a" / "b" / "c" / "deep.txt") not in files2, "第 3 层应被剪枝"
    assert str(kb / "a" / "b" / "c" / "d" / "x.txt") not in files2


def test_settings_reconcile_excludes_parent_dir_pattern(app_ctx, monkeypatch):
    """回归 B5：排除规则写父目录名 → 该目录下已入库文件必须被回收。

    修复前 reconcile 只拿「完整路径 + 文件名」比对规则：规则 `skipme` 与路径
    `/…/skipme/no-index.txt`、文件名 `no-index.txt` 都对不上 → 不回收 → 文件留在
    库里仍可被检索到（用户实测：加了排除规则保存后仍搜得到）。
    """
    app, client, cfg_path, tmp, kb = app_ctx
    (kb / "skipme").mkdir()
    (kb / "keep.txt").write_text("keep", encoding="utf-8")
    (kb / "skipme" / "no-index.txt").write_text("should be excluded", encoding="utf-8")
    for p in (kb / "keep.txt", kb / "skipme" / "no-index.txt"):
        _insert_indexed_row(app.state.db, p)
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)

    r = client.put("/api/settings", json={"exclude_patterns": ["skipme"]})
    assert r.json()["code"] == 0
    db = app.state.db
    assert db.get_file_by_path(str(kb / "skipme" / "no-index.txt")) is None, \
        "规则命中父目录名 skipme → 其下已入库文件必须回收"
    assert db.get_file_by_path(str(kb / "keep.txt")) is not None, "无关文件不受影响"


def test_settings_reconcile_excludes_filename_glob(app_ctx, monkeypatch):
    """排除规则写文件名 glob（`*.log`）→ 任意层级已入库的 .log 回收。"""
    app, client, cfg_path, tmp, kb = app_ctx
    (kb / "a" / "b").mkdir(parents=True)
    (kb / "a" / "b" / "app.log").write_text("log", encoding="utf-8")
    (kb / "a" / "b" / "app.txt").write_text("txt", encoding="utf-8")
    for p in (kb / "a" / "b" / "app.log", kb / "a" / "b" / "app.txt"):
        _insert_indexed_row(app.state.db, p)
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)

    r = client.put("/api/settings", json={"exclude_patterns": ["*.log"]})
    assert r.json()["code"] == 0
    db = app.state.db
    assert db.get_file_by_path(str(kb / "a" / "b" / "app.log")) is None
    assert db.get_file_by_path(str(kb / "a" / "b" / "app.txt")) is not None


def _insert_indexed_row(db, path: Path, status: str = "completed"):
    """直接向 files 表插入一条"已入库 completed"记录（模拟此前扫描已完成的文件）。"""
    import uuid

    meta = {
        "fileId": "f-" + uuid.uuid4().hex[:12],
        "filePath": str(path),
        "fileName": path.name,
        "fileType": path.suffix.lstrip(".") or "txt",
        "fileSize": 1,
        "createdAt": "",
        "modifiedAt": "",
        "accessedAt": "",
        "author": "",
        "directory": str(path.parent),
    }
    return db.upsert_file(meta, "hash-%s" % path.name, "", status=status)


def test_settings_reconcile_max_depth_reclaims_deep_files(app_ctx, monkeypatch):
    """深度回收（1-based）：max_depth 从大改小（0→1）→ 超出新深度的已入库文件被回收。

    现状 bug：reconcile 只按目录范围/类型/排除回收、不看深度——scanner._walk 只拦
    新扫描、hash 幂等跳过旧文件，导致 a/mid.txt 在 max_depth=1（仅 base）时仍留在
    库中可搜。1-based 口径：文件所在目录深度 d >= max_depth 即回收。
    """
    app, client, cfg_path, tmp, kb = app_ctx
    # 造多层结构（相对 kb 的目录深度）：top.txt d=0 / a/mid.txt d=1 / a/b/deep.txt d=2
    (kb / "a" / "b").mkdir(parents=True)
    (kb / "top.txt").write_text("top", encoding="utf-8")
    (kb / "a" / "mid.txt").write_text("mid", encoding="utf-8")
    (kb / "a" / "b" / "deep.txt").write_text("deep", encoding="utf-8")
    for p in (kb / "top.txt", kb / "a" / "mid.txt", kb / "a" / "b" / "deep.txt"):
        _insert_indexed_row(app.state.db, p)
    assert app.state.db.get_file_by_path(str(kb / "a" / "b" / "deep.txt")) is not None

    # 本测试只验 reconcile 深度回收：把 PUT 触发的重扫线程置空，避免真实文件被重入队
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)

    # 0=无限 → PUT 1（收窄，1=仅 base 本身）：a/mid.txt（d=1 >= 1）与
    # a/b/deep.txt（d=2 >= 1）都回收；只有 base 内 top.txt（d=0）保留
    r = client.put("/api/settings", json={"max_depth": 1})
    assert r.json()["data"]["changed"] is True
    db = app.state.db
    assert db.get_file_by_path(str(kb / "a" / "b" / "deep.txt")) is None, \
        "深度 d=2 >= max_depth=1 的已入库文件应被回收"
    assert db.get_file_by_path(str(kb / "a" / "mid.txt")) is None, \
        "1=仅 base：直接子目录里的 a/mid.txt（d=1 >= 1）也要回收"
    assert db.get_file_by_path(str(kb / "top.txt")) is not None, "base 内文件（d=0）不受影响"

    # 1 → PUT 0（无限）：不再按深度回收；重新入库的深层文件应保留
    _insert_indexed_row(db, kb / "a" / "mid.txt")
    _insert_indexed_row(db, kb / "a" / "b" / "deep.txt")
    r = client.put("/api/settings", json={"max_depth": 0})
    assert r.json()["data"]["changed"] is True
    assert app.state.scanner.max_depth == 0
    assert db.get_file_by_path(str(kb / "a" / "b" / "deep.txt")) is not None, \
        "max_depth=0（无限深）不应按深度回收任何文件"
    assert db.get_file_by_path(str(kb / "a" / "mid.txt")) is not None


def test_settings_reconcile_max_depth_uses_matching_base(app_ctx, monkeypatch):
    """深度按文件实际前缀匹配的 index_dir base 计算（嵌套 index_dirs 取最长 base）。

    1-based：max_depth=2 = base+1 层子目录，文件所在目录深度 d >= 2 回收。
    """
    app, client, cfg_path, tmp, kb = app_ctx
    sub = kb / "sub"  # kb 与 kb/sub 同时是索引根
    (sub / "x" / "y").mkdir(parents=True)
    (kb / "sub" / "top.txt").write_text("t", encoding="utf-8")
    (kb / "sub" / "x" / "mid.txt").write_text("m", encoding="utf-8")
    (kb / "sub" / "x" / "y" / "deep.txt").write_text("d", encoding="utf-8")
    for p in (kb / "sub" / "top.txt", kb / "sub" / "x" / "mid.txt",
              kb / "sub" / "x" / "y" / "deep.txt"):
        _insert_indexed_row(app.state.db, p)
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)

    # kb/sub/x/y/deep.txt 相对 kb 深度 4、相对 kb/sub 深度 2；kb/sub/x/mid.txt 相对
    # kb/sub 深度 1、相对 kb 深度 2。PUT 后 index_dirs=[kb, kb/sub]（1-based：2=base+1 层）
    r = client.put("/api/settings", json={"index_dirs": [str(kb), str(sub)], "max_depth": 2})
    assert r.json()["data"]["changed"] is True
    db = app.state.db
    # 最长匹配 base=kb/sub：mid.txt d=1 < 2 → 保留；deep.txt d=2 >= 2 → 回收。
    # 若误按 kb 计深度（mid d=2、deep d=4）两者都会被错删——保留 mid 即证明取了最长 base。
    assert db.get_file_by_path(str(kb / "sub" / "x" / "mid.txt")) is not None
    assert db.get_file_by_path(str(kb / "sub" / "x" / "y" / "deep.txt")) is None, \
        "1-based max_depth=2 下 d=2 >= 2 应回收"
    assert db.get_file_by_path(str(kb / "sub" / "top.txt")) is not None


# ─────────────────────────────────────────────────────────────
# index_dirs 路径校验（2026-09-07：PUT 保存前拦截坏路径）
# ─────────────────────────────────────────────────────────────

def test_settings_put_rejects_nonexistent_dir(app_ctx, monkeypatch):
    """index_dirs 含不存在目录 → 整单拒绝，不落盘。"""
    app, client, cfg_path, tmp, kb = app_ctx
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)
    bad = str(tmp / "no_such_dir")
    r = client.put("/api/settings", json={"index_dirs": [str(kb), bad]})
    j = r.json()
    assert j["code"] == 1001  # ERR_PARAM
    assert "不存在" in j["message"]
    assert j["data"]["changed"] is False
    # 未落盘：config 里 index_dirs 仍是原值
    assert client.get("/api/settings").json()["data"]["index_dirs"] == [str(kb)]


def test_settings_put_rejects_file_as_dir(app_ctx, monkeypatch):
    """index_dirs 含文件路径 → 拒绝（不是目录）。"""
    app, client, cfg_path, tmp, kb = app_ctx
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)
    f = tmp / "afile.txt"
    f.write_text("x", encoding="utf-8")
    r = client.put("/api/settings", json={"index_dirs": [str(f)]})
    j = r.json()
    assert j["code"] == 1001  # ERR_PARAM
    assert "不是目录" in j["message"]
    assert client.get("/api/settings").json()["data"]["index_dirs"] == [str(kb)]


def test_settings_put_expands_tilde(app_ctx, monkeypatch):
    """~/ 前缀展开为 home；合法目录保存成功。"""
    app, client, cfg_path, tmp, kb = app_ctx
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)
    r = client.put("/api/settings", json={"index_dirs": [f"~/{kb.name}"]})
    # kb 在 tmp 下非 home 下，展开后不存在 → 拒绝（验证展开发生而非存字面量）
    j = r.json()
    assert j["code"] == 1001  # ERR_PARAM, "~/xxx 展开后应指向 home 下路径（不存在则拒绝）"
    assert "不存在" in j["message"]
    # 但若路径恰好等于 home/kb.name 则…… 这里 kb 在 tmp，home 下无此目录
    assert client.get("/api/settings").json()["data"]["index_dirs"] == [str(kb)]


def test_settings_put_rejects_relative_path(app_ctx, monkeypatch):
    """相对路径 → 拒绝（须绝对路径）。"""
    app, client, cfg_path, tmp, kb = app_ctx
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)
    r = client.put("/api/settings", json={"index_dirs": ["Documents"]})
    j = r.json()
    assert j["code"] == 1001  # ERR_PARAM
    assert "绝对路径" in j["message"]
    assert client.get("/api/settings").json()["data"]["index_dirs"] == [str(kb)]


def test_settings_put_valid_dirs_passes(app_ctx, monkeypatch):
    """全合法目录 → 保存成功（changed=True）。"""
    app, client, cfg_path, tmp, kb = app_ctx
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)
    kb2 = tmp / "kb2"
    kb2.mkdir()
    r = client.put("/api/settings", json={"index_dirs": [str(kb), str(kb2)]})
    j = r.json()
    assert j["code"] == 0
    assert j["data"]["changed"] is True
    assert client.get("/api/settings").json()["data"]["index_dirs"] == [str(kb), str(kb2)]


def test_settings_put_tilde_normalized_idempotent(app_ctx, monkeypatch):
    """D1 回归：~/x 规范化后与已落盘绝对路径判为幂等（不再每次保存误判 changed）。

    用户填 ~/Documents 保存 → 落盘 /home/<u>/Documents；表单仍显 ~/Documents，
    再保存应 changed=false（不空转重扫）。monkeypatch home 指向真实 kb 所在 tmp。
    """
    import pathlib
    app, client, cfg_path, tmp, kb = app_ctx
    monkeypatch.setattr(app.state.scanner, "restart", lambda: None)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp)
    # 先把 index_dirs 切到另一个目录（确保非初始值，首次 ~/ 保存是"真变化"）
    other = tmp / "other"
    other.mkdir()
    assert client.put("/api/settings", json={"index_dirs": [str(other)]}).json()["data"]["changed"] is True
    # 首次 ~/kb.name → 展开为 tmp/kb.name = kb（存在）→ 保存成功且落盘绝对路径
    r1 = client.put("/api/settings", json={"index_dirs": [f"~/{kb.name}"]})
    assert r1.json()["code"] == 0
    assert r1.json()["data"]["changed"] is True
    assert client.get("/api/settings").json()["data"]["index_dirs"] == [str(kb)]
    # 二次：同样 ~/kb.name 再存 → 规范化后与现值相同 → changed=False（幂等）
    r2 = client.put("/api/settings", json={"index_dirs": [f"~/{kb.name}"]})
    assert r2.json()["code"] == 0
    assert r2.json()["data"]["changed"] is False, \
        "~/x 与落盘绝对路径应判幂等，不得每次保存空转重扫"
