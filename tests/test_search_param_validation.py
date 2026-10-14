"""/api/search 分页参数校验（2026-09-09「检查还有没有 bug」审计暴露）。

现象：POST /api/search {"page":"abc"} / {"pageSize":"x"} → HTTP 500 Internal Server Error。
根因：api/search.py 两个 handler 先 `int(payload.get("page") or 1)` 强转，非数字串在
handler 里抛 ValueError（FastAPI 500）；而纯筛选分支（空 query + filters）又完全绕过
retriever._validate，`-(-total // page_size)` 同样炸。显式 page=0 被 `or 1` 吞成 1 静默放行。

断言：非法分页一律 HTTP 200 + code 1001（契约信封），不得 500；显式 0 必须报错而非吞成 1。
"""
import os

import pytest
import yaml


@pytest.fixture()
def app_ctx(tmp_path):
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    (kb / "gtk.md").write_text("# GTK window rendering\n\nnotes\n", encoding="utf-8")
    cfg = {
        "storage": {"db_path": str(tmp / "search.db")},
        "server": {"port": 18097, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "top_k": 50,
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    from fastapi.testclient import TestClient
    from bibit_filesearch.main import create_app

    app = create_app(str(cfg_path))
    yield app, TestClient(app)


def _post(client, payload):
    r = client.post("/api/search", json=payload)
    assert r.status_code == 200, f"应为契约错误而非 HTTP {r.status_code}: {r.text[:200]}"
    return r.json()


@pytest.mark.parametrize("payload", [
    {"query": "gpu", "page": "abc"},
    {"query": "gpu", "pageSize": "x"},
    {"query": "gpu", "page": 0},
    {"query": "gpu", "pageSize": 0},
    {"query": "gpu", "page": -1},
    {"query": "gpu", "pageSize": 101},
    {"query": "gpu", "page": 1.5},
    {"query": "gpu", "page": True},          # bool 是 int 子类——不能当合法页码
    # NL 净化后 keywords 为空 → 走 retriever 的纯筛选分支（此前该分支绕过 _validate）
    {"query": "本地搜索md文件", "page": "abc"},
    {"query": "本地搜索md文件", "pageSize": 0},
])
def test_invalid_pagination_returns_contract_error(app_ctx, payload):
    """非法分页 → code 1001 信封（不是 HTTP 500，也不是静默修正）。"""
    _, client = app_ctx
    j = _post(client, payload)
    assert j["code"] == 1001, f"应返回 1001 参数错误: {j}"
    assert "page" in j["message"].lower() or "pageSize" in j["message"], j


def test_pure_filter_via_nl_still_works(app_ctx):
    """对照组：NL 净化出空 keywords + filters + 合法分页仍走纯筛选（修 500 不能打死该分支）。

    注意：本 fixture 不注入假 embed，kb 文件未必入库 → 只断言走到纯筛选分支
    （code 0 + 契约字段齐全），不断言命中条数。
    """
    _, client = app_ctx
    j = _post(client, {"query": "本地搜索md文件", "page": 1, "pageSize": 5})
    assert j["code"] == 0, j
    assert "results" in j["data"] and "total" in j["data"], j
    assert j["data"]["page"] == 1 and j["data"]["pageSize"] == 5, j


def test_empty_query_and_empty_filters_still_param_error(app_ctx):
    """query 与 filters 都空 → 仍是 1001（原契约语义不变）。"""
    _, client = app_ctx
    j = _post(client, {"query": ""})
    assert j["code"] == 1001, j
