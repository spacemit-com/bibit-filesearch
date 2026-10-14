"""/api/search 错误路径契约测试（2026-09-09 用户实测暴露）。

现象：embed 掉线时 GUI 检索请求返回 HTTP 500 "Internal Server Error"，
前端只能显示"HTTP 500"而不是可读原因。

根因：api/search.py 在 retriever 返回的 result 上做
`result.setdefault("data", {})["debug"] = {...}`；出错时 retriever 返回
`{"data": None}`，setdefault 见到已存在的 None 键不会替换 → `None["debug"]`
抛 TypeError → FastAPI 500。

断言：embed 不可用时 /api/search 仍是 HTTP 200 + {code:1004, data:null}
（契约错误，不是 500），且 message 可读。
"""
import os

import pytest
import yaml


class _UpEmbed:
    """可用的确定性 embed 替身（用于先把文件索引进库，制造候选）。

    dim 必须与 create_app 从 config 读到的 models.embed.dim 一致（默认 1024），
    否则 Retriever 增量索引会抛维度不符。
    """

    dim = 1024

    def _vec(self, text: str):
        return [float((hash((text, i)) % 97) + 1) for i in range(self.dim)]

    def embed_one(self, text: str):
        return self._vec(text)

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass


class _DownEmbed(_UpEmbed):
    """始终抛连接错误的 embed 替身（模拟 :8082 掉线）。"""

    def __init__(self, url: str = "http://127.0.0.1:8082"):
        self.url = url
        self.dim = 1024

    def embed_one(self, text: str):
        raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")

    def embed_batch(self, texts):
        raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")

    def ping(self) -> bool:
        return False


@pytest.fixture()
def app_ctx(tmp_path):
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    (kb / "gtk.md").write_text("# GTK window rendering\n\nnotes\n", encoding="utf-8")
    cfg = {
        "storage": {"db_path": str(tmp / "search.db")},
        "server": {"port": 18096, "cors_origins": []},
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
    # 2026-09-23（池子化）：本用例考的是**检索错误路径**，不该有后台索引在跑。
    # create_app 会按生产语义启动 scanner（及其常驻消费者）与 watcher，而两者都会
    # 去 offer 同一个 kb/gtk.md；本用例紧接着手工 `index_file` 把该文件置成
    # completed。若后台消费者在那之后又抓到该文件（此刻 embed 已被换成 _DownEmbed），
    # 它会失败并把状态改成 failed → 检索按 completed 过滤候选 → 候选为空 →
    # 直接返回 code 0「success, total 0」，绕过了本用例要验的错误路径
    # （实测：全量随机顺序下偶发，单跑 8/8 通过）。
    # 生产上不存在这个竞争（消费者是唯一执行者，没有「测试手工索引」这条旁路），
    # 故在夹具里停掉后台组件，而不是改产品逻辑去迁就测试。
    app.state.scanner.stop()
    app.state.watcher.stop()
    # stop() 只置停止位：**已经在跑的补扫线程那一轮 offer 不受影响**
    # （`_delayed_backfill` 的等待循环会在 scanner 停下后 break，但紧接着仍执行
    #  一轮 `_backfill_all()`）。不等它跑完就手工入库，它随后会把该文件重新 offer
    # 成 queued 行，检索按 completed 过滤 → 候选为空 → 返回 code 0，绕过本用例。
    # 故 join 两个线程，等后台真的静默下来再入库。
    for _th in (getattr(app.state.watcher, "_backfill_thread", None),
                getattr(app.state.watcher, "_thread", None),
                getattr(getattr(app.state.scanner, "worker", None), "_thread", None)):
        if _th is not None and _th.is_alive():
            _th.join(timeout=5.0)
    # 用可用 embed 先入库，保证检索走到 embed 路径（否则空候选会提前返回空结果）
    app.state.embed = _UpEmbed()
    app.state.retriever.embed = app.state.embed
    app.state.indexer.embed = app.state.embed
    app.state.indexer.index_file(str(kb / "gtk.md"))
    yield app, TestClient(app)
    app.state.embed.close()


def _go_down(app):
    down = _DownEmbed()
    app.state.embed = down
    app.state.retriever.embed = down
    app.state.indexer.embed = down


def test_search_returns_contract_error_when_embed_down(app_ctx):
    """embed 掉线 → HTTP 200 + code 1004 + data null（不得是 HTTP 500）。"""
    app, client = app_ctx
    _go_down(app)

    r = client.post("/api/search", json={"query": "GTK", "page": 1, "pageSize": 5})
    assert r.status_code == 200, f"应为契约错误而非 HTTP {r.status_code}: {r.text[:200]}"
    j = r.json()
    assert j["code"] == 1004, f"应返回内部错误码: {j}"
    assert j["data"] is None
    assert j["message"], "message 应非空（前端要展示可读原因）"


def test_search_no_debug_key_when_error(app_ctx):
    """错误响应不应被 debug 字段污染（原 bug 正是在此崩溃）。"""
    app, client = app_ctx
    _go_down(app)

    j = client.post("/api/search", json={"query": "GTK"}).json()
    assert j.get("data") is None
