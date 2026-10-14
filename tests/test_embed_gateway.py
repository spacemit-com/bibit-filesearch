"""embed 端点接入测试（2026-09-20）。

背景：推理后端从「各自直连推理进程」改为统一走一个 OpenAI 兼容端点。filesearch 原先
写死 127.0.0.1:8082 的 `/v1/embeddings`，必须改成走可配置端点。

新旧差异：

| 项 | 直连推理进程 | 经端点 |
|----|------------------|-----------|
| 地址 | 127.0.0.1:8082 | 127.0.0.1:18790 |
| 接口 | /v1/embeddings | /v1/embed/embeddings（端点侧 embed 接口） |
| 模型 | 由该进程加载的模型决定 | 必须传 "model": "qwen3-embedding-0.6b-q4_0" |
| 认证 | 无 | 当前设备已关闭鉴权 |
| 返回 | data[0].embedding | 相同（1024 维） |
| 启停 | 自己管理推理进程 | 端点的 load/unload 接口 |
| 模型端口 | 固定 8082 | 端点动态分配，调用方不关心 |

本文件守五条：
  1. 请求必须打到**配置的** path（默认 /v1/embed/embeddings），
     且带上 model / encoding_format —— 不再写死 /v1/embeddings
  2. 模型未加载（503 not_ready）时，必须调 load 接口后重试，且**不消耗业务重试次数**
  3. 探活（ping）与维度探测（probe_dim）走同一条配置化 path
  4. 仍可退回直连推理进程（path=/v1/embeddings、load_path 留空）—— 不能只支持端点模式
  5. 既有行为不回归：批量/超时二分/InputTooLarge 字符二分照旧
"""
from __future__ import annotations

import httpx
import pytest

from bibit_filesearch.config import DEFAULTS
from bibit_filesearch.model_client import EmbedClient

from fake_embed import DIM as FAKE_DIM  # noqa: F401  （保持与其它测试同源）

# ---------------------------------------------------------------- 伪端点


class _Resp:
    def __init__(self, code: int, payload=None, text: str = ""):
        self.status_code = code
        self._payload = payload if payload is not None else {}
        self.text = text or ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=None, response=None)


class _FakeEndpoint:
    """最小端点替身：记录 (method_path, payload)，支持「未加载 → 503 → load → 就绪」。"""

    def __init__(self, *, loaded: bool = True, dim: int = 1024, over_at: int | None = None,
                 too_large_chars: int | None = None):
        self.loaded = loaded
        self.dim = dim
        self.over_at = over_at              # 输入条数 >= 此值 → ReadTimeout（模拟算不完）
        self.too_large_chars = too_large_chars
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json=None, timeout=None):
        body = json or {}
        texts = body.get("input")
        n = len(texts) if isinstance(texts, list) else 1
        self.calls.append((url, body))

        if url.endswith("/models/load"):
            self.loaded = True
            return _Resp(200, {"model": body.get("model"), "status": "loaded"})

        if not self.loaded:
            # 端点：模型未加载时推理接口返 503（bibit 侧 health 拿到的 not_ready 同源）
            return _Resp(503, {"detail": "model not loaded"}, text="model not loaded")

        if self.too_large_chars is not None:
            for t in (texts if isinstance(texts, list) else [texts]):
                if len(str(t)) > self.too_large_chars:
                    return _Resp(500, text=(
                        f"input ({len(str(t))} tokens) is too large to process. "
                        f"increase the physical batch size (current batch size: 512)"))

        if self.over_at is not None and n >= self.over_at:
            raise httpx.ReadTimeout("simulated: batch too slow")

        return _Resp(200, {"data": [{"index": i, "embedding": [0.5] * self.dim}
                                    for i in range(n)]})

    def close(self):
        pass


def _gw_client(gw, **kw) -> EmbedClient:
    kw.setdefault("dim", gw.dim)
    kw.setdefault("model", "qwen3-embedding-0.6b-q4_0")
    kw.setdefault("path", "/v1/embed/embeddings")
    kw.setdefault("encoding_format", "float")
    kw.setdefault("load_path", "/v1/embed/models/load")
    c = EmbedClient("http://127.0.0.1:18790", **kw)
    c._client = gw
    return c


# ------------------------------------------------- 1. 契约：地址/路径/模型名


def test_defaults_point_at_gateway():
    """出厂默认必须指向端点（旧默认 8082 已退休）。"""
    emb = DEFAULTS["models"]["embed"]
    assert emb["url"].endswith(":18790"), f"默认地址应为 :18790，实际 {emb['url']}"
    assert emb["path"] == "/v1/embed/embeddings", (
        f"默认应走端点的本地模型限定接口，实际 {emb.get('path')}")
    assert emb["model"] == "qwen3-embedding-0.6b-q4_0", f"端点要求传 model=qwen3-embedding-0.6b-q4_0，实际 {emb['model']}"
    assert emb["dim"] == 1024
    assert emb.get("load_path") == "/v1/embed/models/load", "缺少模型加载接口配置"


def test_request_uses_configured_path_and_model():
    """请求必须打到配置的 path，并带 model + encoding_format。"""
    gw = _FakeEndpoint()
    c = _gw_client(gw)
    c.embed_batch(["hello"])

    path, body = gw.calls[-1]
    assert path == "/v1/embed/embeddings", f"路径写死了？实际 {path}"
    assert body["model"] == "qwen3-embedding-0.6b-q4_0", f"model 必须显式传，实际 {body.get('model')}"
    assert body["encoding_format"] == "float", (
        f"端点契约要求 encoding_format=float，实际 {body.get('encoding_format')}")
    assert body["input"] == ["hello"]


def test_ping_and_probe_dim_use_same_path():
    """探活与维度探测必须走同一条配置化 path（原先三处各自写死）。"""
    gw = _FakeEndpoint(dim=1024)
    c = _gw_client(gw)

    assert c.ping() is True
    assert gw.calls[-1][0] == "/v1/embed/embeddings"

    dim = c.probe_dim()
    assert dim == 1024
    assert gw.calls[-1][0] == "/v1/embed/embeddings"


def test_business_request_does_not_disable_httpx_timeout():
    """业务请求不得显式传 timeout=None —— httpx 里那是「禁用超时」，会让 60s 读超时静默失效。

    这是本轮新增 `_post_embeddings` 时踩到的真坑：用 `timeout=None` 当「不传」的默认值，
    结果把整条超时机制（含超时二分）关掉了，而假客户端无法感知（它不看 timeout 参数）。
    故用记录型客户端断言：普通业务请求**不带** timeout 关键字。
    """
    class _Recording:
        dim = 4  # _gw_client 会从这里取默认维度

        def __init__(self):
            self.kwargs: list[dict] = []

        def post(self, url, json=None, **kw):
            self.kwargs.append(kw)
            n = len((json or {}).get("input") or [])
            return _Resp(200, {"data": [{"index": i, "embedding": [0.5] * 4} for i in range(n)]})

        def close(self):
            pass

    rec = _Recording()
    c = _gw_client(rec)
    c.embed_batch(["a"])          # 业务路径（走 _embed_request → _post_embeddings）
    c.ping()                      # 探活路径（显式 probe_timeout，应该带）

    assert "timeout" not in rec.kwargs[0], (
        f"业务请求泄漏了 timeout={rec.kwargs[0].get('timeout')}——httpx 会据此禁用超时")
    assert rec.kwargs[1].get("timeout") == c.probe_timeout, (
        "探活必须显式带 probe_timeout（要远小于业务超时，快速失败）")


# ------------------------------------------------- 2. 模型未加载 → 自动 load


def test_unloaded_model_is_loaded_then_retried():
    """503（模型未加载）→ 调 load → 重试成功。"""
    gw = _FakeEndpoint(loaded=False)
    c = _gw_client(gw)

    out = c.embed_batch(["a", "b"])
    assert len(out) == 2, "load 后应当成功拿到向量"

    paths = [p for p, _ in gw.calls]
    assert "/v1/embed/models/load" in paths, f"未调用 load 接口：{paths}"
    # 首个推理请求必然在 load 之前（正是它拿到 503 才发现未加载）；
    # 关键不变量是 **load 之后还有一次推理请求**（即真的重试了）。
    load_i = paths.index("/v1/embed/models/load")
    assert "/v1/embed/embeddings" in paths[load_i + 1:], \
        f"load 之后必须重试推理：{paths}"
    # 503 那次 + load 后成功那次 = 2 次推理请求，没有多余重试
    assert paths.count("/v1/embed/embeddings") == 2, f"请求数异常：{paths}"


def test_auto_load_does_not_consume_business_retries():
    """load 不算业务重试：load 后一次成功，不得再触发 retries 次白重试。"""
    gw = _FakeEndpoint(loaded=False)
    c = _gw_client(gw, retries=2)
    c.embed_batch(["x"])
    assert gw.calls[0][0].endswith("/v1/embed/embeddings")
    assert len([p for p, _ in gw.calls if p.endswith("/embeddings")]) == 2


def test_load_only_attempted_once_per_call():
    """load 失败时不得无限重试 load（每次调用最多一次）。"""
    class _NeverLoads(_FakeEndpoint):
        def post(self, url, json=None, timeout=None):
            self.calls.append((url, json or {}))
            if url.endswith("/models/load"):
                return _Resp(500, text="load failed")
            return _Resp(503, text="model not loaded")

    gw = _NeverLoads(loaded=False)
    c = _gw_client(gw, retries=0)
    with pytest.raises(RuntimeError):
        c.embed_batch(["x"])
    assert len([p for p, _ in gw.calls if p.endswith("/models/load")]) == 1, \
        f"load 应只尝试一次：{gw.calls}"


# ------------------------------------------------- 3. 仍支持直连推理进程


def test_direct_endpoint_still_supported():
    """退回直连：path=/v1/embeddings、load_path 留空 → 不发 load、不带 encoding_format。"""
    gw = _FakeEndpoint()
    c = EmbedClient("http://127.0.0.1:8082", dim=1024, model="qwen3-embedding-0.6b",
                    path="/v1/embeddings", load_path="")
    c._client = gw

    c.embed_batch(["hi"])
    path, body = gw.calls[-1]
    assert path == "/v1/embeddings"
    assert "encoding_format" not in body, "直连模式不应附加端点专有字段"


def test_direct_mode_has_no_load_path_configured():
    """出厂默认已切网关，但 config 必须允许改回直连（键齐全）。"""
    emb = DEFAULTS["models"]["embed"]
    for key in ("url", "path", "model", "load_path", "encoding_format", "load_timeout"):
        assert key in emb, f"配置项 {key} 缺失，切换后端将需要改代码"


# ------------------------------- 4. 既有机制不回归（超时二分 / 超限字符二分）


def test_timeout_bisect_still_works_through_gateway():
    """经端点时超时二分照旧生效（32 → 16 + 16）。"""
    gw = _FakeEndpoint(over_at=16)
    c = _gw_client(gw, batch_size=32)
    out = c.embed_batch([f"c{i}" for i in range(32)])
    assert len(out) == 32
    sizes = [len(b.get("input") or []) for p, b in gw.calls if p.endswith("/embeddings")]
    assert 32 in sizes and 16 in sizes, f"二分未生效：{sizes}"


def test_input_too_large_still_char_splits_through_gateway():
    """经端点时「单条超物理批次」仍走字符二分（端点透传底层报错）。"""
    gw = _FakeEndpoint(too_large_chars=512)
    c = _gw_client(gw, batch_size=0)
    out = c.embed_batch(["z" * 900])
    assert len(out) == 1, "向量数必须等于输入数"
    assert len(out[0]) == gw.dim
