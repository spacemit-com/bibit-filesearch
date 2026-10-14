"""embed 响应必须与请求**逐条对应**（2026-10-08，审查 R10）。

原实现只按 `index` 排序后直接取值，不校验数量与 index 的完整性：

    请求 2 条、服务只回 1 条 → `embed_batch` 返回 1 条，不报错（实测）

下游 `indexer` 有数量断言兜底，所以不会把错的向量写进块；但结论仍要说清——
这不是「已证实的静默错写」，而是**协议边界缺一道防线**：

  · 其它调用点（如 embed_one 之外的使用）拿到错位结果；
  · 数量够但 index 重复（两个 index=0）时，数量断言看不出来，排序后位置是错的；
  · 失败暴露在离协议错误很远的地方，排障要往回翻。

本文件守住：数量不符、index 重复、index 越界都要在客户端这一层报错；
合法但乱序的响应仍要恢复成输入顺序。
"""
from __future__ import annotations

import httpx
import pytest

from bibit_filesearch.model_client import EmbedClient, EmbedError


def _client(dim: int = 2, retries: int = 0) -> EmbedClient:
    """构造一个不连真服务的 EmbedClient。"""
    c = EmbedClient("http://test", dim=dim, retries=retries)
    c._client.close()
    return c


def _patch_response(c: EmbedClient, payload) -> None:
    c._client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
        base_url="http://test",
    )


def _vec(n: int, dim: int = 2) -> list[float]:
    v = [0.0] * dim
    v[n % dim] = 1.0
    return v


# ── 数量不符必须报错 ──────────────────────────────────────────────────────────

def test_fewer_vectors_than_requested_is_rejected():
    """请求 2 条、只回 1 条 → 必须报错（改前静默返回 1 条）。"""
    c = _client()
    _patch_response(c, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    with pytest.raises(EmbedError) as ei:
        c.embed_batch(["first", "second"])
    assert "请求 2" in str(ei.value) and "返回 1" in str(ei.value)
    c.close()


def test_more_vectors_than_requested_is_rejected():
    """回多了同样不对——向量会错位映射到别的输入上。"""
    c = _client()
    _patch_response(c, {"data": [
        {"index": 0, "embedding": [1.0, 0.0]},
        {"index": 1, "embedding": [0.0, 1.0]},
        {"index": 2, "embedding": [1.0, 0.0]},
    ]})
    with pytest.raises(EmbedError):
        c.embed_batch(["only-one"])
    c.close()


# ── index 完整性 ──────────────────────────────────────────────────────────────

def test_duplicate_index_is_rejected():
    """数量够但 index 重复：数量断言看不出，必须在客户端拦。"""
    c = _client()
    _patch_response(c, {"data": [
        {"index": 0, "embedding": [1.0, 0.0]},
        {"index": 0, "embedding": [0.0, 1.0]},
    ]})
    with pytest.raises(EmbedError) as ei:
        c.embed_batch(["a", "b"])
    assert "index" in str(ei.value)
    c.close()


def test_out_of_range_index_is_rejected():
    """index 超出 0..n-1（如从 1 开始编号）也不能放行。"""
    c = _client()
    _patch_response(c, {"data": [
        {"index": 1, "embedding": [1.0, 0.0]},
        {"index": 2, "embedding": [0.0, 1.0]},
    ]})
    with pytest.raises(EmbedError):
        c.embed_batch(["a", "b"])
    c.close()


# ── 合法响应仍然照常工作 ──────────────────────────────────────────────────────

def test_in_order_response_passes():
    """对照组：顺序正确 → 原样返回。"""
    c = _client()
    _patch_response(c, {"data": [
        {"index": 0, "embedding": [1.0, 0.0]},
        {"index": 1, "embedding": [0.0, 1.0]},
    ]})
    out = c.embed_batch(["a", "b"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]
    c.close()


def test_shuffled_response_is_restored_to_input_order():
    """乱序返回（真实服务会这样）→ 按 index 恢复成输入顺序，不报错。"""
    c = _client()
    _patch_response(c, {"data": [
        {"index": 2, "embedding": _vec(2)},
        {"index": 0, "embedding": _vec(0)},
        {"index": 1, "embedding": _vec(1)},
    ]})
    out = c.embed_batch(["a", "b", "c"])
    assert out == [_vec(0), _vec(1), _vec(2)], f"乱序响应未按 index 还原：{out}"
    c.close()


def test_single_input_round_trip():
    """单条请求（embed_one / probe_dim 走的路径）不受影响。"""
    c = _client()
    _patch_response(c, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    assert c.embed_one("x") == [1.0, 0.0]
    c.close()


# ── 与既有维度防线共存 ────────────────────────────────────────────────────────

def test_dimension_mismatch_still_reported():
    """维度不符仍报维度错（别被新加的校验抢在前面报成数量错）。"""
    c = _client(dim=4)          # 配置 4 维，服务回 2 维
    _patch_response(c, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    with pytest.raises(EmbedError) as ei:
        c.embed_batch(["x"])
    assert "维度" in str(ei.value), f"应报维度不符，实际：{ei.value}"
    c.close()
