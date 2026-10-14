"""embed 超时预算与「失败时缩小工作单元」测试（2026-09-20）。

背景（169 实测）：/home/user/test/k1_ds.md（286KB → 599 块，受
max_chunks_per_file 截为 32 块 = 恰好 1 批 = 1 个 HTTP 请求）在机器有负载时
索引失败，reason = "embed 请求失败（重试 2 次）: timed out"。

实测数据：
  · 单批 32 块（markdown）空闲耗时 45.8s，而客户端 read timeout 是 60s → 余量 24%
  · 同一批压缩 JS 耗时 124~150s（--cache-ram 0）→ 必然超时
  · 失败时 3 次尝试共阻塞 176s ≈ 3 × 60s，服务端日志出现一串 `cancel task`
    （请求到了服务端、正在算，是客户端先超时断开）
  · 该文件最终 failed / attempt_count=1 / chunks=0 —— 0 个向量，前功尽弃

根因不是数值取小，而是机制用错了模型：现有实现按「服务偶尔抖动」设计
（退避 + 原样重发），但真实负载是「工作量有长尾」——同样的 32 条重发，
工作量不变，因此必然再次超时。而代码对另一类失败（_InputTooLarge）已经
做对了：**缩小工作单元**（队列二分 → 字符二分）。本文件守的就是把超时
也接进这条路径。

本文件守四条：
  1. timeout / connect_timeout 必须可配置（不得硬编码）
  2. read 与 connect 必须分离（本机回环连不上要秒级失败，read 要覆盖整批推理）
  3. 批超时 → 队列二分（32→16→8→…），**不得原样重发同样大小**
  4. 单条仍超时 → 明确失败（不无限深挖）；正常的 _InputTooLarge 字符二分不回归
"""
from __future__ import annotations

import os

import httpx
import pytest
import yaml

from bibit_filesearch.config import DEFAULTS
from bibit_filesearch.model_client import EmbedClient, _InputTooLarge

from fake_embed import DIM as FAKE_DIM
from fake_embed import FakeEmbed

# ---------------------------------------------------------------- 伪 HTTP 层


class _Resp:
    """最小响应替身：带 text / status_code / json / raise_for_status。"""

    def __init__(self, code: int, payload=None, text: str = ""):
        self.status_code = code
        self._payload = payload if payload is not None else {}
        self.text = text or str(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=None
            )


class _SlowHTTP:
    """伪 HTTP：输入条数 ≥ over_at 就抛 ReadTimeout（模拟「这批算不完」）。

    为什么用 ReadTimeout 而不是直接造 _Timeout：要让**真实的映射代码**参与
    （_embed_request 把 httpx 超时转成 _Timeout），否则测的是替身而不是产品。
    """

    def __init__(self, over_at: int):
        self.over_at = over_at
        self.batches: list[int] = []   # 每次请求的输入条数

    def post(self, url, json=None, timeout=None):
        texts = (json or {}).get("input") or []
        n = len(texts)
        self.batches.append(n)
        if n >= self.over_at:
            raise httpx.ReadTimeout("simulated: this batch takes too long")
        return _Resp(200, {"data": [{"index": i, "embedding": [0.5, 0.5, 0.5, 0.5]}
                                    for i in range(n)]})

    def close(self):
        pass


class _PhysBatchHTTP:
    """伪 HTTP：单条输入超字符上限就返 500（真实服务原文）。"""

    def __init__(self, limit_chars: int):
        self.limit = limit_chars
        self.batches: list[list[int]] = []

    def post(self, url, json=None, timeout=None):
        texts = (json or {}).get("input") or []
        self.batches.append([len(t) for t in texts])
        for t in texts:
            if len(t) > self.limit:
                return _Resp(
                    500, text=f"input ({len(t)} tokens) is too large to process. "
                              f"increase the physical batch size (current batch size: 512)"
                )
        return _Resp(200, {"data": [{"index": i, "embedding": [0.5] * 4}
                                    for i in range(len(texts))]})

    def close(self):
        pass


def _client(http, **kw) -> EmbedClient:
    kw.setdefault("batch_size", 32)
    kw.setdefault("timeout", 60.0)
    c = EmbedClient("http://127.0.0.1:1", dim=4, **kw)
    c._client = http
    return c


# ------------------------------------------------- 1. 超时必须可配置且分离


def test_embed_timeout_defaults_are_configurable():
    """timeout / connect_timeout 必须是配置项（禁止只藏在构造默认值里）。"""
    emb = DEFAULTS["models"]["embed"]
    assert "timeout" in emb, "models.embed.timeout 缺失——超时预算必须是配置项"
    assert "connect_timeout" in emb, "models.embed.connect_timeout 缺失——必须与 read 分离"
    assert emb["timeout"] == 60, "read 超时保持 60s（2026-09-20 定案）"
    assert emb["connect_timeout"] <= 10, "本机回环：连不上应快速失败，不该等 60s"


def test_read_and_connect_timeout_are_separate():
    """httpx 客户端必须拿到分开的 read / connect 超时。"""
    c = EmbedClient("http://127.0.0.1:1", dim=4, timeout=60.0, connect_timeout=3.0)
    t = c._client.timeout
    assert t.read == 60.0, f"read 应为 60s，实际 {t.read}"
    assert t.connect == 3.0, f"connect 应为 3s，实际 {t.connect}"
    c.close()


def test_fake_embed_accepts_create_app_kwargs():
    """create_app 按关键字构造 embed 客户端，替身签名必须跟上。

    2026-09-18 的教训：create_app 新增 batch_size / probe_timeout 透传后，
    FakeEmbed 签名没跟上 → 用 FakeEmbed 的集成测试在构造处直接 TypeError。
    """
    import inspect

    params = inspect.signature(FakeEmbed.__init__).parameters
    for name in ("dim", "model", "query_instruction", "timeout", "connect_timeout",
                 "batch_size", "probe_timeout"):
        assert name in params, f"FakeEmbed 缺形参 {name}（create_app 会按名传入）"


def test_create_app_passes_timeout_from_config(tmp_path, monkeypatch):
    """配置里的 timeout / connect_timeout 必须真的传到 EmbedClient。"""

    class _Recording(FakeEmbed):
        last_kwargs: dict = {}

        def __init__(self, *a, **kw):
            type(self).last_kwargs = dict(kw)
            super().__init__(*a, **kw)

    kb = tmp_path / "kb"
    kb.mkdir()
    cfg = {
        "storage": {"db_path": str(tmp_path / "t.db")},
        "server": {"port": 18098, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "models": {"embed": {"dim": FAKE_DIM, "timeout": 45, "connect_timeout": 2}},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    os.environ["BIBIT_CONFIG"] = str(cfg_path)

    from bibit_filesearch import model_client as mc
    from bibit_filesearch.main import create_app

    monkeypatch.setattr(mc, "EmbedClient", _Recording)
    create_app(str(cfg_path))

    kw = _Recording.last_kwargs
    assert float(kw.get("timeout")) == 45.0, f"timeout 未透传：{kw.get('timeout')}"
    assert float(kw.get("connect_timeout")) == 2.0, \
        f"connect_timeout 未透传：{kw.get('connect_timeout')}"


# ------------------------------------------------- 2. 超时 → 缩小工作单元


def test_read_timeout_becomes_timeout_error_not_retry_storm():
    """httpx.ReadTimeout 必须被识别为「工作量超预算」，且**不走退避重试**。

    原实现：3 次尝试各等满 60s（实测共 176s），服务端白算 3 遍。
    """
    http = _SlowHTTP(over_at=1)
    c = _client(http, timeout=60.0)
    with pytest.raises(RuntimeError, match="未算完|队列二分"):
        c.embed_batch(["x" * 500])
    assert http.batches == [1], (
        f"单条超时应只发 1 次、立即冒泡，不得退避重试同样的输入；实际 {http.batches}"
    )


def test_batch_timeout_splits_queue_instead_of_resending():
    """核心守卫：批超时后必须**队列二分**，不得原样重发同样大小。"""
    http = _SlowHTTP(over_at=16)          # 16 条及以上算不完，8 条能过
    c = _client(http, timeout=60.0)

    out = c.embed_batch([f"chunk-{i}" for i in range(32)])

    assert len(out) == 32, "向量数必须等于输入数（1:1，不破坏 indexer 的数量断言）"
    assert 32 in http.batches, "先按 batch_size 整批试一次"
    assert http.batches.count(32) == 1, (
        f"不得原样重发同样大小——那正是原 bug；实际 {http.batches}"
    )
    assert 16 in http.batches, f"超时后必须队列二分到 16；实际 {http.batches}"
    assert 8 in http.batches, f"16 仍超时必须继续二分到 8；实际 {http.batches}"
    assert max(http.batches) == 32


def test_normal_batch_makes_exactly_one_request():
    """正常内容零额外请求（本次改动不得引入开销）。"""
    http = _SlowHTTP(over_at=10_000)
    c = _client(http, timeout=60.0)
    out = c.embed_batch([f"c{i}" for i in range(32)])
    assert len(out) == 32
    assert http.batches == [32], f"应只发一次整批，实际 {http.batches}"


def test_single_item_timeout_fails_fast_without_infinite_split():
    """单条也超时 = 服务真过载/卡死，拆分无济于事 → 明确失败，不无限递归。"""
    http = _SlowHTTP(over_at=1)
    c = _client(http, timeout=60.0)
    with pytest.raises(RuntimeError, match="未算完"):
        c.embed_batch(["x" * 500])
    assert http.batches.count(1) == 1, "单条不得反复重试/拆解"


def test_timeout_split_depth_is_bounded():
    """深度有界：32 条最多拆到第 3 层（4 条），请求数不得指数爆炸。"""
    http = _SlowHTTP(over_at=4)           # 只有 ≤3 条能过 → 到深度上限仍失败
    c = _client(http, timeout=60.0)

    with pytest.raises(RuntimeError, match="队列二分"):
        c.embed_batch([f"c{i}" for i in range(32)])

    assert len(http.batches) <= 8, f"请求数应有界，实际 {len(http.batches)}：{http.batches}"
    assert min(http.batches) >= 4, (
        f"深度上限为 3 层 → 最小批是 4 条，不该继续下探；实际 {http.batches}"
    )


def test_timeout_message_names_the_service_not_the_file():
    """报错要指向服务端负载，而不是让用户以为文件有问题。"""
    http = _SlowHTTP(over_at=4)
    c = _client(http, timeout=60.0)
    with pytest.raises(RuntimeError) as ei:
        c.embed_batch([f"c{i}" for i in range(32)])
    msg = str(ei.value)
    assert "过载" in msg or "卡死" in msg, (
        f"错误信息应引导去看 embed 服务负载；实际：{msg}"
    )


# --------------------------------- 3. 既有 _InputTooLarge 路径不得回归


def test_input_too_large_char_split_still_returns_one_vector():
    """单条超限 → 字符二分 + 向量平均，仍返回 1 条向量（既有行为）。"""
    http = _PhysBatchHTTP(limit_chars=512)
    c = _client(http, timeout=60.0)
    out = c.embed_batch(["z" * 900])
    assert len(out) == 1, "向量数必须等于输入数"
    assert len(out[0]) == 4
    flats = [s for b in http.batches for s in b]
    assert flats[0] == 900, "首个请求应是原始长度"
    assert min(flats) <= 512, f"应切到限内，实际最小 {min(flats)}"


def test_input_too_large_queue_split_still_narrows_to_offender():
    """整批被拒时先按队列二分定位超限那条（既有行为，不受深度上限影响）。"""
    http = _PhysBatchHTTP(limit_chars=512)
    c = _client(http, timeout=60.0)
    texts = ["a" * 100, "b" * 100, "c" * 900, "d" * 100]
    out = c.embed_batch(texts)
    assert len(out) == 4, "4 条输入必须得 4 条向量"
    assert all(len(v) == 4 for v in out)


def test_input_too_large_is_not_retried():
    """超限不得走重试（重试结果相同，纯浪费）。"""
    http = _PhysBatchHTTP(limit_chars=512)
    c = _client(http, timeout=60.0, batch_size=0)   # 不分批，直接进 _embed_resilient
    c.embed_batch(["z" * 900])
    assert http.batches[0] == [900], "首个请求应是原始长度"
    # 900（被拒）→ 450 + 450（成功）：共 3 次，且不重复同样尺寸
    assert len(http.batches) == 3, f"应为 3 次，实际 {http.batches}"
