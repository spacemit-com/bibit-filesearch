"""模型客户端：Embed（OpenAI 兼容接口）；rerank / LLM 已移除。"""
from __future__ import annotations

import logging
import time
from typing import Any

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEPS = _PKG_ROOT / "deps"
if str(_DEPS) not in sys.path:
    sys.path.insert(0, str(_DEPS))

import httpx

logger = logging.getLogger("bibit.model")


class _InputTooLarge(Exception):
    """服务端拒收：单条输入超过它的物理批次上限。不重试（重试结果相同）。"""


class _Timeout(Exception):
    """一批输入在超时预算内没算完（工作量超预算）。

    2026-09-20：与 _InputTooLarge 并列的第二种「原样重发没有意义」的失败。
    区别在于失败信号的来源：_InputTooLarge 是服务端**明确拒收**（HTTP 500），
    这个是客户端**等不到响应**（httpx.ReadTimeout）。共同点是**同样的输入
    → 同样的工作量 → 同样的结果**，所以两者都必须走「缩小工作单元」，
    而不是退避后原样重发。

    为什么不能靠调大 timeout 解决：该接口是**非流式**
    （算完整批才返回，中途没有进度信号），于是这个超时实际是「这批 32 条必须
    算完的硬预算」，而单批耗时随内容与负载浮动——169 实测 markdown 45.8s、
    压缩 JS 124~150s（3.3×）。固定常数不可能覆盖这个区间。
    """


class EmbedError(RuntimeError):
    """向量化失败（**服务侧**原因：过载、超时、拒收、维度不符、重试耗尽）。

    2026-09-21 新增，专供上游分类。此前这些位置抛的都是裸 `RuntimeError`，而
    scanner 的 worker 把「除 PermissionError 之外的一切异常」都归成 `"parse"`，
    于是 embed 过载超时在界面上显示成「错误解析」——名不符实，排查时会把人引向
    解析器/文件格式，而真正的原因在服务负载。

    继承 `RuntimeError` 是刻意的：既有调用方与测试都用
    `except RuntimeError` / `pytest.raises(RuntimeError)` 兜这些失败，改动不破坏
    它们的语义，只是让**能区分的地方**区分开。
    """


def _is_input_too_large(body: str) -> bool:
    """识别服务端的物理批次超限报错。

    原文形如：
      `input (616 tokens) is too large to process. increase the physical
       batch size (current batch size: 512)`
    """
    b = (body or "").lower()
    return "too large to process" in b or "increase the physical batch size" in b



class EmbedClient:
    """Embed 客户端（OpenAI 兼容接口）。

    2026-09-03：模型语义配置化——model / dim / query_instruction 全部来自配置，
    切换 embed 模型（如 bge-small-zh-v1.5 512 维 → Qwen3-Embedding-0.6B 1024 维）
    只需改 config 并全库重索引，不改代码。
    """

    # 二分降级的下界（字符）：短于此仍被拒就停止递归。64 字符即使按最坏的
    # 1 字符 = 1 token 也远小于任何合理的物理批次（实测服务端 512）。
    _MIN_SPLIT_CHARS = 64

    # 超时降级的最大层数（仅用于「超时」这条路径）。3 层 = 32→16→8→4，
    # 覆盖 8× 的耗时膨胀；再深说明服务端不是「慢」而是「坏」，继续拆没有意义，
    # 只会在每层白等一个 timeout。这个上限的存在意义是**封顶最坏阻塞时间**：
    # 超时路径每层都要等满 timeout，而 _InputTooLarge 路径的每层都是秒级 500
    # 快速返回（且每层都确定性地缩小到那条超限输入），所以那条路径不设此限。
    _MAX_TIMEOUT_SPLIT_DEPTH = 3

    def __init__(self, url: str, dim: int = 512, model: str = "bge", query_instruction: str = "",
                 timeout: float = 60.0, connect_timeout: float = 5.0,
                 retries: int = 2, batch_size: int = 32,
                 probe_timeout: float = 5.0,
                 path: str = "/v1/embeddings", encoding_format: str = "",
                 load_path: str = "", load_timeout: float = 150.0,
                 health_grace: float = 0.0):
        self.url = url.rstrip("/")
        self.dim = dim
        self.model = model  # 请求体 "model" 字段透传（服务端不校验模型名）
        self.query_instruction = query_instruction  # 查询指令前缀；空 = 裸 query（bge 默认行为）
        self.retries = retries  # 2026-08-26：embed 服务抖动重试（原直接抛异常冒泡到扫描线程）
        # 2026-09-18：单请求最大输入数（分批上界，见 embed_batch）；0 = 不分批（不推荐）
        self.batch_size = batch_size
        # 2026-09-18：真实探活超时（见 ping）；比业务请求的 timeout 短得多——探活要快速失败
        self.probe_timeout = probe_timeout
        # 2026-09-20：read 与 connect 分离。原先单一 60s 同时管两件事，两边都取不到正确值：
        #   · read 太短——非流式 embeddings 要等整批算完，实测单批 32 块 45.8s（markdown）
        #     ~150s（压缩 JS，--cache-ram 0），60s 覆盖不了长尾；
        #   · connect 太长——本机回环连不上时应秒级失败，不该白等 60s。
        # 超时不再是「唯一答案」：超时后按 _MAX_TIMEOUT_SPLIT_DEPTH 缩小工作单元，
        # 所以 60s 只需覆盖「一个能过的批」，不需要覆盖最坏内容。
        self.timeout = timeout
        # 2026-09-20：后端接入方式配置化——推理后端由「各自直连推理进程」改为统一走
        # 一个 OpenAI 兼容端点，差异只在**路径与模型名**（响应形状相同）：
        #   直连推理进程: /v1/embeddings, 模型由该进程加载决定
        #   统一端点    : /v1/embed/embeddings, 必须显式传 "model"
        # 故不做成分支代码，只把三个差异点变成配置：
        #   path            —— 推理路径（默认 /v1/embed/embeddings）
        #   encoding_format —— 附加字段（端点契约要求 "float"；直连留空则不发送，
        #                      避免给不认识该字段的服务端塞未知键）
        #   load_path       —— 端点的模型加载接口（直连留空 = 不适用）。模型不再由
        #                      filesearch 自己拉起进程：端点动态分配端口，调用方
        #                      只管请求 load。
        self.path = path or "/v1/embeddings"
        self.encoding_format = encoding_format
        self.load_path = load_path
        # 模型加载可能真的要百来秒（读 GGUF + 建 KV），与业务超时分开口径：
        # 用 timeout 去等 load 会必然超时，用 load_timeout 去等推理又会把最坏阻塞拉长。
        self.load_timeout = load_timeout
        self._client = httpx.Client(
            base_url=self.url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )
        self._cache: dict[str, list[float]] = {}  # 同 query 缓存兜底
        # 2026-09-21：**被动**健康信号（见 is_healthy）。ping() 是「向被测对象要一次
        # 空闲」——扫描期间 embed 被批次占满，探针排在队尾、probe_timeout(5s) 必然
        # 超时，把好端端的后端误判成 unreachable，界面于是在「扫描中」与
        # 「索引服务不可用 / 向量化服务未启动」之间抖动（118 文件目录实测 100% 复现）。
        # 被动信号问的是另一件事：**最近有没有真的成功产出过向量**。扫描中恒定健康，
        # 且零额外请求。grace 取 2×timeout（≥120s）：覆盖「一个批次的正常耗时」+
        # 一次超时后二分重试的间隙，避免把「正在算」误判成坏。
        self.health_grace = float(health_grace or 0) or max(2.0 * float(timeout), 120.0)
        self._last_ok = 0.0  # monotonic；0 = 从未成功

    def _post_embeddings(self, texts: list[str], timeout: float | None = None):
        """推理请求（带 encoding_format 等后端专有字段），统一走 self.path。

        注意 timeout 的处理：**不能**把 None 透传给 httpx —— 在 httpx 里
        `timeout=None` 的语义是「禁用所有阶段超时」（等价 Timeout(None)），
        而不是「用客户端默认」。若透传，业务请求的 60s 读超时会静默失效，
        超时二分机制随之失守（请求会一直挂着而不是超时重试）。
        故只在显式传了 timeout 时才带该关键字，否则沿用构造时的客户端默认。
        """
        kw: dict = {} if timeout is None else {"timeout": timeout}
        return self._client.post(self.path, json=self._payload(texts), **kw)

    def _payload(self, texts: list[str]) -> dict:
        """构造请求体。encoding_format 仅在配置了才发送（直连模式留空）。"""
        body: dict = {"input": texts, "model": self.model}
        if self.encoding_format:
            body["encoding_format"] = self.encoding_format
        return body

    def _load_model(self) -> bool:
        """请端点加载模型（重启后 / 首次使用时模型处于未加载态）。

        2026-09-20：模型进程不再由 filesearch 拉起——端点统一管理，端口动态分配。
        未加载时推理接口返 503，此时先 POST load 再重试。
        返回是否已就绪；load_path 为空（直连模式）直接返回 False。
        """
        if not self.load_path:
            return False
        try:
            resp = self._client.post(self.load_path, json={"model": self.model},
                                     timeout=self.load_timeout)
            ok = resp.status_code < 400
            logger.info("embed 模型加载请求已送达（%s）：HTTP %s", self.load_path, resp.status_code)
            return ok
        except Exception as e:  # noqa: BLE001 —— 加载失败交给上层按普通失败处理
            logger.warning("embed 模型加载失败：%s", e)
            return False


    def embed_one(self, text: str) -> list[float]:
        """单条文本向量化（带缓存，LRU 上限防冻结——2026-08-26 修复）。

        查询文本专用：配置了 query_instruction（instruction-based 模型如 Qwen3-Embedding）
        时按官方模板拼 "Instruct: <任务描述>\\nQuery: <query>" 后再向量化；缓存 key 用
        原始文本（同 query 恒等命中）。bge 等 instruction 留空 → 输出与旧版完全一致。
        """
        if text in self._cache:
            return self._cache[text]
        payload = text
        if self.query_instruction:
            payload = f"Instruct: {self.query_instruction}\nQuery: {text}"
        vec = self.embed_batch([payload])[0]
        if len(self._cache) >= 10000:
            # 简单 LRU：满了移除最旧插入的（dict 保插入序）
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec
        return vec

    def probe_dim(self, timeout: float = 3.0) -> int | None:
        """向服务探一次真实向量维度；探测不到返回 None。

        2026-09-15 新增：维度**以服务为准**，配置 `models.embed.dim` 只是提示。
        修的是「配置里 dim 是老版出厂化石（512）而服务早已换成 1024 维」这一静默故障：
        每次 embed_batch 都因维度防线抛错 → 检索全返 1004，而 /health/deep 探的是
        :8082/health、不经过维度校验，所以看不出异常。
        本方法刻意绕过 embed_batch 的维度校验与重试（它存在的目的就是发现不符），
        失败一律返回 None——探不到不等于维度错，调用方应退回配置值。
        """
        # 2026-09-20：改走 self.path（原先此处写死
        # /v1/embeddings → 接入端点后探测永远失败，维度会静默退回配置值）
        try:
            resp = self._post_embeddings(["dim-probe"], timeout=timeout)
            resp.raise_for_status()
            data = resp.json().get("data") or []
            if not data:
                return None
            return len(data[0].get("embedding") or []) or None
        except Exception as e:  # noqa: BLE001 —— 探测失败不得影响启动
            logger.warning("embed 维度探测失败（%s）：%s", self.url, e)
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（按 batch_size 拆分为多次请求，逐批带重试与维度防线）。

        2026-09-18：**必须分批**。此前把整个文件的全部块塞进一次请求——实测一个
        24.8MB 的 minified JS 会切成 **83,782** 块（生产日志实测；早期「约 4.3 万」
        是估算，错了一倍），即「一个 HTTP 请求携带 83,782 条输入」。服务端按
        输入 1:1 建任务并为其分配缓冲，单请求就把服务端内存顶起来。

        分批买到的是**两件具体的事**，不是「内存增幅可控」：
          · 单请求体积/输入数有上界（不再出现一次 8 万条的请求）；
          · 失败重试只重发一小批，不必重发整份。
        它**不降低**服务端最终占用的总内存——实测同一批 16 条「没见过的新内容」：
        拆成 16 个单条请求 +576MB，合成 1 个请求 +191MB（拆小反而更费）。服务端如何
        为每条新内容计账不在本组件控制内。
        """
        if not texts:
            return []
        size = int(self.batch_size or 0)
        if size <= 0 or size >= len(texts):
            return self._embed_resilient(texts)
        out: list[list[float]] = []
        for i in range(0, len(texts), size):
            out.extend(self._embed_resilient(texts[i : i + size]))
        return out

    def _embed_resilient(self, texts: list[str], *, _timeout_depth: int = 0) -> list[list[float]]:
        """发送一批；服务端拒收**或客户端超时**时，缩小工作单元再试。

        为什么需要：分批只限制了**每请求的输入条数**，管不住**单条的长度**。
        服务端的物理批次（实测 512）按单条输入计，而 token
        密度因内容而异——普通文本 600 字符约 150~200 token，但压缩过的 JS/CSS
        能到 1 字符 ≈ 1 token（实测 600 字符 = 616 token，直接被拒收：
        `input (616 tokens) is too large to process`）。按字符数预估 token 数
        因此不可靠，只能在被拒收后按实际结果降级。

        降级路径：整批被拒 → 队列二分（找出超限的那条）；降到单条仍被拒 →
        把该条按字符二分、各自向量化后**取平均并重新归一化**，仍以 1 条向量
        返回。这样向量数与输入数保持 1:1，不破坏 indexer 的数量断言，且比
        直接丢弃后半段保留更多语义。

        代价：只有超限的那条会产生额外请求，正常内容一次通过，无额外开销。

        2026-09-20 扩展（超时）：原实现对超时走「退避 + **原样重发同样的输入**」，
        与上面这段自相矛盾——两类失败的共同点恰恰是「重发同一份工作量不会成功」。
        实测后果（169）：一批 32 块超时后重试 2 次，3 次尝试共阻塞 176s（≈3×60s），
        服务端日志出现一串 `cancel task`（它在算，是客户端先走的）→ 白算 3 遍，
        文件最终 0 个向量。改为：超时也走队列二分（32→16→8→4），把工作单元降到
        「能在预算内算完」的规模。normal 内容（markdown 单批 45.8s < 60s）零额外开销。
        """
        if not texts:
            return []
        reason = ""
        timed_out = False
        try:
            return self._embed_request(texts)
        except _InputTooLarge as exc:
            reason = f"服务端拒收（{exc}）"
        except _Timeout as exc:
            timed_out = True
            reason = f"在 {self.timeout}s 内未算完"

        if len(texts) > 1:
            if timed_out and _timeout_depth >= self._MAX_TIMEOUT_SPLIT_DEPTH:
                # 已经拆到足够小仍算不完：不是「批太大」而是「服务不行」。
                # 继续拆只会每层再白等一个 timeout，所以在此止损并说清方向。
                raise EmbedError(
                    f"已按队列二分到第 {_timeout_depth} 层仍失败（{reason}）："
                    f"{len(texts)} 条输入。embed 服务可能过载或卡死，"
                    f"请检查服务端负载与 -c/-t 配置"
                )
            mid = len(texts) // 2
            logger.warning(
                "批请求失败（%s），按队列二分：%d → %d + %d",
                reason, len(texts), mid, len(texts) - mid,
            )
            return (self._embed_resilient(texts[:mid], _timeout_depth=_timeout_depth + 1)
                    + self._embed_resilient(texts[mid:], _timeout_depth=_timeout_depth + 1))

        # 降到单条
        text = texts[0]
        if timed_out:
            # 单条 ≤600 字符（chunker 上界）才几秒；这都超时说明服务过载/卡死。
            # 再按字符切碎去猜毫无意义——问题的量级根本不在「输入多大」。
            raise EmbedError(
                f"单条输入 {len(text)} 字符在 {self.timeout}s 内未算完——"
                f"embed 服务可能过载或卡死，请检查服务端负载"
            )
        if len(text) <= self._MIN_SPLIT_CHARS:
            # 已经切到极短仍被拒：不再递归（避免无意义深挖），交给上层记失败
            raise EmbedError(
                f"单条输入 {len(text)} 字符仍被 embed 服务拒收（物理批次过小？），请检查服务端 -b 配置"
            )
        mid = len(text) // 2
        left = self._embed_resilient([text[:mid]])
        right = self._embed_resilient([text[mid:]])
        logger.warning(
            "单条输入被服务端拒收，已按字符二分（%d → %d + %d）并合并向量",
            len(text), mid, len(text) - mid,
        )
        return [self._merge_vec(left[0], right[0])]

    @staticmethod
    def _merge_vec(a: list[float], b: list[float]) -> list[float]:
        """两段向量的均值 + L2 归一化（服务端 --embd-normalize 2 保持同口径）。"""
        import math

        if not a or not b or len(a) != len(b):
            return a
        m = [(x + y) / 2.0 for x, y in zip(a, b)]
        n = math.sqrt(sum(v * v for v in m))
        return [v / n for v in m] if n else m

    def _embed_request(self, texts: list[str]) -> list[list[float]]:
        """单次 embed 请求（分批的基本单元）。失败重试 retries 次。

        2026-09-20（端点接入）：模型**不再由本进程管理**。端点动态分配模型端口，
        重启后模型处于未加载态、推理接口返 503。此处首次遇到 503 时先请求端点
        加载（self._load_model），再重发同一次尝试——**load 不算业务重试**，
        否则「等模型加载」会把 retries 次数白耗掉。
        """
        if not texts:
            return []
        last_exc: Exception | None = None
        attempt = 0
        load_attempted = False
        while True:
            try:
                resp = self._post_embeddings(texts)
                # 模型未加载（端点返 503）：请端点加载后重发，不消耗重试次数。
                # 只尝试一次——load 失败说明这是真故障，再 load 也是同样结果。
                if resp.status_code == 503 and self.load_path and not load_attempted:
                    load_attempted = True
                    if self._load_model():
                        continue
                # 物理批次超限：重试无用，直接冒泡给 _embed_resilient 走二分降级
                if resp.status_code >= 400 and _is_input_too_large(resp.text):
                    raise _InputTooLarge(resp.text[:200])
                resp.raise_for_status()
                data = resp.json()["data"]
                # 按 index 排序（服务端可能乱序）
                data.sort(key=lambda x: x["index"])
                # ★ 2026-10-08（审查 R10）：协议完整性防线——**数量与 index 都要对**。
                #   此前只按 index 排序后直接取值，数量少一条也照样返回（实测：请求
                #   2 条、服务只回 1 条，embed_batch 返回 1 条不报错）。下游
                #   `indexer` 有数量断言兜底，所以不会静默写错块；但别的调用点拿到的
                #   就是错位结果，且失败暴露得太晚（离协议错误本身很远）。
                #   index 重复同样要拦：数量够但两个都标 index=0 时，排序后第 2 条的
                #   位置是错的，而数量断言看不出来。
                indices = [d.get("index") for d in data]
                if len(data) != len(texts) or sorted(indices) != list(range(len(texts))):
                    raise EmbedError(
                        f"embed 服务返回与请求不匹配：请求 {len(texts)} 条，"
                        f"返回 {len(data)} 条，index={indices[:20]}"
                        f"{'…' if len(indices) > 20 else ''}（应为 0..{len(texts) - 1} 各一次）"
                    )
                vecs = [d["embedding"] for d in data]
                # 维度防线（2026-09-03）：服务返回维度 ≠ 配置 dim 立即报错——换模型后未同步
                # config.models.embed.dim 的典型症状；若放行会把错误维度向量写进索引/矩阵，
                # 造成静默错乱（旧 512 代码则直接 numpy 崩）。
                if self.dim and any(len(v) != self.dim for v in vecs):
                    raise EmbedError(
                        f"embed 服务返回维度与配置不符：models.embed.dim={self.dim}，实际 {len(vecs[0])}。"
                        f"若已切换 embed 模型，请同步更新 config 中 dim 并清空索引后全库重索引"
                    )
                # 2026-09-21：这里才是「真活」的权威证据——服务真的算完并返回了合格向量。
                # 供 is_healthy 的被动判据使用（探针在忙时会假超时，见 is_healthy）。
                self._mark_ok()
                return vecs
            except _InputTooLarge:
                raise
            except httpx.ReadTimeout as e:
                # 2026-09-20：**等一下等不到**——服务在算，只是这批在预算内算不完。
                # 不消耗重试次数、不退避、不原样重发：直接冒泡给 _embed_resilient
                # 走「缩小工作单元」。原实现让超时落进下面的通用 except，于是退避
                # 0.3/0.6s 后重发同样大小 —— 同样的工作量、同样的耗时，必然再超时
                # （169 实测：3 次尝试全超时，共阻塞 176s，服务端白算 3 遍）。
                # 注意只认 ReadTimeout：ConnectTimeout 属于「服务不在听」，
                # 缩小批大小无济于事（那是看门狗的职责），故留给通用重试路径。
                raise _Timeout(str(e) or "read timeout") from e
            except Exception as e:
                last_exc = e
                if attempt < self.retries:
                    attempt += 1
                    time.sleep(0.3 * attempt)  # 退避：0.3s / 0.6s
                    continue
                break
        raise EmbedError(f"embed 请求失败（重试 {self.retries} 次）: {last_exc}") from last_exc

    def _mark_ok(self) -> None:
        """记一次「真的成功产出了向量」的时间戳（被动健康信号，见 is_healthy）。

        只在**真实推理成功**时调用（_embed_request 拿到合格向量之后）。刻意**不**在
        ping()/probe_dim() 成功时记：那两处本身是探针，若让探针刷新被动信号，
        「每隔 grace 秒探一次」就会自我延续成「永远不用再探」，空闲期真掉线会被
        掩盖 grace 秒。让被动信号只反映**业务负载**的健康，语义单一。
        """
        self._last_ok = time.monotonic()

    def is_healthy(self, grace: float | None = None) -> bool:
        """**被动**健康：最近 grace 秒内真的成功算出过向量 → 健康（不发任何请求）。

        2026-09-21 新增。修的是「探针与被测对象抢同一个串行服务」导致的**假阴性**：

        探活原先是唯一的健康判据，而它是发一个真实 embeddings 请求（ping）。但
        该服务串行处理请求——扫描期间批次一个接一个（单批 32 块实测
        25~46s），探针只能排在队尾，等待时间 = 当前批次剩余时间，远超
        probe_timeout=5s → 必然超时 → 判 unreachable。于是 embed 明明在满负荷
        正常工作，界面却报「索引服务不可用」，并在批次间隙又跳回「扫描中」——
        5s 轮询撞上 25~46s 的批次，抖动是稳定复现的，不是偶发。
        （实测：118 文件 / 40MB 目录，抖动持续整个扫描过程。）

        判据换了问法：不问「现在有空吗」，问「最近有没有成功过」。扫描中恒定健康，
        空闲时才回落真探针。代价是**检测延迟**：后端在 grace 秒内刚成功过、随后
        立刻死掉的话，最多晚 grace 秒才报 unreachable；grace 从 timeout 推导
        （2×timeout，≥120s），远小于「一次全盘扫描」的量级。
        """
        g = self.health_grace if grace is None else float(grace)
        return bool(self._last_ok) and (time.monotonic() - self._last_ok) <= g

    def ping(self) -> bool:
        """真实探活：发一个最小 embeddings 请求，验证服务**真能干活**（非仅端口活着）。

        2026-09-18：原实现只 `GET /health` 看 200。实测该服务存在「能探活但
        干不了活」的半死态——服务还在加载模型或资源紧张时，`/v1/embeddings` 返 503，
        而 `/health` 仍回 200（实测：6 并发把请求排队 24s 期间 health 全 200）。
        看门狗因此看不见故障；而一旦服务彻底崩溃被拉起，它又立刻触发全量重扫，
        与「OOM → 重启 → 再灌一轮」形成正反馈。改为真活探测：
        连接失败 / 超时 / 非 200 / 空向量 一律判不健康。

        探针极轻（1 条 4 字符输入），节拍由调用方控制（看门狗默认 5s）。
        刻意走 `_post_embeddings` 而不复用 embed_batch：探测失败不得抛异常，
        也**不参与**维度防线（换模型时探针本身就应发现不符，见 probe_dim）。
        2026-09-20：改走 self.path（原先写死 /v1/embeddings，接入端点后探活恒失败
        → 看门狗会把「好的后端」判成 down，触发无意义的重扫）。
        """
        try:
            r = self._post_embeddings(["ping"], timeout=self.probe_timeout)
            if r.status_code != 200:
                return False
            data = r.json().get("data") or []
            return bool(data) and bool(data[0].get("embedding"))
        except Exception:  # noqa: BLE001 —— 探活不得抛出
            return False

    def close(self):
        self._client.close()


def normalize_scores(scores: list[float]) -> list[float]:
    """原始得分 → 0~1 归一化（softmax，负值安全）。

    rerank 返回 logit（如 4.68），需归一化后才可作为 confidence。
    多文档场景用 softmax 保证和为 1；单文档用 sigmoid。
    """
    import math

    if not scores:
        return []
    if len(scores) == 1:
        return [1.0 / (1.0 + math.exp(-scores[0]))]
    # softmax（数值稳定）
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    s = sum(exps)
    return [e / s for e in exps]


if __name__ == "__main__":
    # 冒烟自检
    logging.basicConfig(level=logging.INFO)
    emb = EmbedClient("http://127.0.0.1:8082")
    t0 = time.perf_counter()
    v = emb.embed_one("项目周报")
    print(f"embed 单条: {(time.perf_counter()-t0)*1000:.1f}ms, dim={len(v)}")
