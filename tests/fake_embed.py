"""确定性伪 embed 客户端（测试替身，替代真实 embed 服务）。

背景：T 系列集成测试（t4/t5/t7/t8/bm25_retriever）原本连真实 embed 服务
（http://127.0.0.1:8082）。2026-09-04 embed 切 Qwen3（1024 维、单块 ~0.4s）后：
- 旧断言硬编码 512 维失效（维度来自服务，测试不该假设具体模型）
- Qwen3 慢 → 大量 embed 请求超时 → 集成测试 24 failed + 8 errors

FakeEmbed 语义 = 词袋重叠（弱语义）：
- 文本 → token 集合（英文词 + 中文双字），token → 固定随机单位方向
- 文本向量 = token 向量之和并归一化 → 余弦相似度 ≈ 共享 token 比例
- 效果：query 与文档共享词越多越相似（契约排序断言可用）；稀有词单 token
  共享相似度 ~0.3（低于 min_similarity 0.35）→ BM25 路兜底意图仍可验证
- 完全确定性：同文本恒同向量；dim 由测试自选（默认 512，与存量断言自洽）
"""
import hashlib
import re

import numpy as np

DIM = 512


def _tokens(text: str) -> set[str]:
    t = (text or "").lower()
    toks = set(re.findall(r"[a-z0-9_]+", t))
    # 中文双字（bigram）——让中文文档间/中英混合有可度量的共享度
    for i in range(len(t) - 1):
        pair = t[i : i + 2]
        if any("\u4e00" <= c <= "\u9fff" for c in pair):
            toks.add(pair)
    return toks or {"<empty>"}


class FakeEmbed:
    """确定性伪 embed：词袋重叠余弦。接口兼容 EmbedClient（embed_one/batch/ping/close/dim）。

    构造签名对齐 EmbedClient(url, dim, model, query_instruction, timeout,
    connect_timeout, retries, batch_size, probe_timeout)——create_app 从 config
    构造时也直接可用（url/model/query_instruction 忽略，dim 固定测试值）。
    2026-09-18：补 batch_size / probe_timeout 形参——create_app 新增了这两个透传，
    测试替身签名不跟上的话集成测试会在构造处直接 TypeError。
    2026-09-20：同上补 connect_timeout（create_app 把 read/connect 超时拆开了）。
    """

    def __init__(self, url: str | None = None, dim: int = DIM, model: str | None = None,
                 query_instruction: str | None = None, timeout: float | None = None,
                 connect_timeout: float | None = None,
                 retries: int | None = None, batch_size: int = 32,
                 probe_timeout: float | None = None,
                 path: str | None = None, encoding_format: str | None = None,
                 load_path: str | None = None, load_timeout: float | None = None):
        self.dim = dim
        self.batch_size = batch_size
        self.probe_timeout = probe_timeout
        self.batch_calls: list[int] = []  # 每次 embed_batch 的输入数（分批契约断言用）

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for tok in _tokens(text):
            seed = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v += np.random.default_rng(seed).normal(size=self.dim)
        n = float(np.linalg.norm(v))
        return (v / n).tolist() if n else v.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(len(texts))
        return [self._vec(t) for t in texts]

    def probe_dim(self) -> int | None:
        """服务端维度探测（对齐 EmbedClient.probe_dim，2026-09-15 新增接口）。

        伪服务的维度恒等于自身 dim——即「配置与服务一致」，因此启动时的
        reconcile_embed_dim 永远不会清索引，既有测试的索引状态不受影响。
        """
        return self.dim

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass
