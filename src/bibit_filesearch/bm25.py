"""轻量 BM25 稀疏检索（自写倒排表，无第三方依赖）。

D-BM25 决策（方案 2 自写）：
- 块级倒排索引：chunk_id → 词频；term → 出现块（posting list）
- 中文分词：jieba；英文/代码 token：按字母数字连续段保留（`wl_surface.commit()` → wl_surface/commit）
- 检索：query 分词 → 每词查倒排 → 块级 BM25 得分 → 按文件聚合（max 池化，与 embed 路一致）
- 融合：外部（retriever）做，本模块只算 BM25 得分

指标：k1=1.5, b=0.75（BM25 经典默认）。
"""
from __future__ import annotations

import math
import re

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEPS = _PKG_ROOT / "deps"
if str(_DEPS) not in sys.path:
    sys.path.insert(0, str(_DEPS))

import jieba

# 代码/英文 token：字母数字连续段（保留 wl_surface、commit、dma_fence 等完整 token）
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# 中文连续段
_CN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """混合分词：英文/代码 token 完整保留 + 中文走 jieba **搜索模式**。

    2026-09-09 起中文用 `jieba.lcut`（精确模式），2026-09-29 改为 `cut_for_search`。

    原因：精确模式对同一串字在**文档侧**与**查询侧**会切出不同长度的词，而 BM25
    要求 token 精确相等，于是产生召回洞：

        文档「…居民身份证…」 → ['居民身份证']
        查询「身份」         → ['身份']            ← 不等 → 匹配失败

    真机后果（query「身份」，8 个已完成文件）：`id_card_2.jpg` 拿不到「BM25 命中」
    豁免，又被语义门槛（cos 0.2675 < 0.35）挡住 → **整份文件被剔除**；同批的
    `id_card_1` / `id_rotate` 文本写的是「公民身份号码」，恰好切出「身份」所以命中。
    同一个人的同一张证件，只因 OCR 文本措辞不同就一进一出。

    修法选了 `cut_for_search` 而不是"补相邻双字（bigram）"：两者都能补上召回，
    但双字会产出跨词边界的噪声 token（'民身'、'份号'、'国居'…）——它们既污染
    tf 语义，又把索引撑大（实测 avg_len 1369 → 2576）。搜索模式产出的**全部是
    词典认可的词或词的真子词**，是同一套语言学规则同时作用于文档与查询，故两侧
    天然对齐。实测对比（『中华人民共和国居民身份证』）：

        精确模式  → 中华人民共和国, 居民身份证                       (2 个)
        搜索模式  → 中华人民共和国, 中华, 华人, 人民, 共和, 共和国,
                    居民, 居民身份证, 身份, 身份证                    (10 个，全是真词)
        双字兜底  → 上列 + 民共, 和国, 国居, 民身, 份证              (13 个，含 5 个噪声)

    代价：搜索模式只补**词典认得**的子词，故召回面略窄于无脑双字（例：'票号'
    不是 jieba 认可的子词，双字能对上、搜索模式不能）。当前语料下目标查询
    （身份/身份证/居民身份证/公民身份号码）全部命中，未见丢失。

    英文/代码 token 的行为不变（`wl_surface` 等仍完整保留）。
    """
    tokens: list[str] = []
    # 先按非字母数字字符切出英文/代码 token（含下划线）
    for seg in _TOKEN_RE.findall(text):
        s = seg.lower()
        if len(s) >= 2:  # 单字符 token 噪声大
            tokens.append(s)
    # 中文连续段走 jieba 搜索模式（长词补出词典认可的子词）
    for cn in _CN_RE.findall(text):
        for w in jieba.cut_for_search(cn):
            w = w.strip()
            if len(w) >= 2:
                tokens.append(w)
    return tokens


def _dedup(tokens: list[str]) -> list[str]:
    """去重保序 —— 只用于**查询**侧。

    `cut_for_search` 会让一个词以多种粒度出现（「身份证」→ '身份证' + '身份'），
    而 `search()` 是逐 token 累加倒排贡献的。同一概念被算多次会让该查询的分数
    虚高，故查询侧按"出现即计"（BM25 的标准二元查询权重）去重。

    ★ 文档侧**不能**去重：tf 是 BM25 的核心分量，去重会把 tf 抹成 1。
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class BM25Index:
    """倒排索引 + BM25 打分。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[int, int]] = {}  # term -> {chunk_idx: tf}
        self._doc_len: list[int] = []  # chunk_idx -> token 数
        self._chunk_files: dict[int, str] = {}  # chunk_idx -> file_id
        self._total_len = 0

    # ---------- 构建 ----------
    def add_document(self, file_id: str, text: str):
        """入库一块（增量维护：仅新增块调用）。

        块序号由内部计数器分配（chunk_idx 参数此前从未被使用，接口误导，已移除）。
        """
        toks = tokenize(text)
        if not toks:
            return
        self._doc_len.append(len(toks))
        self._chunk_files[len(self._doc_len) - 1] = file_id
        self._total_len += len(toks)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        idx = len(self._doc_len) - 1
        for t, c in tf.items():
            self._postings.setdefault(t, {})[idx] = c

    def remove_file(self, file_id: str):
        """删除文件的全部块（增量维护）。"""
        # 重建倒排（文件删除低频，重建可接受；与 retriever.on_file_deleted 策略一致）
        keep = [i for i, f in self._chunk_files.items() if f != file_id]
        new_idx = {old: new for new, old in enumerate(keep)}
        new_postings: dict[str, dict[int, int]] = {}
        new_len: list[int] = []
        new_files: dict[int, str] = {}
        total = 0
        for old, new in new_idx.items():
            new_files[new] = self._chunk_files[old]
            new_len.append(self._doc_len[old])
            total += self._doc_len[old]
        for term, posts in self._postings.items():
            np = {}
            for old, tf in posts.items():
                if old in new_idx:
                    np[new_idx[old]] = tf
            if np:
                new_postings[term] = np
        self._postings = new_postings
        self._doc_len = new_len
        self._chunk_files = new_files
        self._total_len = total

    def replace_file(self, file_id: str, texts: list[str]):
        """替换文件的全部块（文件内容变更重算：先删旧块再添加，避免新旧内容并存）。"""
        self.remove_file(file_id)
        for t in texts:
            self.add_document(file_id, t)

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 20) -> dict[str, float]:
        """BM25 检索 → {file_id: 文件级 max 池化得分}。"""
        # 查询侧去重（搜索模式让一个词以多粒度出现 → 同一概念被算多次）。见 _dedup 注释。
        toks = _dedup(tokenize(query))
        if not toks or not self._postings:
            return {}
        n_docs = max(len(self._doc_len), 1)
        avg_len = self._total_len / n_docs if n_docs else 1.0
        # 文档频率
        df = {t: len(self._postings.get(t, {})) for t in toks if t in self._postings}
        if not df:
            return {}
        idf = {t: math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for t, d in df.items()}

        scores: dict[int, float] = {}
        for t in toks:
            if t not in self._postings:
                continue
            for chunk_idx, tf in self._postings[t].items():
                dl = max(self._doc_len[chunk_idx], 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avg_len)
                scores[chunk_idx] = scores.get(chunk_idx, 0.0) + idf[t] * (tf * (self.k1 + 1)) / denom

        # 文件级聚合（max 池化，与 embed 路一致）
        file_scores: dict[str, float] = {}
        for chunk_idx, s in scores.items():
            fid = self._chunk_files[chunk_idx]
            if s > file_scores.get(fid, 0.0):
                file_scores[fid] = s
        # 取 top_k 文件
        top = sorted(file_scores.items(), key=lambda x: -x[1])[:top_k]
        return dict(top)
