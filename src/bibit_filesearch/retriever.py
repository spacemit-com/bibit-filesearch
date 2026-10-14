"""检索服务：文件级向量检索 + 结构化筛选 + 融合排序。

链路：query embed + BM25 双路召回 → 余弦 topK 块（numpy）→ 文件聚合（max 池化）→
      结构化筛选（SQL 先行）→ 文件名/路径关键词融合 → confidence 归一化 → 排序分页。
融合：embed + 文件名 + BM25 三路，权重和 = 1（rerank / summary 已移除）
在该平台上开销过大，代码保留存档、默认关闭。
"""
from __future__ import annotations

import logging
import threading
import datetime as dt
from typing import Any

import jieba
import numpy as np

from bibit_filesearch.model_client import EmbedClient, normalize_scores  # noqa: E402
from bibit_filesearch.db import Database
from bibit_filesearch.parser import TEXT_TYPES, CODE_TYPES, IMAGE_TYPES, OFFICE_TYPES
from bibit_filesearch.bm25 import BM25Index, tokenize, _dedup  # noqa: E402

logger = logging.getLogger("bibit.retriever")

# 返回条数的兜底默认，取自 config 的出厂默认（单一来源，避免两处各写一个数）。
from bibit_filesearch.config import DEFAULTS as _CFG_DEFAULTS  # noqa: E402

_DEFAULT_TOP_K = int(_CFG_DEFAULTS.get("top_k", 10))

# 契约 fileTypes 枚举（开发计划.md 2.1 补充语义）
KNOWN_TYPES = TEXT_TYPES | CODE_TYPES | IMAGE_TYPES | OFFICE_TYPES | {"other"}

# 错误码（开发计划.md 2.3）
ERR_PARAM = 1001
ERR_FILTER = 1002
ERR_INTERNAL = 1004


class Retriever:
    def __init__(self, db: Database, embed: EmbedClient, candidate_pool: int = 1000,
                 top_k: int | None = None,
                 embed_weight: float = 0.538462, filename_weight: float = 0.230769, min_similarity: float = 0.35,
                 max_chunks: int = 100_000, bm25_weight: float = 0.0, dim: int | None = None,
                 bm25_half_score: float = 2.8):
        self.db = db
        self.embed = embed
        # 检索深度：最多取相似度最高的多少个块参与文件级聚合（BM25 路同值按文件数）。
        # **与 top_k 无关**——见 config.DEFAULTS["search"]["candidate_pool"] 的沿革说明。
        # 2026-09-15：此前该参数叫 top_candidates 且被接到用户的 top_k 上，导致「改显示
        # 条数会改排序」（某文件最佳块被块级窗口切掉）。改名即为此：名字里带 "top" 是
        # 当初误判「这和 top_k 是一回事」的诱因。<=0 表示不截断。
        self.candidate_pool = int(candidate_pool)
        # 单次检索最多返回条数（用户设置 top_k）。2026-09-11 前该值只被当作「召回池
        # 大小」，最终结果条数不受约束（多路合并 + 文件名路不受池约束），与 API 契约
        # 的「top_k：检索返回条数」和界面提示「单次检索最多返回的结果数」都不符——
        # 实测 top_k=3 仍返回 7 条。现在它就是硬上限：结果排序后截断到此条数。
        # 未显式传时取 config.DEFAULTS["top_k"]（单一来源，不再退回候选池——那正是
        # 被修掉的耦合）。
        self.top_k = int(top_k) if top_k is not None else _DEFAULT_TOP_K
        self.embed_weight = embed_weight
        self.filename_weight = filename_weight
        # 语义门槛：低于则从结果里剔除（纯 BM25 命中的文件不受此约束——它们没有
        # embed 分，见 _do_search 的过滤分支）。
        # 2026-09-15：出厂值 0.35 的注释曾写「实测校准：无关 0.14~0.28 / 相关 0.53+」，
        # 但那是**查询侧还带 query_instruction 前缀**时的测量——前缀把所有余弦整体
        # 压低。2026-09-09 清空前缀后噪声地板抬到 ~0.5 以上，门槛从此**不再触发**
        # （实测 10 组查询含 5 组纯噪声，无一被剔除）→ 界面「没有找到相关内容」空态
        # 不可达。详见 tests/test_min_similarity_gate.py 的标定守卫。
        # 改这个值必须同步重测标定数据。
        self.min_similarity = min_similarity
        # 向量维度（2026-09-03 解耦：矩阵预分配/扩容不再硬编码 512）。
        # 解析顺序：显式 dim 参数（main.py 传 config.models.embed.dim）> embed 客户端 dim 属性 > 0（
        # 0 = 未知，加载存量向量时自动探测）。维度不符在 _ensure_loaded 抛明确错误而非静默错乱。
        self._dim = int(dim or getattr(embed, "dim", None) or 0)
        # BM25 稀疏检索（D-BM25：多路召回，权重由调用方注入；0 = 关闭）
        self.bm25_weight = bm25_weight
        # BM25 绝对分 → 0~1 的映射常数（S_bm25 = raw / (raw + K)，K 即"半分点"）。
        #
        # 2026-09-29：**替换掉原来的批内 min-max 归一化**。min-max 把"此次命中集"里的
        # 最小分映成 0、最大分映成 1，于是命中集小且分数接近时会把"都命中"拉成
        # "一个满分一个零分"。实测（真机，query「身份」、8 个已完成文件）：
        #   id_card_1.png  BM25 原始分 2.3033 → min-max 1.0000
        #   id_rotate.png  BM25 原始分 2.3020 → min-max 0.0000
        #   两者只差 0.0013（同一张身份证的两个方向版本），却被分到 0.3 权重的两端，
        #   最终 score 0.3920 vs 0.1145，用户看到的是"一张排第一、一张排最后"。
        # 绝对映射下两者都是 2.30/(2.30+2.3) ≈ 0.5002 / 0.5003，并列。
        #
        # K 的取值：2026-09-29 判据法标定 2.8（同日分词三次变更 → 重标三次；
        # "取好命中中位分"的旧规则在原始分跨度大时不成立，被否取值与理由见
        # docs/bm25-score-calibration.md）。
        # ★ 语料变了（块变长/词表变）要重标一次：BM25 原始分随 avg_len 与 IDF 漂移。
        self.bm25_half_score = float(bm25_half_score or 0.0)
        self.bm25 = BM25Index()
        # 文件名/路径的 token 集合缓存（key=(name, path)）。见 _filename_scores。
        self._name_token_cache: dict[tuple[str, str], set[str]] = {}
        # Rerank 可选增强（D5 决策：K3 实测 ~155ms/短候选对，仅 ≤2 候选可进 500ms 完整查询预算）
        # 预热 jieba（首次 lcut 需加载词典 ~2s，预热后 ~1ms）
        jieba.initialize()
        # 内存向量矩阵（预分配，增量维护；查询免 SQLite 大 BLOB 传输）
        self._max_chunks = max_chunks
        self._fids: list[str] = []
        self._mat: np.ndarray | None = None
        self._n = 0
        self._loaded = False
        # 并发写保护：watcher 补扫/增量 + scanner worker 多线程调 on_file_indexed/on_file_deleted，
        # 内存矩阵与 BM25 倒排必须串行更新（D-5 加固后并发写入者 ≥2）
        self._matrix_lock = threading.Lock()
        # BM25 增量：块全局计数（warmup 全量重建后继续递增）
        self._bm25_chunk_counter = 0

    # ---------- 内存矩阵维护 ----------
    def _result_cap(self) -> int:
        """单次检索最多返回条数（用户 top_k）。<=0 视为不限（防御非法值）。"""
        return self.top_k if self.top_k and self.top_k > 0 else 10**9

    def warmup(self):
        """应用启动时预热：加载向量矩阵 + jieba 词典 + BM25 倒排（避免首次查询冷启动延迟）。"""
        self._ensure_loaded()
        # BM25 全量重建（服务启动时，从 chunks 表加载文本）
        #
        # 2026-09-21（A）：**不再按 `vector IS NOT NULL` 过滤**，与增量路径
        # （on_file_indexed）同口径。此前启动重建只收有向量的块、而增量路径收全部块，
        # 于是「同一个文件在 BM25 里有没有」取决于它是启动前还是启动后索引的 ——
        # 实测同一个 1 字节文件：warmup 建的倒排里没有、增量加的会有。
        # 极短块（A 之后无向量但有文本）必须被 BM25 收录，否则文件在语义路被挡、
        # 稀疏路也搜不到，等于彻底不可检索（如 a.txt 内容 `liuyingjie`）。
        if self.bm25_weight > 0 and self._bm25_chunk_counter == 0:
            rows = self.db.conn.execute(
                "SELECT file_id, text FROM chunks ORDER BY chunk_id"
            ).fetchall()
            for i, (fid, text) in enumerate(rows):
                self.bm25.add_document(fid, text or "")
            self._bm25_chunk_counter = len(rows)
        jieba.cut_for_search("预热分词")   # 与 tokenize 用同一模式，预热同一份词典

    def _ensure_loaded(self):
        """首次查询时从 SQLite 全量加载并归一化（10k 块 ≈ 200ms，一次性）。

        2026-09-03 维度防线：存量索引实际维度 ≠ 配置 dim（换 embed 模型未重索引的典型症状）
        → 立即抛明确、可操作的错误，禁止把 1024 维向量塞进 512 维矩阵的静默错乱/无声崩。
        """
        if self._loaded:
            return
        rows = self.db.conn.execute(
            "SELECT file_id, vector FROM chunks WHERE vector IS NOT NULL"
        ).fetchall()
        if rows:
            blob = b"".join(r["vector"] for r in rows)
            n = len(rows)
            if len(blob) == 0 or len(blob) % (n * 4) != 0:
                raise RuntimeError(
                    f"chunks 向量数据异常：{n} 条向量共 {len(blob)} 字节，无法整除 4×条数"
                    f"（疑似维度混杂或数据损坏）。若刚切换过 embed 模型，请清空旧索引并全库重索引。"
                )
            actual = len(blob) // (n * 4)  # float32 每维 4 字节
            if self._dim and actual != self._dim:
                raise RuntimeError(
                    f"向量维度不一致：配置/客户端 dim={self._dim}，存量索引实际 dim={actual}。"
                    f"若已更换 embed 模型（如 bge 512 维 → Qwen3 1024 维），旧索引不可复用："
                    f"请清空索引并全库重索引（删除 data 下 db 文件或触发重新索引）后再启动。"
                )
            if not self._dim:
                self._dim = actual  # 调用方未给 dim 时按存量向量自动探测
        self._alloc(len(rows))
        if rows:
            mat = np.frombuffer(blob, dtype=np.float32).reshape(len(rows), self._dim)
            # 预归一化（查询时免每次归一化拷贝）
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            self._mat[: len(rows)] = mat / norms
            self._fids = [r["file_id"] for r in rows]
            self._n = len(rows)
        self._loaded = True

    def _alloc(self, need: int):
        if self._dim <= 0:
            raise RuntimeError(
                "无法确定向量维度（dim=0）：请通过 config models.embed.dim 或 embed 客户端 dim 提供"
            )
        if self._mat is None:
            self._mat = np.zeros((max(self._max_chunks, need), self._dim), dtype=np.float32)

    def on_file_indexed(self, file_id: str, vectors: list[list[float]]):
        """增量追加（索引完成回调），追加前归一化。多线程安全（D-5 并发写）。

        替换语义：先移除该 file_id 的旧块向量再追加——否则内容变更重索引后旧内容仍可被
        检索命中（max 池化被旧块污染），且 _mat/_fids 只增不清（对齐 BM25 replace_file / SQL replace_chunks）。

        2026-09-21（A）：**空向量也必须先清旧行**。此前开头 `if not vectors: return`
        在移除之前早退，而「极短块不向量化」会让整个文件一个向量都产不出 →
        DB 里 vector 已置 NULL、内存矩阵里旧向量还活着，用户把文件改短后**反而搜得到
        旧内容**。实测复现：①正常索引→矩阵 1 行；②改短重索引→仍 1 行；
        ③本节改动前 `on_file_indexed(fid, [])` →残留 1 行（on_file_deleted 则清 0 行）。
        """
        with self._matrix_lock:
            self._ensure_loaded()
            # 移除该 file_id 的旧块（与 on_file_deleted 同策略：keep 列表压缩）
            if file_id in self._fids:
                keep = [i for i, f in enumerate(self._fids) if f != file_id]
                self._fids = [self._fids[i] for i in keep]
                if keep:
                    self._mat[: len(keep)] = self._mat[keep]
                self._n = len(keep)
            # BM25 同步（replace 语义）——必须在早退之前：极短块没有向量，但**有文本**，
            # 稀疏路要照常收录（否则 BM25 是否命中取决于「文件是启动前还是启动后索引的」）。
            if self.bm25_weight > 0:
                rows = self.db.conn.execute(
                    "SELECT text FROM chunks WHERE file_id=? ORDER BY chunk_id", (file_id,)
                ).fetchall()
                texts = [r["text"] or "" for r in rows]
                if texts:
                    self.bm25.replace_file(file_id, texts)
                    self._bm25_chunk_counter += len(texts)
            if not vectors:
                return
            arr = np.array(vectors, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self._dim:
                raise RuntimeError(
                    f"增量索引向量维度与检索矩阵不符：{arr.shape[1] if arr.ndim == 2 else '?'} ≠ dim={self._dim}。"
                    f"若已切换 embed 模型，请先清空旧索引并全库重索引，再增量索引。"
                )
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            arr = arr / norms
            need = self._n + len(arr)
            if need > self._mat.shape[0]:
                # 扩容（拷贝，罕见）——维度用 self._dim（原硬编码 512，2026-09-03 解耦）
                new_mat = np.zeros((max(2 * self._mat.shape[0], need), self._dim), dtype=np.float32)
                new_mat[: self._n] = self._mat[: self._n]
                self._mat = new_mat
            self._mat[self._n : need] = arr
            self._fids.extend([file_id] * len(arr))
            self._n = need

    def on_file_deleted(self, file_id: str):
        """删除后重建矩阵（删除低频，重建可接受）。多线程安全。"""
        with self._matrix_lock:
            keep = [i for i, f in enumerate(self._fids) if f != file_id]
            self._fids = [self._fids[i] for i in keep]
            if keep:
                self._mat[: len(keep)] = self._mat[keep]
            self._n = len(keep)
            self._loaded = True
            # BM25 同步删除
            if self.bm25_weight > 0:
                self.bm25.remove_file(file_id)

    # ---------- 入口 ----------
    def search(self, payload: dict) -> dict:
        """POST /api/search 契约实现。返回完整响应体。"""
        # 2026-09-09：校验前置到所有分支之前。此前纯筛选（空 query + filters）绕过
        # _validate 直接进 _do_search，page/pageSize 非法时在 `-(-total // page_size)`
        # 抛 ZeroDivisionError/TypeError → HTTP 500（实测 page="abc"/pageSize="x"）。
        # 空 query + filters 有值仍是合法的纯筛选场景（见 _validate）。
        err = self._validate(payload)
        if err:
            return err
        query = (payload.get("query") or "").strip()
        filters = payload.get("filters") or {}
        page = payload.get("page")
        page_size = payload.get("pageSize")

        try:
            return self._do_search(query, filters, page, page_size)
        except Exception as e:
            logger.exception("search 内部错误")
            return {"code": ERR_INTERNAL, "message": "服务内部错误", "data": None}

    # ---------- 校验 ----------
    def _validate(self, payload: dict) -> dict | None:
        q = (payload.get("query") or "").strip()
        f = payload.get("filters") or {}
        # 空 query 合法当且仅当带了 filters（纯筛选检索，2026-08-26 语义）
        if not q and not f:
            return {"code": ERR_PARAM, "message": "query 不能为空", "data": None}
        page = payload.get("page")
        page_size = payload.get("pageSize")
        # 显式 null 视为未传（用默认值）；显式 0 必须判非法——不能 `or 1` 吞掉 0
        if page is None:
            page = 1
        if page_size is None:
            page_size = 20
        # 2026-09-09：类型必须严格 int（bool 是 int 子类，显式排除）——此前 API 层
        # 先 int() 强转，非数字串在这里变成 500、显式 0 被 `or 1` 吞成 1 静默放行。
        if isinstance(page, bool) or not isinstance(page, int) or page < 1 or page > 10000:
            return {"code": ERR_PARAM, "message": "page 取值非法（1 ~ 10000）", "data": None}
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1 or page_size > 100:
            return {"code": ERR_PARAM, "message": "pageSize 取值非法（1 ~ 100）", "data": None}
        for t in f.get("fileTypes") or []:
            if t not in KNOWN_TYPES:
                return {"code": ERR_FILTER, "message": f"未知 fileType: {t}", "data": None}
        dr = f.get("dateRange") or {}
        if dr.get("start") and dr.get("end") and dr["start"] > dr["end"]:
            return {"code": ERR_FILTER, "message": "时间范围起始日期不能晚于结束日期", "data": None}
        return None

    # ---------- 主检索 ----------
    def _do_search(self, query: str, filters: dict, page: int, page_size: int) -> dict:
        # 1. SQL 筛选候选文件（fileTypes / dateRange / directories 任意组合）
        candidates = self._filter_files(filters)
        if not candidates:
            return self._empty_response(page, page_size)

        # 1.5 空 query（2026-08-26 关键词净化后可能为空串）→ 纯筛选检索：
        #     候选文件直接返回（按 modifiedAt 降序，无语义排序），跳过 embed/BM25。
        #     对应 NL 场景「给我找出上周修改过的markdown」→ keywords='' + filters={md, 上周}。
        if not (query or "").strip():
            results = []
            for fid, f in candidates.items():
                results.append({
                    "fileId": fid,
                    "fileName": f["name"],
                    "filePath": f["path"],
                    "fileType": f["file_type"],
                    "fileSize": f["size"],
                    "createdAt": f["created_at"],
                    "modifiedAt": f["modified_at"],
                    "author": f["author"],
                    # 纯筛选检索无相关度概念（未经向量/BM25 打分）→ 绝对值恒 0
                    "score": 0.0,
                    "confidence": 0.0,
                })
            results.sort(key=lambda r: -dt.datetime.fromisoformat(r["modifiedAt"]).timestamp())
            results = results[: self._result_cap()]  # Top-K 硬上限（见 top_k 注释）
            total = len(results)
            total_pages = max(1, -(-total // page_size))
            cur = min(page, total_pages)
            start = (cur - 1) * page_size
            return {
                "code": 0,
                "message": "success",
                "data": {
                    "total": total,
                    "page": cur,
                    "pageSize": page_size,
                    "totalPages": total_pages,
                    "results": results[start : start + page_size],
                },
            }

        # 2. 双路召回（D-BM25）：embed 语义路 + BM25 稀疏关键词路（内容级）
        #    BM25 路对专业词/代码符号一击必中；embed 路对自然语言更强
        bm25_scores: dict[str, float] = {}
        if self.bm25_weight > 0:
            bm25_scores = self.bm25.search(query, top_k=self._pool_size())
            # 与筛选候选取交集（BM25 索引含全部文件，筛选后仅保留候选内文件）
            bm25_scores = {f: s for f, s in bm25_scores.items() if f in candidates}

        # 3. query embed
        q_vec = np.array(self.embed.embed_one(query), dtype=np.float32)

        # 4. 余弦 topK 块 → 文件聚合（max 池化）
        file_scores = self._cosine_aggregate(candidates, q_vec)

        # 5. 合并去重（多路召回）：embed 命中 ∪ BM25 命中
        merged = dict(file_scores)
        if bm25_scores:
            for fid in bm25_scores:
                merged.setdefault(fid, 0.0)

        if not merged:
            return self._empty_response(page, page_size)
        # 语义门槛：低于门槛的「仅向量命中」文件从结果里剔除。
        # 约定（与 bm25_weight/filename_weight 一致）：<=0 视为**关闭门槛**——0 是
        # 「不设最低相似度」的自然写法，而不是「余弦必须非负」。出厂默认 0.35。
        # 2026-09-15 修复：此前只从 merged 里删，但下游的归一化与结果组装遍历的是
        # file_scores → **被剔除的文件照样出现在结果里**（只有 merged 全空时的早退
        # 有效）。即门槛自加入以来从未真正过滤过输出，与标定是否准确无关。
        # 纯 BM25 命中的文件（在 merged 里但不在 file_scores 里）不受门槛约束——
        # 它们没有 embed 分；第 6.5 步的融合循环会遍历 merged 把它们补进 file_scores。
        if self.min_similarity > 0:
            for fid in list(merged):
                if fid in file_scores and file_scores[fid] < self.min_similarity and fid not in bm25_scores:
                    del merged[fid]
            if not merged:
                return self._empty_response(page, page_size)
            # 把 file_scores 收敛到存活集合，让门槛真正作用于输出
            file_scores = {fid: s for fid, s in file_scores.items() if fid in merged}

        # 6/6.5 三路加权融合（2026-09-29 重写）
        #
        #     score = (embed_weight·S_sem + filename_weight·S_name + bm25_weight·S_bm25) / ΣW
        #
        # 改前的问题（实测复现，query「身份」、8 个已完成文件）：
        #
        #   (a) 权重乘了两次。原 step 6 先把向量分乘一次 embed_weight
        #       （无文件名命中时 `file_scores[fid] *= self.embed_weight`），
        #       原 step 6.5 的融合又乘一次。于是 cos 的实际系数是 0.7×0.7=**0.49**
        #       而不是注释里写的 0.7；文件名路的 0.3 也被同一除法折成 0.21。
        #       实测校验：0.7×0.7×0.1875 + 0.3×1.0 = 0.3919，与接口返回的 0.3920 逐位吻合。
        #   (b) BM25 用批内 min-max 归一。那个「一个满分一个零分」的病态见
        #       `self.bm25_half_score` 的注释（极差 0.0013 被拉成 1.0/0.0）。
        #
        # 现在：三路各自先算 0~1 的**绝对**分，再一次性加权平均，权重只乘一次。
        #   S_sem  = 块级余弦的文件级 max 池化（step 3 已算好，尺度即 cos）
        #   S_name = jieba 命中率（0~1，绝对）
        #   S_bm25 = raw/(raw+K)（绝对；无命中 = 0）
        kw_scores: dict[str, float] = {}
        if self.filename_weight > 0:
            kw_scores = self._filename_scores(query, candidates)

        S_sem = dict(file_scores)                                   # 复用 step 3 的池化结果
        S_name = dict(kw_scores)
        S_bm25: dict[str, float] = {}
        if self.bm25_weight > 0 and bm25_scores:
            K = self.bm25_half_score
            for fid, r in bm25_scores.items():
                S_bm25[fid] = (r / (r + K)) if (K > 0 and r > 0) else (1.0 if r > 0 else 0.0)

        # 逐文件加权平均。`merged` 是「语义命中 ∪ BM25 命中」的全集 —— 遍历它保证
        # 纯 BM25 命中（无向量分）的文件不会被漏掉，这是本函数原有的语义，保留。
        total_w = self.embed_weight + self.filename_weight + self.bm25_weight
        if total_w <= 0:                     # 三个权重全 0：退化为纯语义
            total_w = 1.0
        file_scores = {}
        for fid in merged:
            file_scores[fid] = (
                self.embed_weight * S_sem.get(fid, 0.0)
                + self.filename_weight * S_name.get(fid, 0.0)
                + self.bm25_weight * S_bm25.get(fid, 0.0)
            ) / total_w

        # 4.5 可选 Rerank 增强（默认关闭，开启时 ≤3 候选精排）

        # 5. 归一化 confidence（0~1）
        confidences = self._normalize(list(file_scores.values()))

        # 6. 组装 + 排序（confidence 降序，同分 modifiedAt 降序）+ 分页
        results = []
        for fid, conf in zip(file_scores.keys(), confidences):
            f = candidates[fid]
            results.append({
                "fileId": fid,
                "fileName": f["name"],
                "filePath": f["path"],
                "fileType": f["file_type"],
                "fileSize": f["size"],
                "createdAt": f["created_at"],
                "modifiedAt": f["modified_at"],
                "author": f["author"],
                # score：未归一化的绝对融合分（典型 0~1，余弦为负时理论上可略低于 0）。
                # **不要直接展示给用户**——它的尺度只在同一 embedding 配置下可比。存在
                # 的意义是给上游留一个不撒谎的排序/分档依据（confidence 是 min-max 归一化，
                # 每次检索头名恒 100%、末名恒 0%，只能读相对名次）。
                "score": round(float(file_scores[fid]), 4),
                "confidence": round(float(conf), 4),
            })
        results.sort(key=lambda r: (-r["confidence"], -dt.datetime.fromisoformat(r["modifiedAt"]).timestamp()))
        results = results[: self._result_cap()]  # Top-K 硬上限（见 top_k 注释）

        total = len(results)
        total_pages = max(1, -(-total // page_size))
        cur = min(page, total_pages)
        start = (cur - 1) * page_size
        return {
            "code": 0,
            "message": "success",
            "data": {
                "total": total,
                "page": cur,
                "pageSize": page_size,
                "totalPages": total_pages,
                "results": results[start : start + page_size],
            },
        }

    # ---------- 筛选 ----------
    def _filter_files(self, filters: dict) -> dict[str, dict]:
        """SQL 先行筛选 → {file_id: file_row}。筛选条件为空则全部。"""
        where, args = [], []
        f = filters or {}
        if f.get("fileTypes"):
            where.append(f"file_type IN ({','.join('?' * len(f['fileTypes']))})")
            args.extend(f["fileTypes"])
        dr = f.get("dateRange") or {}
        # D2（2026-08-25）：dateField 选择时间戳列——modified（默认）/ created / accessed。
        # 白名单映射防 SQL 注入（列名只可能来自 dict 值）。
        _DATE_COLS = {"modified": "modified_at", "created": "created_at", "accessed": "accessed_at"}
        _date_col = _DATE_COLS.get((dr.get("dateField") or "modified"), "modified_at")
        if dr.get("start"):
            where.append(f"{_date_col} >= ?")
            args.append(dr["start"])
        if dr.get("end"):
            # 2026-08-26：闭区间 end 当天含——时区偏移动态取本地（原硬编码 Z=UTC，
            # 与 metadata 本地时区时间戳字符串比较会错位）
            _tz_off = dt.datetime.now().astimezone().strftime("%z") or "+08:00"
            where.append(f"{_date_col} <= ?")
            args.append(dr["end"] + f"T23:59:59{_tz_off}")
        dirs = f.get("directories") or []
        if dirs:
            # 目录递归前缀，大小写敏感（Linux 语义）——SQLite LIKE 对 ASCII 不区分大小写，用 GLOB
            conds = " OR ".join(["(directory = ? OR directory GLOB ?)"] * len(dirs))
            for d in dirs:
                d = d.rstrip("/")
                args.append(d)
                args.append(d + "/*")
            where.append(f"({conds})")

        sql = "SELECT * FROM files WHERE " + " AND ".join(where) if where else "SELECT * FROM files"
        # 仅候选已完成向量化的文件（vectorize_status='completed'）——未向量化文件无块可检索
        sql += " AND vectorize_status='completed'" if where else " WHERE vectorize_status='completed'"
        rows = self.db.conn.execute(sql, args).fetchall()
        return {r["file_id"]: dict(r) for r in rows}

    # ---------- 余弦 + 聚合 ----------
    def _pool_size(self) -> int:
        """候选池的实际限量。<=0 按「不截断」处理（与 min_similarity<=0、max_depth=0 同款约定）。

        BM25 路直接拿它当 `[:n]` 切片长度，故负数必须归一为大整数——切片负值会从尾部
        反删（`[: -5]` 语义），悄悄丢结果。
        """
        return self.candidate_pool if self.candidate_pool > 0 else 10 ** 9

    def _cosine_aggregate(self, candidates: dict[str, dict], q_vec: np.ndarray) -> dict[str, float]:
        """块级余弦 → 文件级 max 池化（D7 决策）。

        用内存矩阵：mask 候选文件块 → 单次 matmul。实测 1 万块 ≈ 5~15ms。
        """
        self._ensure_loaded()
        if self._n == 0:
            return {}
        cand = set(candidates)
        # mask 出候选文件的块（list comp 比 np.isin 快）
        idx = [i for i, f in enumerate(self._fids) if f in cand]
        if not idx:
            return {}
        sub = self._mat[idx]
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        sims = sub @ q_norm

        k = min(self._pool_size(), len(sims))
        if k >= len(sims):
            # 池 >= 候选块数 = 不截断。此时 argpartition 白算一遍，直接全量排序
            # （实测 1 万块 0.8ms；比走分区再排序还快）。
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        file_scores: dict[str, float] = {}
        for i in top_idx:
            fid = self._fids[idx[int(i)]]
            s = float(sims[int(i)])
            if s > file_scores.get(fid, -1.0):
                file_scores[fid] = s  # max 池化
        return file_scores

    def _filename_scores(self, query: str, candidates: dict[str, dict]) -> dict[str, float]:
        """文件名/路径的 jieba 命中率 → 0~1 得分。

        2026-09-29（P12/P13）：**改用与 BM25 同一套分词与匹配规则**。

        改前用的是 `jieba.lcut`（精确模式）+ 子串包含（`w in hay`）。两处不一致：
          · 分词器不一致：BM25 已换 `cut_for_search`，这里还是 `lcut` →
            查询「居民身份证」时 words=['居民身份证']，而文件名「身份证扫描件.pdf」
            里的「身份证」没有独立词命中 → 该文件漏掉文件名分。
          · 匹配规则不一致：这里子串包含、BM25 token 相等 → 同一份配置里两个
            「0.3」，语义却不同。
        现在两处都用 `tokenize`（含 `cut_for_search`）+ **token 集合相交**，
        于是两条词法通道问的是同一个问题：「查询的词，在不在名字/路径里」。
        `_dedup` 与 BM25 同款（同一概念不重复计数）。
        """
        toks = set(_dedup(tokenize(query)))
        if not toks:
            return {}
        # ★ 名字/路径的 token 集合走缓存：一次检索最多上百个候选，逐候选现分词
        #   会让 P95 从 ~100ms 涨到 ~120ms（实测 test_t5_search 的 P95 基准失败）。
        #   jieba 是纯函数、进程内不变，故按 (name, path) 缓存是安全的。
        cache = self._name_token_cache
        scores: dict[str, float] = {}
        for fid, f in candidates.items():
            key = (f["name"], f["path"])
            name_toks = cache.get(key)
            if name_toks is None:
                name_toks = set(_dedup(tokenize(key[0] + " " + key[1])))
                # 简单上限，避免长期运行下无界增长（条目很小，8k 足够覆盖典型库）
                if len(cache) < 8192:
                    cache[key] = name_toks
            hits = len(toks & name_toks)
            if hits:
                scores[fid] = hits / len(toks)
        return scores

    # ---------- 归一化 ----------
    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        """批内 min-max 归一——**现在只用于 `confidence` 展示**（2026-09-29）。

        不要再用它做排序依据（BM25 曾用它，导致"命中集小且分数接近时把都命中拉成
        一个满分一个零分"，见 `self.bm25_half_score` 的实测）。min-max 的性质就是
        "名次映射"：它保证头名恒 1.0、末名恒 0.0，与绝对相关度无关。作为**给人看的
        0~100% 相对强弱**它是够的（接口注释也写明 confidence 只能读相对名次）；
        作为**排序**依据它不行——排序认 `score`（绝对融合分）。
        """
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-9:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    @staticmethod
    def _empty_response(page: int, page_size: int) -> dict:
        return {
            "code": 0,
            "message": "success",
            "data": {"total": 0, "page": page, "pageSize": page_size, "totalPages": 1, "results": []},
        }
