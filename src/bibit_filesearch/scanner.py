"""全盘扫描服务——重构后降级为「批量生产者」。

两阶段状态机：
- 遍历阶段（快）：os.walk 配置目录 → 逐个 `IndexPool.offer()` 放进索引池
                 → 检测磁盘删除（从池中撤掉）
- 向量化阶段（慢）：**由常驻消费者 `IndexWorker` 负责**（见 pool.py）。
  本模块不再自己消费队列 —— 历史上「scanner 一条路径 + watcher 另一条路径」
  正是状态分裂的根源。

暂停/继续作用于整个任务（遍历 + 向量化都停）；状态持久化（scan_state + files.vectorize_status），
重启后服务自动触发扫描，从持久化队列续跑。

职责边界（设计文档 §3.2）：
  本模块 = 生产者：只 offer / purge 池行，不调 `indexer.index_file()`
  消费者   = `pool.IndexWorker`：只 take / commit
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from bibit_filesearch.excludes import is_excluded
from bibit_filesearch.metadata import extract_metadata
from bibit_filesearch.parser import IMAGE_TYPES
from bibit_filesearch.pathutil import describe_unencodable_filename, display_path
from bibit_filesearch.pool import IndexPool, IndexWorker
from bibit_filesearch.state import ScanState, now_iso

logger = logging.getLogger("bibit.scanner")

STATUS_IDLE = "idle"
STATUS_SCANNING = "scanning"
STATUS_PAUSED = "paused"
STATUS_DONE = "done"

# 「未纳入索引」明细样本的条数上限（2026-09-23）。
# `_excluded_count` 是**全量真值**，不受此限；这里限的只是随 progress 一起下发的
# 明细条数 —— 被剪掉的子树可能有成千上万个文件，全带出去既占内存也拖慢 2 秒一次的
# 轮询。超限时置 `_excluded_overflow`，界面据此如实说明「只列出前 N 条」，
# 保证「格上的数」与「展开的条数」不会互相打架（这正是失败文件那格踩过的坑，见 F7）。
_EXCLUDED_DETAILS_CAP = 200


class ScanService:
    def __init__(self, config, db, indexer, pool: IndexPool | None = None):
        self.index_dirs = [str(d) for d in config.index_dirs]
        self.excludes = list(config.exclude_patterns)
        self.io_throttle_ms = config.scan.get("io_throttle_ms", 2)
        # 索引深度（1-based，2026-09-04）：0=无限；N=base+下 N-1 层子目录（1=仅 base 本身）
        self.max_depth = int(config.scan.get("max_depth", 0) or 0)
        # 2026-09-18：单文件最多重试次数（达上限即置终态失败、不再每轮重排）。
        # 原实现失败文件既不计次也不落 hash，导致每轮扫描都把全部失败文件当「待重算」
        # 重灌一遍（实测 113 个文件 / 约 6.8 万个 embed 输入，每轮全量）。
        self.max_attempts = int(config.scan.get("max_attempts", 3) or 3)
        self.db = db
        self.indexer = indexer
        self.state = ScanState(db)
        # 索引池 + 常驻消费者（重构）：状态真相收在池子里，见 pool.py 的模块文档。
        # pool 可由外部注入（测试 / main.py 需要把它同时交给 watcher）。
        self.pool = pool if pool is not None else IndexPool(
            db, indexer, max_attempts=self.max_attempts)
        self.worker = IndexWorker(self.pool, indexer, scanner=self)

        self._lock = threading.Lock()
        self._traverse_thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._pause_event.set()  # 非暂停态
        self._status = STATUS_IDLE
        self._walked = 0
        self._walk_total = 0
        self._walk_errors = 0
        self._error_logs: list[dict] = []  # 本次扫描错误明细（Q2-3 拆分 kind）
        # 未纳入索引（本轮读数）：命中排除规则 / 超出 max_depth / 不在 index_dirs 内。
        # 这些文件**连池行都没有**，此前四个观测面全静默（无行、无计数、无明细、无日志）
        # —— 「我的文件怎么没被索引」最常见的答案就在这里。每轮遍历复位。
        #
        # 2026-09-23（第二版）：除计数外再留**一份明细样本**（excludedDetails），
        # 供界面「查看未纳入索引文件」展开 —— 只有个数没有「是哪些」时，用户仍然
        # 查不下去（这与「失败文件」格的做法一致：数 + 明细同源）。
        # 有上限（_EXCLUDED_DETAILS_CAP）：被剪掉的子树可能成千上万，全存会白占内存；
        # `_excluded_count` 仍是**全量真实计数**，两者不等时界面会如实说明。
        self._excluded_count = 0
        self._excluded_details: list[dict] = []
        self._excluded_overflow = False  # 明细样本是否已溢出（用于界面补一句说明）
        self._excluded_kinds: dict[str, int] = {}  # 按原因细分：rule / depth
        # 2026-10-10：增量侧重算「遍历侧排除读数」的节流/挂起位。
        #   · `_excluded_dirty` —— 有变更需要重算，但还没重算（watcher 打标）。
        #   · `_excluded_recount_at` —— 上次重算的单调时刻（节流，避免事件风暴里反复全走）。
        # 为什么需要：这份读数原先**只有全盘遍历**会填，watcher 增量路径完全不碰它。
        # 于是把文件移出被排除目录后（`mv excluded/* .`），文件被正常索引成 completed，
        # 而「未纳入索引」仍停在上一轮的旧值 + 旧明细里，要等下一次全盘扫描才收敛
        # （用户实测：已索引已到 6，未纳入索引还显示 14）。见 recount_excluded()。
        self._excluded_dirty = False
        self._excluded_recount_at = 0.0

    # ---------- 状态 ----------
    @property
    def status(self) -> str:
        return self._status

    def progress(self) -> dict:
        """两阶段进度。

        2026-09-21：出接口前把 path 安全化。文件系统允许名字不是 UTF-8（Windows/GBK
        环境拷进来的文件，实测 `cuda/docs/` 下两个 GBK 名字），`os.walk` 会把它经
        surrogateescape 变成含 `\\udcc9` 的字符串 —— 那种字符串 `sqlite3` 与 JSON 序列化
        都编不出来。此前明细里的原始路径会一路带到 `JSONResponse.render()` 的
        `.encode("utf-8")` 上炸掉，**整个端点 500**（实测：17:54:22 起界面「服务不可用」，
        而 /health/deep 与 /api/search 都正常）。
        判存在（下一步的 os.path.exists）必须用**原路径**，只有出接口的副本换
        display_path —— 见 pathutil 的说明。
        """
        with self._lock:
            walked, total, errors = self._walked, self._walk_total, self._walk_errors
            status = self._status
            excluded = self._excluded_count
            excluded_details = list(self._excluded_details)
            excluded_overflow = self._excluded_overflow
            excluded_kinds = dict(self._excluded_kinds)
            traverse_alive = bool(self._traverse_thread and self._traverse_thread.is_alive())
        counts = self.pool.counts()
        # 2026-09-23（重构）：**对外状态 = 池子里还有没有活在干**。
        #
        # 此前判据含 `indexer.inflight_indexes`（进程内整数）—— 它挂在两条路径的
        # 公共入口上，于是 scanner 正在算的文件被数两遍（实测「待向量化」稳定多算 1，
        # 8 个文件的库算出 9）。池子化后 watcher 也写池行，那个整数不再需要，
        # 判据只剩池行状态：queued > 0（有排队）或 vectorizing > 0（有在算）。
        #
        # 设计依据：
        #   done ⟺ 池中 queued == 0 ∧ vectorizing == 0
        # 不再依赖「遍历线程是否存活」这个旁证——全盘遍历结束时只保证「该放的都放完了」。
        queued = counts.get("queued", 0)
        vectorizing = counts.get("vectorizing", 0)
        if status == STATUS_DONE and (queued > 0 or vectorizing > 0):
            status = STATUS_SCANNING
        # 2026-09-09：剔除磁盘上已不存在的路径——文件删除回收后明细残留，会出现
        # "失败 0 但展开还有条目"（用户实测：删掉失败文件后条目仍在）。权限类错误
        # （目录不可读）路径仍存在，不受影响。
        # 2026-09-21：判存在用**原路径**，出接口的副本换 display_path（见方法文档）。
        details = [dict(d, path=display_path(d["path"]))
                   for d in self._error_logs
                   if not d.get("path") or os.path.exists(d["path"])]
        # 2026-09-21：截断**移出** errorDetails，单列一份。
        # 它此前由 worker 调 _record_error("oversized", ...) 记进来，造成三个问题：
        #   ① 语义错位——文件其实索引成功、64 块全部入库、完全可检索，只是内容不全，
        #      把它列在「失败文件」下会让用户以为坏了；
        #   ② 计数与列表对不上——「失败文件」计数取 vectorizePhase.failed（真失败），
        #      而这个列表是 errorDetails（含截断），于是出现「失败 1，展开列 12 条」；
        #   ③ 生命周期错位——截断是**持久**属性，却被放进「本轮扫描」的明细，扫描一结束
        #      或被下一轮清空后就再也看不到（用户实测：界面只字不提「只索引了一部分」）。
        # 改按 DB 求值（db.truncated_files）：持久、可跨轮查看，且天然与 files.total_chunks
        # 同源——那个字段本来就是为「让截断可见」而加的。
        # 2026-09-21：判存在用**原路径**，出接口的副本换 display_path（理由同 errorDetails）
        # 2026-09-23（重构 P4）：`unsupported` 拆成两格 + 「未纳入索引」独立给出。
        #
        # 为什么必须拆：`unsupported` 有**两个写入点**，语义完全不同，此前合成一格：
        #   A 类型没勾（`file_type ∉ indexable_types`）——「根本没打算索引它」
        #   B 解析了但没文本（`∈ types` 且 hash 非空）——「试过了，读不出内容」
        # 判别零成本：`file_type ∈ indexable_types`。混在一起时，用户取消勾选一个格式
        # 会让「已索引」掉、「失败」不变，读数解释不通（实测隔离环境 3 个 unsupported
        # 里 2 个是 A、1 个是 B）。
        #
        # `excluded`（未纳入索引）来自遍历时计数：命中排除规则 / 超出 max_depth /
        # 不在 index_dirs 内。这类文件**连池行都没有**，此前四个观测面全静默。
        no_text, not_indexable = 0, 0
        unsupported_details: list[dict] = []
        try:
            types = list(self.indexer.indexable_types)
            if types:
                ph = ",".join("?" * len(types))
                row = self.db.conn.execute(
                    f"SELECT SUM(CASE WHEN file_type IN ({ph}) THEN 1 ELSE 0 END),"
                    f"       SUM(CASE WHEN file_type IN ({ph}) THEN 0 ELSE 1 END)"
                    "  FROM files WHERE vectorize_status='unsupported'",
                    (*types, *types),
                ).fetchone()
                # 2026-09-23（第二版）：连**明细**一起给，供「查看未纳入索引文件」展开。
                # 此前只有计数没有「是哪些」，用户看到「未纳入索引 3」仍然查不下去。
                # reason 直接算好（noText / type），前端只做文案映射，不再自行判类型。
                unsupported_details = [
                    {"path": display_path(r["path"]),
                     "fileType": r["file_type"],
                     "reason": "noText" if r["file_type"] in set(types) else "type"}
                    for r in self.db.conn.execute(
                        f"SELECT path, file_type FROM files"
                        f" WHERE vectorize_status='unsupported'"
                        f"   AND path IS NOT NULL"
                        f" ORDER BY CASE WHEN file_type IN ({ph}) THEN 0 ELSE 1 END,"
                        f"          path LIMIT ?",
                        (*types, _EXCLUDED_DETAILS_CAP),
                    ).fetchall()
                    if r["path"] and os.path.exists(r["path"])
                ]
            else:
                row = self.db.conn.execute(
                    "SELECT 0, COUNT(*) FROM files WHERE vectorize_status='unsupported'"
                ).fetchone()
            if row:
                no_text = int(row[0] or 0)
                not_indexable = int(row[1] or 0)
        except Exception:  # noqa: BLE001
            logger.warning("统计 unsupported 细分失败", exc_info=True)
        truncations = [dict(t, path=display_path(t["path"]))
                       for t in self.db.truncated_files()
                       if not t.get("path") or os.path.exists(t["path"])]
        # 2026-09-23（第二版）：把「未纳入索引」收成一个**独立小节**（notIndexedPhase）。
        #
        # 为什么单列一节，而不是把明细塞进 scanPhase / vectorizePhase：
        # 用户看到的「未纳入索引」是**一个概念**，但它在后端有两个来源 ——
        #   · 遍历侧（scanner）：命中排除规则 / 超出索引深度 → 连池行都没有；
        #   · 库侧（files 表）：vectorize_status='unsupported' → 有行但没向量，
        #     再分「格式没勾」与「解析了没文本」。
        # 计数此前由前端把两处相加；明细若各回各的，前端就得自己拼、自己判原因，
        # 正是这次重构要清掉的「两条路径各写一份」的老毛病。这里合并成一个出口：
        # `total` 为权威计数，`details` 为明细样本，两者不等时 `detailsTruncated=true`。
        # ★ 2026-10-09：`total` 必须含 `no_text`。此前写 `excluded + not_indexable`，
        # 漏了库侧那一半的「解析了但没文本」——而**明细**是三者相加
        # （excluded_details + unsupported_details，后者含 noText 与 type），于是
        # 出现「格上数字 < 展开条数」。用户实测（取消勾选 md 后）：格上 2、明细 3，
        # 差值恰好是那 1 个 noText（image_only_pg88.docx）；本仓测试
        # `test_not_indexed_total_matches_details_when_small` 的契约也是
        # 「total 等于明细条数」，只是它用排除规则造样本、只盖了 excluded 分支，
        # 没盖到 noText，才漏网。三者取和才与 byKind（noText+type+rule+depth）
        # 及明细口径一致。
        by_kind = dict(excluded_kinds)
        for d in unsupported_details:
            by_kind[d["reason"]] = by_kind.get(d["reason"], 0) + 1
        not_indexed_total = excluded + no_text + not_indexable
        not_indexed_details = list(excluded_details) + list(unsupported_details)
        not_indexed_truncated = bool(
            excluded_overflow or len(not_indexed_details) < not_indexed_total)
        # ── 2026-09-28：「失败文件」明细改成**按 DB 求值**，与计数同源 ──
        #
        # 实测（早期版本）：状态胶囊写「已完成（4 个失败）」，
        # 而「失败文件」格是 0、明细点开为空。根因是**两条不同源的路**：
        #   · 计数 vectorizePhase.failed  ← 按 DB 求值（持久）
        #   · 明细 scanPhase.errorDetails ← 内存流水，只有遍历/embed 抛异常才写，
        #     且每轮 start() 清空
        # 而解析失败（PDF 损坏、xlsx 缺 [Content_Types].xml）走的是 index_file
        # **正常返回** status=failed —— 从不写那条内存流水。于是计数 4、明细 0 条。
        # 改用与 truncated_files 同一范式（那次也是「持久事实不该塞进本轮流水」）。
        #
        # 与内存流水的合并：DB 明细打头（持久、与计数同源），内存里**有路径且不在
        # DB 里**的条目补其后（它们多是「根本没进库」的失败，例如文件名不是 UTF-8
        # —— 连行都没有，DB 永远数不到，不能让它们从界面消失）。去重按 path，
        # 同一条两处都有时以 DB 那条为准（它带持久原因）。
        failed_persistent = self.db.failed_files()
        # 去重键用**出口形态**（display_path）：details 里的 path 已经安全化过，
        # 用原路径比会漏判（含 surrogate 的路径两处形态不同）。
        _seen = {display_path(r["path"]) for r in failed_persistent}
        # 分类以**内存流水**为准（它由记录方判定，比从文本猜可靠）：embed 掉线这类
        # 只有记录方知道是服务侧问题，DB 里那条 reason 只是一段文字。库里没有对应
        # 内存条目时才退回按 reason 前缀判（indexer 落 embed 原因时带 "embed:" 前缀）。
        _mem = {d.get("path"): d for d in details}
        failed_details = []
        for r in failed_persistent:
            # 与 errorDetails 原口径一致：磁盘上已不存在的路径不展示
            # （文件删掉/移走后，明细不该还挂着它）
            if not r["path"] or not os.path.exists(r["path"]):
                continue
            out_path = display_path(r["path"])
            mem = _mem.get(out_path) or {}
            reason = r["fail_reason"] or mem.get("reason") or ""
            # kind：**内存流水优先**（记录方按异常类型判定，比从文本猜可靠）；
            # 库里没有对应内存条目时按 reason 前缀兜底。
            # ⚠️ 为什么不能反过来「有 DB 行就一律按前缀判」：watcher 增量失败**只写
            # 内存流水、不落 DB 原因**（见 watcher._mark_failed，它连库都不碰），
            # 于是 DB 那条的 reason 是空串 → 前缀判成 "parse" → 把 embed 掉线错标成
            # 解析失败（撤改动时测试逮到了：`应标 embed：['parse']`）。
            # reason 同样「DB 优先、内存补空」，否则内存那侧的原文会整句丢失。
            if reason.startswith("embed:"):
                by_prefix = "embed"
            else:
                # ocrFailed: / ocr: / parseFailed: / 裸文本 一律归解析侧
                by_prefix = "parse"
            failed_details.append({
                "kind": mem.get("kind") or by_prefix,
                "path": out_path,
                "reason": reason,
            })
        # 内存流水里「库里没有行」的失败（例如文件名不是 UTF-8：连行都建不出来）
        failed_details += [d for d in details if d.get("path") not in _seen]

        return {
            "status": status,
            "scanPhase": {
                "walked": walked,
                "total": total,
                "errors": errors,
                # Q2-3：错误明细（kind：permission / embed / parse / other）
                #
                # 2026-09-28：**失败明细**（paths 以失败终态收场的那批）与遍历/embed
                # 错误合并输出，DB 侧在前（见上方 failed_details 的说明）。
                # 契约不变：列表元素仍是 {kind, path, reason}，只是数据源扩了一个
                # ——「本轮流水 ∪ 库里失败行」。因此它现在与 vectorizePhase.failed
                # **条数可以不等**（流水里可能有没进库的、库里可能有本轮没重试的），
                # 前端「失败文件」格已按本列表长度取值，不再与 failed 计数绑定。
                "errorDetails": failed_details,
                # 2026-09-23：未纳入索引的文件数（排除规则 / 超出深度 / 不在知识库路径内）。
                # 与 `errors` 的区别：错误是「试了没成」，这里是「从来没试」。
                "excluded": excluded,
                # 2026-09-23：明细样本，供界面「查看未纳入索引文件」展开。
                # 数（excluded）是**全量真实值**；明细有上限，超限时
                # excludedDetailsTruncated=true，界面须如实说明（数 > 明细条数）。
                "excludedDetails": excluded_details,
                "excludedDetailsTruncated": excluded_overflow,
                # 按原因细分（rule = 命中排除规则；depth = 超出索引深度），
                # 供面板给出一句汇总，用户不用逐条看。
                "excludedKinds": excluded_kinds,
            },

            # 2026-09-23（第二版）：界面「未纳入索引」格的**唯一**数据源。
            # total 与 details 同源同长（detailsTruncated 时按说明少列几行）。
            # ★ 2026-10-09：total 是**三者之和**（excluded + noText + type），
            # 与 byKind 及各明细同口径；此前漏了 noText，见上方说明。
            "notIndexedPhase": {
                "total": not_indexed_total,
                "details": not_indexed_details,
                "detailsTruncated": not_indexed_truncated,
                # 原因细分计数：rule（排除规则）/ depth（超深度）/ type（格式没勾）/
                # noText（读不出文本）。前两项由遍历侧给，后两项由库侧给。
                "byKind": by_kind,
            },
            "vectorizePhase": {
                # 2026-09-21：计数补全。此前只回 queued/completed/failed 三个键，留下两个缺口：
                #   ① `vectorizing`（正在算的那个）不在任何一格 → 三数之和 != 文件数；
                #   ② `unsupported`（无文本/纯图片 PDF）完全不出现 → 有这类文件的库进度条
                #      永远到不了 100%。
                # 补上 total 之后，上游可以只用这一组数算进度，与进度文字同源同长。
                # total = files 表总行数（按 vectorize_status 全表分组求和，含未知状态），
                # **不是**扫描轮次的概念——所以它天然把遍历之后新增的文件也算进去。
                "total": sum(counts.values()),
                "queued": counts.get("queued", 0),
                "vectorizing": counts.get("vectorizing", 0),
                # 2026-09-23（重构）：**待向量化 = 池中未算完的行数**。
                # 此前前端要自己拼 `queued + vectorizing + inflight`（那个 inflight 是
                # 进程内计数，两条路径共用入口 → 同一个文件被数两遍，实测 8 个文件的库
                # 算出 9）。池子化后 watcher 也写池行，inflight 不再需要，此键即精确值。
                "pending": counts.get("queued", 0) + counts.get("vectorizing", 0),
                "completed": counts.get("completed", 0),
                "failed": counts.get("failed", 0),
                "unsupported": counts.get("unsupported", 0),
                # unsupported 细分（见上）：无文本 vs 类型不在索引格式内
                "unsupportedNoText": no_text,
                "unsupportedType": not_indexable,
                # 2026-09-21：被块数上限截断（非失败）。计数与明细同源同长，
                # 不会出现「计数 N 条、展开 M 条」。
                "truncated": len(truncations),
                "truncationDetails": truncations,
                # 2026-09-22 新增、2026-09-23 语义收窄：正在索引、**尚未登记进池**的文件数。
                # 池子化后这个窗口只剩「消费者刚从池中取走、index_file 还没写完」——
                # 界面上是 0 或 1，不再用它做「有没有在算」的主判据（改用 pending > 0）。
                "inflight": self.indexer.inflight_indexes,
            },
        }

    # ---------- 控制 ----------
    def _recover_interrupted(self) -> int:
        """启动/重启扫描时：把中断残留的 vectorizing 复位为 queued 重试。

        2026-09-03 bug：进程在置 'vectorizing' 之后被杀（服务重启/embed 超时）会留下
        vectorizing 悬挂——该文件既不重试也不修正，UI 计数错乱。

        2026-09-23（重构）：实现搬到 `pool.IndexPool.recover_interrupted`
        （池子化后「悬挂的行」是池子自己的事）。此处保留同名方法作为转发，
        避免外部调用点失配。
        """
        return self.pool.recover_interrupted()

    def start(self):
        """启动扫描（自动触发；若已在扫描则忽略）。"""
        with self._lock:
            if self._status in (STATUS_SCANNING,):
                return {"status": self._status, "message": "扫描已在运行"}
            self._status = STATUS_SCANNING
            self._walked = 0
            self._walk_errors = 0
            # 2026-09-09：清空上一轮错误明细——progress() 的 errorDetails 语义是
            # "本次扫描"的错误；不清则新扫描跑完后仍展示上一轮失败项（实测：
            # 失败文件已归 0，展开明细还是旧的 embed 连接失败）。
            self._error_logs = []
            self._excluded_count = 0  # 同理：未纳入索引是本轮读数
            self._excluded_details = []
            self._excluded_overflow = False
            self._excluded_kinds = {}
            # 2026-10-10：本轮遍历会自己重填这份读数 → 清掉挂起的重算标记，
            # 免得遍历刚填完、池子静默时又白走一遍全量重算。
            self._excluded_dirty = False
            self._excluded_recount_at = time.monotonic()
            self._pause_event.set()
        self._recover_interrupted()
        self.state.set("scan_status", STATUS_SCANNING)
        # 顺序要紧：**先起遍历线程，再起消费者**。
        # 反过来会有一个真实竞态：消费者一起来就发现池子空（遍历还没开始放活），
        # 而 `_traverse_thread` 此刻还是上一轮的僵尸线程（is_alive()=False）→
        # 直接把状态落成 done。之后遍历才慢悠悠入池，界面先显示「已完成」再变
        # 「扫描中」，测试里表现为「等 done 通过、但计数还没上去」（实测 flaky：
        # test_scan_full_flow / test_scan_http_api 交替失败，assert 2 >= 3 / 0 >= 1）。
        self._traverse_thread = threading.Thread(target=self._run_traverse, daemon=True)
        self._traverse_thread.start()
        # 消费者常驻（重构）：不等遍历结束——遍历边放它边算，池空也不退出（等唤醒）。
        self.worker.start()
        return {"status": STATUS_SCANNING, "message": "扫描已启动"}

    def restart(self):
        """强制重启扫描（2026-08-26 settings 运行时改目录用）：即使正在跑也重新遍历。

        与原"直接改 _status=idle 再 start"等价，但封装为公开方法（不暴露内部状态机）。
        """
        with self._lock:
            self._status = STATUS_SCANNING
            self._walked = 0
            self._walk_errors = 0
            # 2026-09-09：同 start()——新一轮扫描从干净的错误明细开始
            self._error_logs = []
            self._excluded_count = 0
            self._excluded_details = []
            self._excluded_overflow = False
            self._excluded_kinds = {}
            # 2026-10-10：本轮遍历会自己重填这份读数 → 清掉挂起的重算标记，
            # 免得遍历刚填完、池子静默时又白走一遍全量重算。
            self._excluded_dirty = False
            self._excluded_recount_at = time.monotonic()
            self._pause_event.set()
        self._recover_interrupted()
        self.state.set("scan_status", STATUS_SCANNING)
        # 顺序同 start()：先遍历、后消费者（避免「池空 + 僵尸遍历线程」误判 done）
        self._traverse_thread = threading.Thread(target=self._run_traverse, daemon=True)
        self._traverse_thread.start()
        self.worker.start()
        return {"status": STATUS_SCANNING, "message": "扫描已重启"}

    def pause(self):
        with self._lock:
            if self._status != STATUS_SCANNING:
                return {"status": self._status, "message": "当前不在扫描中"}
            self._status = STATUS_PAUSED
            self._pause_event.clear()  # 遍历/消费者在检查点停下
        self.state.set("scan_status", STATUS_PAUSED)
        return {"status": STATUS_PAUSED, "message": "扫描已暂停"}

    def resume(self):
        with self._lock:
            if self._status != STATUS_PAUSED:
                return {"status": self._status, "message": "当前未暂停"}
            self._status = STATUS_SCANNING
            self._pause_event.set()
        self.state.set("scan_status", STATUS_SCANNING)
        # 消费者常驻，恢复时只需唤醒（暂停期间它仍在轮询 _pause_event）
        self.worker.start()
        self.pool.wake()
        return {"status": STATUS_SCANNING, "message": "扫描已继续"}

    def stop(self):
        """彻底停下（关闭服务时调用）：消费者退出，池中未消费的行留给下次启动。"""
        with self._lock:
            self._status = STATUS_IDLE
            self._pause_event.set()
        self.worker.stop()
        self.state.set("scan_status", STATUS_IDLE)

    # ---------- 遍历阶段 ----------
    def _run_traverse(self):
        try:
            # 预扫统计总数（快：仅遍历目录名）
            self._walk_total = self._count_files()
            logger.info("全盘扫描：共 %d 个文件待处理", self._walk_total)

            for base in self.index_dirs:
                if not Path(base).is_dir():
                    logger.warning("索引目录不存在，跳过: %s", base)
                    continue
                self._walk_dir(base)

            # 遍历完成 → 检测磁盘删除（清理索引中的幽灵文件）
            self._prune_deleted()

            with self._lock:
                if self._status == STATUS_SCANNING:
                    self.state.set("scan_status", STATUS_SCANNING)  # 遍历完成但池未排空，仍 scanning
                    logger.info("遍历阶段完成，索引池继续消费")
            # 消费者早已常驻（start() 里就起来了）——这里只需唤醒它，
            # 因为遍历可能是在池空之后才放完最后一批活。
            self.pool.wake()
        except Exception as e:
            logger.exception("遍历异常: %s", e)
            with self._lock:
                self._status = STATUS_IDLE

    def _explain_walk_error(self, path: str, exc: Exception) -> str:
        """遍历失败的对外说明：**给分类，不给成句文案**。

        `sqlite3`/JSON 都要求合法 UTF-8，含 surrogate 的路径会在 `_enqueue_file` 里抛
        UnicodeEncodeError。这类原因给成结构化的 `filenameUnencodable` 标记（UI 按语言
        出词），其余异常给原始消息串（服务侧技术细节，无法翻译）。
        """
        if isinstance(exc, UnicodeEncodeError):
            return self._unencodable_marker(path)
        return str(exc)

    def _unencodable_marker(self, path: str) -> str:
        """文件名非 UTF-8 的**结构化**原因标记（JSON 串，UI 解析后按语言出词）。

        为什么序列化成字符串而不是 dict：`errorDetails[].reason` 在契约里是字符串
        （见 docs/scan-status-contract.md），改动它的类型会波及既有消费端与测试。
        故在 reason 里带一个带标记的 JSON 对象，界面按 `kind` 字段识别后再取词。
        """
        import json as _json

        info = describe_unencodable_filename(path)
        return "filenameUnencodable:" + _json.dumps(info, ensure_ascii=False)

    def _record_error(self, kind: str, path: str, reason: str):
        """记录扫描错误明细（Q2-2 C：后端日志 + 汇总文件 data/scan_errors.log）。

        2026-09-09：同一路径按 path 去重——只保留最新一条（原实现每轮追加，
        同一文件反复失败时 failed=1 而明细堆积成 4 条，用户实测）。汇总文件
        仍逐次追加（它是审计流水，不是展示数据）。

        2026-09-21：**内存里的明细保留原路径，只有落盘的副本安全化**。
          · 内存侧保留原路径有两个用处：progress() 判 `os.path.exists` 需要它（含 surrogate
            的字符串能碰文件系统），以及去重键与遍历时用的 path 严格一致。
          · 落盘必须安全化：`open(..., encoding="utf-8")` 对含 surrogate 的路径同样抛
            UnicodeEncodeError，而旧代码把它包在 `except Exception: pass` 里 —— **这类失败
            在审计文件里完全查不到**（实测：cuda 下两个 GBK 文件名导致的失败，scan_errors.log
            里 0 条，只能靠 journal 定位）。故改成：写前 display_path + errors="replace"，
            并保留一个窄兜底（磁盘不可写等），失败时**至少留一行日志**而不是静默。
        """
        self._error_logs = [d for d in self._error_logs if d.get("path") != path]
        self._error_logs.append({"kind": kind, "path": path, "reason": reason})
        # 追加到汇总文件（路径与 db 同级：data/scan_errors.log）
        try:
            log_path = Path(self.db.db_path).parent / "scan_errors.log"
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{kind}] "
                        f"{display_path(path)} :: {display_path(reason)}\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("写错误明细日志失败（%s）: %s", log_path, e)

    def record_error(self, kind: str, path: str, reason: str):
        """公开入口：供 watcher 增量路径记录错误明细（2026-09-09）。

        增量索引失败此前只在 watcher 里 logger.warning 吞掉——文件不入库、不计
        失败数、明细也没有，与全盘扫描行为不一致（用户实测：embed 掉线时新建
        文件失败数为 0）。
        """
        with self._lock:
            self._record_error(kind, path, reason)

    def _note_failure(self, row, kind: str, reason: str):
        """（2026-09-18 起不再用于计数）失败留痕已下沉到 Indexer.mark_failed。

        保留此方法仅为兼容：真正需要「落 content_hash + 计数」的是**所有**调用
        index_file 的路径，而 index_file 才是公共入口。此处若也计数，同一失败会被
        记两次（attempt_count 每轮 +2，重试预算提前耗尽）。

        本方法现在只记错误明细，供 UI 失败列表展示。
        """
        self._record_error(kind, row["path"], reason)

    def _count_files(self) -> int:
        n = 0
        for base in self.index_dirs:
            for root, dirs, files in self._walk(base):
                # 2026-09-18：与 _walk_dir 同口径——按完整路径判排除
                dirs[:] = [d for d in dirs if not self._excluded(os.path.join(root, d))]
                n += sum(1 for f in files if not self._excluded(os.path.join(root, f)))
        return n

    def _walk(self, base: str, onerror=None, note_depth: bool = False, depth_collector=None):
        """os.walk 封装：支持索引深度限制（max_depth>0 时剪枝）。

        深度语义（1-based，2026-09-04）：base 目录本身（depth 0）的文件恒索引；
        max_depth=N → 索引 base + 下 N-1 层子目录的文件（文件所在目录相对 base
        的层数 < N），到达 depth>=N 的目录时其自身文件与子树都不索引。
        max_depth=0 → 无限深。例：1=仅 base 内文件（不进入任何子目录）；
        2=base+1 层子目录。

        2026-09-23（第二版）：`note_depth=True` 时把这一层被判出局的真实文件记进
        「未纳入索引」明细（kind="depth"）—— 提示文案一直写着「超出索引深度」也算
        未纳入索引，此前却一条不记，理由给了、账对不上。
        **只记这一层**，不递归：整棵子树都已越界，数一层足以说明「是被深度挡的」；
        递归会把计数放大到与用户理解不符（用户数的是「哪些文件没进来」）。
        默认 False —— `_count_files()` 的预扫只负责给出分母，不该顺带记账。

        2026-10-10：新增 `depth_collector(root, files)` —— 在**清空 files 之前**回调，
        把这一层被判出局的真实文件交给调用方。用于「增量重算」路径
        （`_collect_excluded_files`）走同一套深度判据，而不必重写一份 os.walk。
        `depth_collector` 与 `note_depth` 互斥使用（前者优先），避免同一层记两次。
        """
        if self.max_depth <= 0:
            yield from os.walk(base, onerror=onerror)
            return
        base_depth = base.rstrip("/").count("/")
        for root, dirs, files in os.walk(base, onerror=onerror):
            depth = root.rstrip("/").count("/") - base_depth
            if depth >= self.max_depth:
                # 1-based：该目录已超最深可索引层——自身文件与子树都不索引
                # （只清 dirs 仍会 yield 当前层文件，须把 files 一并清空）
                if depth_collector is not None:
                    depth_collector(root, files)
                elif note_depth:
                    self._note_depth_cut(root, files)
                dirs[:] = []
                files[:] = []
            yield root, dirs, files

    def _note_depth_cut(self, root: str, files) -> None:
        """把「因超出索引深度而被挡下」的**本层**文件记进未纳入索引明细。"""
        for fname in files:
            path = os.path.join(root, fname)
            if self._excluded(path):
                continue  # 同时命中排除规则的，由规则那一路记（避免双计）
            self._note_excluded_file(path, "depth")

    def _walk_dir(self, base: str):
        # 目录权限错误不中断扫描：onerror 跳过并记录日志
        def onerror(err):
            self._walk_errors += 1
            self._record_error("permission", str(err.filename or ""), str(err))
            logger.warning("目录不可访问，跳过: %s (%s)", err.filename, err)

        for root, dirs, files in self._walk(base, onerror=onerror, note_depth=True):
            # 暂停检查点
            self._pause_event.wait()
            if self._status == STATUS_IDLE:  # 外部停止
                return
            # 2026-09-18：排除判定用完整路径（原只用条目名）——路径型规则
            # （如 /kb/secret/*）与父目录名规则才能生效，与 watcher 侧同语义
            kept = []
            for d in dirs:
                dp = os.path.join(root, d)
                if self._excluded(dp):
                    # 2026-09-23（重构）：被剪掉的子树此前**四个观测面全静默**——
                    # 无池行、无计数、无明细、无日志。用户问「我的文件怎么没被索引」
                    # 时这是最常见的答案，却哪儿都查不到。这里只做廉价计数。
                    self._note_excluded_tree(dp)
                else:
                    kept.append(d)
            dirs[:] = kept
            for fname in files:
                path = os.path.join(root, fname)
                if self._excluded(path):
                    self._note_excluded_file(path, "rule")
                    continue
                self._pause_event.wait()
                try:
                    self._enqueue_file(path)
                except Exception as e:
                    self._walk_errors += 1
                    self._record_error(
                        "permission" if isinstance(e, PermissionError) else "other",
                        path, self._explain_walk_error(path, e))
                    logger.warning("遍历失败 %s: %s", path, e)
                self._walked += 1
                if self.io_throttle_ms:
                    import time

                    time.sleep(self.io_throttle_ms / 1000)

    # ---------- 遍历侧排除读数：单一口径（全盘遍历 + 增量重算共用） ----------
    #
    # ★ 2026-10-10：抽出这条公共路径，是为了让「全盘遍历」与「增量重算」用**同一套**
    #   记账逻辑，避免两份实现漂移（本仓反复栽在「两条路径各写一份判据」上）。
    #   全盘遍历：`_walk_dir` 边遍历边调用（走 os.walk + 剪枝）；
    #   增量重算：`recount_excluded()` 用同一批 `_collect_excluded_*` 从头重算一遍。
    def _collect_excluded_files(self, base: str, onerror=None) -> tuple[int, list[dict], dict]:
        """遍历 `base`，返回 (排除计数, 明细样本, 按 kind 计数)。

        判据与 `_walk_dir` 完全一致（同一套 excludes + max_depth 语义），区别只是
        这里**不索引任何文件**，纯粹把「本不该索引的文件」数出来 —— 供遍历时的记账，
        以及增量变更后的重算共用。
        """
        count = 0
        details: list[dict] = []
        kinds: dict[str, int] = {}

        def _bump(path: str, kind: str):
            nonlocal count
            count += 1
            kinds[kind] = kinds.get(kind, 0) + 1
            if len(details) < _EXCLUDED_DETAILS_CAP:
                details.append({"path": display_path(path), "kind": kind})

        def _collect_depth(root: str, files):
            # 与 `_note_depth_cut` 同判据：同时命中排除规则的由规则那一路记，避免双计
            for fname in files:
                path = os.path.join(root, fname)
                if not self._excluded(path):
                    _bump(path, "depth")

        for root, dirs, files in self._walk(base, onerror=onerror, note_depth=False,
                                            depth_collector=_collect_depth):
            kept = []
            for d in dirs:
                dp = os.path.join(root, d)
                if self._excluded(dp):
                    # 子树：与 `_note_excluded_tree` 同口径——整棵子树里的真实文件全数
                    sub_count, sub_details, sub_kinds = self._scan_excluded_subtree(dp, "rule")
                    count += sub_count
                    for k, v in sub_kinds.items():
                        kinds[k] = kinds.get(k, 0) + v
                    room = _EXCLUDED_DETAILS_CAP - len(details)
                    details.extend(sub_details[:max(0, room)])
                else:
                    kept.append(d)
            dirs[:] = kept
            # 当前层文件（深度剪枝层由 depth_collector 负责，这里的 files 在剪枝层已被清空）
            for fname in files:
                path = os.path.join(root, fname)
                if self._excluded(path):
                    _bump(path, "rule")
        return count, details, kinds

    def _scan_excluded_subtree(self, dir_path: str, kind: str) -> tuple[int, list[dict], dict]:
        """被剪掉的子树：数清里面全部真实文件（显式栈，不跟符号链接）。

        与 `_note_excluded_tree` 同一遍历，抽出来供「即时记账」与「增量重算」共用。
        """
        collected: list[dict] = []
        count = 0
        stack = [dir_path]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    entries = list(it)
            except OSError:
                continue
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                        continue
                    if not e.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                count += 1
                if len(collected) < _EXCLUDED_DETAILS_CAP:
                    collected.append({"path": display_path(os.path.join(d, e.name)), "kind": kind})
        return count, collected, ({kind: count} if count else {})

    def _enqueue_file(self, path: str):
        """把一个文件交给索引池（重构后：不再自己写 queued，也不自己索引）。

        原先这里有约 50 行「按现有状态分支」的判定（unsupported 重启用 / hash 相同跳过 /
        失败达上限不重排 / 类型取消勾选回收）——全部搬进 `pool.IndexPool.offer()`，
        因为 watcher 增量路径需要**同一套**判定，两份实现必然漂移
        （这正是「两条路径状态机不对称」的由来）。
        """
        return self.pool.offer(path)

    def _prune_deleted(self):
        """遍历完成后：清理索引中已失效的文件（删除向量+元数据+关联记录）。

        清理条件（2026-08-18 修复：知识库路径调整后旧目录文件残留）：
        ① 磁盘文件已不存在（真正删除）→ 清理；
        ② 文件仍在磁盘但**路径不在当前 index_dirs 范围内**（设置页改知识库路径移除了
           该目录）→ 清理。此前只判断 is_file()，目录被移出索引范围但文件还在时
           is_file() 为 True → 旧记录残留、仍可被检索到（用户反馈的 bug）。
        """
        rows = self.db.conn.execute("SELECT file_id, path FROM files").fetchall()
        for r in rows:
            p = r["path"]
            if not Path(p).is_file() or not self._in_index_dirs(p):
                logger.info("清理索引（删除或移出索引范围）: %s", p)
                self.pool.purge(p)

    def _in_index_dirs(self, path: str) -> bool:
        """路径是否在当前 index_dirs 范围内（含递归子目录）。"""
        p = Path(path)
        try:
            pres = p.resolve()
        except OSError:
            return False
        for base in self.index_dirs:
            try:
                pres.relative_to(Path(base).resolve())
                return True
            except (ValueError, OSError):
                continue
        return False

    def _excluded(self, name: str) -> bool:
        # 判据收敛到 excludes.is_excluded（与索引回收共用，避免两处语义漂移）
        return is_excluded(name, self.excludes)

    # ---------- 向量化阶段 ----------
    #
    # 消费者（`_run_worker`）已整体搬进 `pool.IndexWorker`（重构 P1）。
    # 搬走的理由：watcher 增量路径需要的**是同一套消费语义**，而旧实现的
    # 出口是 `return`（线程结束），池空即停机 —— 全盘扫描跑完后新入池的行
    # 无人消费（实测 8 秒无人处理）。常驻 + 唤醒的实现在 pool.py。

    def mark_idle_if_drained(self):
        """池空时把扫描状态落成 done（消费者调用，重构后由 worker 触发）。

        判据（设计文档 §7.6）：
            池中 queued == 0 ∧ vectorizing == 0 ∧ 遍历线程已结束
        第三条不可省。消费者在 `start()` 里就常驻了，**早于遍历线程**开始工作：
        若只看池子，遍历还没开跑时池子是空的 → 会把状态直接落成 done，
        于是遍历期间 `pause()` 报「当前不在扫描中」、界面也早早显示已完成
        （实测：30 个文件的扫描用例，pause 拿到的是 done）。

        `progress()` 出口还会反向派生一次（status 停在 done 而池中又有活 → scanning），
        两处互补：这里负责「忙完收尾」，出口负责「又来了新活」。
        """
        if self.pool.has_work():
            return
        with self._lock:
            if self._traverse_thread is not None and self._traverse_thread.is_alive():
                return
            if self._status == STATUS_SCANNING:
                self._status = STATUS_DONE
                self.state.set("scan_status", STATUS_DONE)
                logger.info("索引池已排空，扫描完成")

    # ---------- 未纳入索引 ----------
    def _note_excluded(self, n: int):
        """记录「未纳入索引」文件数（命中排除规则 / 超出深度）。

        为什么单列一格：这类文件**不进池、没有行、不进错误明细**，
        此前在四个观测面全静默。实测隔离环境里 6 个磁盘文件 → 库 5 行 → 三格只数到 2，
        用户最常问的「我的文件怎么没被索引」就卡在这个静默上。
        """
        with self._lock:
            self._excluded_count += n

    # ---------- 增量变更后的重算（2026-10-10） ----------
    def request_excluded_refresh(self):
        """打一个「遍历侧排除读数需要重算」的标记（watcher 收到目录/文件增删移时调）。

        为什么要有这一层：被排除的目录**本身不注册 inotify watch**（`scanner._walk` 会
        把命中规则的子树整棵剪掉），所以「文件从被排除目录里移出来」这件事 watcher
        **看不到原目录的事件**，只能靠「目标目录有新文件进来」这一侧推断。
        两处都会调本方法：目标目录的文件 create/move-in 事件，以及目录级 create/move。

        只打标记、不立即重算：事件常常成批到来（`mv` 一个目录会连打几十个事件），
        逐个重算等于把目录树走几十遍。真正的重算落在 `recount_excluded()`，由索引池
        在**池子静默**时执行（与 OCR worker 的空闲释放同一个收尾点）。
        """
        with self._lock:
            self._excluded_dirty = True

    def recount_excluded(self, *, force: bool = False, throttle_secs: float = 2.0) -> bool:
        """重算遍历侧排除读数（计数 + 明细），返回是否真的重算了。

        判据（三条都要过）：
          · `_excluded_dirty` 为真（有变更）或 `force=True`（全盘遍历用）；
          · 距上次重算 >= `throttle_secs`（默认 2s，压掉事件风暴里的反复全走）；
          · 当前不在扫描中（全盘遍历自己会填这份读数，别和它抢）。

        ★ 为什么可以整体重算而不是「增量加减」：被排除目录不注册 watch，从它**移出**
          的事件天然收不到 —— 想靠加减法维护就得先能看见「减少」那一侧，做不到。
          重算的代价实测可控（`~/test` 12 文件 0.002s；2.7 万文件约 3.8s），且被节流
          到「一次变更风暴最多走一遍」，比分文件维护更简单也更不容易漂移。
        """
        now = time.monotonic()
        with self._lock:
            if self._status == STATUS_SCANNING:
                return False          # 全盘遍历正在填这份读数，别抢
            if not force and not self._excluded_dirty:
                return False
            if now - self._excluded_recount_at < throttle_secs:
                return False          # 节流：稍后还有一次机会（标记仍在）
            self._excluded_recount_at = now
            self._excluded_dirty = False

        count = 0
        details: list[dict] = []
        kinds: dict[str, int] = {}
        overflow = False
        for base in self.index_dirs:
            if not Path(base).is_dir():
                continue
            try:
                c, d, k = self._collect_excluded_files(base)
            except Exception:                 # noqa: BLE001 —— 重算失败不该影响主流程
                logger.warning("排除读数重算失败: %s", base, exc_info=True)
                continue
            count += c
            for kk, vv in k.items():
                kinds[kk] = kinds.get(kk, 0) + vv
            room = _EXCLUDED_DETAILS_CAP - len(details)
            if len(d) > room:
                overflow = True
            details.extend(d[:max(0, room)])
        if count > _EXCLUDED_DETAILS_CAP:
            overflow = True

        with self._lock:
            self._excluded_count = count
            self._excluded_details = details
            self._excluded_kinds = kinds
            self._excluded_overflow = overflow
        logger.info("未纳入索引读数已重算（增量变更触发）：%d 个文件", count)
        return True

    def _note_excluded_file(self, path: str, kind: str):
        """记一条「未纳入索引」明细（数 + 明细同源，供界面展开）。

        `kind` 只给原因分类（`rule` 命中排除规则 / `depth` 超出索引深度），
        文案由前端按语言给出 —— 与 `errorDetails` 的 `kind` 同一套做法。

        明细是**样本**、有上限；`_excluded_count` 始终是全量真值。超限时把
        `_excluded_overflow` 置真，界面据此如实说明「只列出前 N 条」，
        不会出现「计数 800、展开 200 条」这种对不上的观感。
        """
        with self._lock:
            self._excluded_count += 1
            self._excluded_kinds[kind] = self._excluded_kinds.get(kind, 0) + 1
            if len(self._excluded_details) < _EXCLUDED_DETAILS_CAP:
                self._excluded_details.append(
                    {"path": display_path(path), "kind": kind})
            else:
                self._excluded_overflow = True

    def _note_excluded_tree(self, dir_path: str, kind: str = "rule"):
        """被剪掉的子树：按里面**全部真实文件**计数，并把明细逐条记下。

        2026-09-23：原来只加计数、不记明细，于是「未纳入索引 12」在界面上无法追查
        是哪 12 个。现在把子树里的真实文件逐条记进明细，数 = 明细条数
        （受 `_EXCLUDED_DETAILS_CAP` 限制时置 `_excluded_overflow`）。

        2026-10-09（修）：此前**只 scandir 一层**（旧代码注释自认「已知下界」），
        于是 `excluded/`（含一层子目录 `excluded/picture/`）只数到 7 个，漏掉
        `picture/` 里的 5 个 —— 界面「未纳入索引 7」而磁盘真实 12。用户实测
        （`test/excluded/` 下 12 个文件）暴露这个数。既然 `_walk` 本就会剪掉
        **整棵**子树，这里就补一次显式栈遍历：代价与「不剪枝地遍历这棵子树」相同
        （都是 scandir 每个节点一次），没有额外放大；真机实测 13 个节点 0.038s。
        用显式栈（非递归调用）避免病态深目录爆栈，与 `os.walk` 的做法一致；
        也不跟随符号链接（`follow_symlinks=False`），避免环状链接把遍历拖死。

        内存有界：只保留前 `_EXCLUDED_DETAILS_CAP` 条明细路径，计数不受此限。

        刻意**不**为目录本身加占位明细：那会让「明细条数」比「计数」多 1，界面就会
        重演「格上写 1、展开 2 条」的不一致（F7 同款观感）。
        """
        # ★ 2026-10-10：遍历抽到 `_scan_excluded_subtree`，这里只做「记账」——
        #   与增量重算路径 `_collect_excluded_files` 共用同一份遍历，避免两份实现漂移。
        count, sub_details, kinds = self._scan_excluded_subtree(dir_path, kind)
        if not count:
            # 空目录 / 不可读：没有文件可计，也就不该出现在「未纳入索引文件」里。
            return
        with self._lock:
            self._excluded_count += count
            for k, v in kinds.items():
                self._excluded_kinds[k] = self._excluded_kinds.get(k, 0) + v
            room = _EXCLUDED_DETAILS_CAP - len(self._excluded_details)
            for d in sub_details:
                if room <= 0:
                    self._excluded_overflow = True
                    break
                self._excluded_details.append(d)
                room -= 1
