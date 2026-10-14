"""索引池 + 常驻消费者（重构 P1）。

术语见 `CONTEXT.md`。

**它解决什么**：索引的状态此前分居两处 —— `files 表`（queued/vectorizing/completed）与
`indexer._inflight`（进程内整数）。watcher 增量路径不写库、只动进程内计数，于是

  · 增量索引期间正在算的文件在库中**不存在**，「待向量化」只能靠那个整数兜底；
  · 那个整数挂在两条路径的公共入口上 → scanner 正在算的文件被**数了两遍**
    （实测：库中 vectorizing 与进程内活跃路径指同一文件的帧占 99.2%；
     「待向量化」相对库内真实未算完行数稳定多算 1）；
  · 队列排空后 worker 线程**退出**，新入队的行没人消费（实测 8 秒无人处理）。

**本模块把状态收成一处**：`files 表`就是池子，一个文件在池中有且只有一行。

    生产者（watcher / 全盘遍历）  只 offer（放）与 cancel（撤），不索引任何东西
    消费者（IndexWorker，常驻）   只 take（取活）与 commit（提交/清场）
    执行器（Indexer）             只算，不决定文件状态

三条铁律对应设计文档 §3.2。`files` 表已有的 UNIQUE(path) 与
`upsert_file` 的 ON CONFLICT(path) 是「一行」这个不变式的物理保障。
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("bibit.pool")

# 池行状态（与 files.vectorize_status 取值域一致，不新增状态值）
ST_QUEUED = "queued"
ST_VECTORIZING = "vectorizing"
ST_COMPLETED = "completed"
ST_FAILED = "failed"
ST_UNSUPPORTED = "unsupported"

# offer() 的返回值
OFF_ADDED = "added"          # 新入池
OFF_REFRESHED = "refreshed"  # 已在池中，内容变了 → 重新排队
OFF_NOOP = "noop"            # 已在池中且无需动作（排队中/在算/内容未变）
OFF_SKIPPED = "skipped"      # 达重试上限且内容未变
OFF_UNSUPPORTED = "unsupported"  # 类型不在索引格式内（只落元数据，不入队）
OFF_REMOVED = "removed"      # 类型已取消勾选 → 行被回收
OFF_GONE = "gone"            # 文件已不在磁盘 / 不在索引范围内

# cancel() 的返回值
CAN_CANCELLED = "cancelled"  # 未消费的行，直接撤掉
CAN_DEFERRED = "deferred"    # 正在算，不打断，算完丢弃
CAN_MISSING = "missing"      # 池中本来就没有


class IndexPool:
    """索引池。生产者只放/撤，消费者只取/交。"""

    def __init__(self, db, indexer, max_attempts: int = 3):
        self.db = db
        self.indexer = indexer
        self.max_attempts = int(max_attempts or 0) or 3
        # 错误明细的落点（ScanService）。消费者在**没人注入 scanner** 的构造方式下
        # 也要能把失败写进界面的明细面板 —— 例如单独构造 `Watcher(...)`（测试、
        # 脚本、watcher 自持池的路径）时，消费者是 watcher 建的，拿不到 scanner。
        # 池子持有这个引用，`IndexWorker._record_error` 用它兜底。
        self.scanner_ref = None
        # 取活互斥：保证「SELECT + UPDATE 置 vectorizing」这段不被两个消费者交错。
        # 池子只有一个消费者，但测试/将来的扩展可能起两个，加上更安全。
        self._take_lock = threading.Lock()
        # 唤醒信号：offer/cancel 后 set；消费者带超时 wait（丢信号也有兜底）
        self._wake = threading.Event()
        # 在算期间被取消的路径（算完丢弃，不入库）
        self._cancelled: set[str] = set()
        # 在算期间内容又变了（算完重新排队一次，不丢更新）
        self._dirty: set[str] = set()

    # ================= 生产者侧 =================
    def offer(self, path: str) -> str:
        """把一个文件放进池子（或把内容已变的行重新排队）。

        幂等：一次 `cp` 会为同一路径产生多个事件（CREATE/WRITE/CLOSE_WRITE），
        池中已有该行时不重复入池 —— 这就是「同一文件不被算两遍」的机制，
        取代了原先的进程内单飞（`indexer._active_paths`）。

        必须走 `indexer.indexable_types` 判类型：与 `_enqueue_file` 同口径，
        否则取消勾选格式后的文件会被重新入队（历史 bug）。
        """
        if not path:
            return OFF_GONE
        p = Path(path)
        try:
            if not p.is_file():
                return OFF_GONE
        except OSError:
            return OFF_GONE

        existing = self.db.get_file_by_path(path)
        # 类型判定：元数据取不到时退回库中已有记录的类型，避免把可索引文件误判成不支持
        ftype = None
        try:
            from bibit_filesearch.metadata import extract_metadata

            ftype = extract_metadata(path)["fileType"]
        except Exception:  # noqa: BLE001
            if existing is None:
                return OFF_GONE
            ftype = existing.get("file_type") or ""

        type_ok = ftype in self.indexer.indexable_types

        # ---- 池中无此行 ----
        if existing is None:
            if not type_ok:
                self._mark_unsupported(path)
                return OFF_UNSUPPORTED
            self._enqueue_new(path)
            self.wake()
            return OFF_ADDED

        status = existing["vectorize_status"]

        # ---- 类型不再被支持：回收该行（用户取消勾选格式）----
        if not type_ok:
            if status == ST_UNSUPPORTED:
                return OFF_NOOP
            logger.info("索引回收（类型已取消勾选）: %s", path)
            self.indexer.delete_file(existing["file_id"])
            return OFF_REMOVED

        # ---- 已在排队 / 正在算：不重复入池（单飞）----
        if status in (ST_QUEUED, ST_VECTORIZING):
            if status == ST_VECTORIZING:
                # 在算期间又变了 → 记 dirty，算完再排一次，不丢更新
                if self._content_changed(existing, path):
                    self._dirty.add(path)
            return OFF_NOOP

        # ---- 已算完且内容未变：无动作 ----
        if status == ST_COMPLETED and self._content_hash_same(existing, path):
            return OFF_NOOP

        # ---- 失败行：达上限且内容未变 → 不再重排（防每轮全量重灌）----
        if status == ST_FAILED:
            attempts = int(existing.get("attempt_count") or 0)
            if attempts >= self.max_attempts and self._content_hash_same(existing, path):
                return OFF_SKIPPED
            if attempts >= self.max_attempts:
                self.db.reset_attempt(existing["file_id"])

        # ---- 其余（内容变了 / 曾 unsupported 现已启用 / 曾失败但改了内容）→ 重新排队 ----
        self._requeue(path)
        self.wake()
        return OFF_REFRESHED

    def cancel(self, path: str) -> str:
        """把一个文件从池子里撤掉（文件被删 / 移出知识库路径 / 改名离开）。

        - 未消费（queued）→ 直接删行 + 删块，'cancelled'
        - 正在算（vectorizing）→ **不打断**（解析+embed 是同步的，中断需协作式取消），
          打取消标记，消费者算完后丢弃并清理，'deferred'
        - 终态行 → 交由调用方决定（`purge` 用于真正删除的场景）
        """
        if not path:
            return CAN_MISSING
        row = self.db.get_file_by_path(path)
        if row is None:
            return CAN_MISSING
        status = row["vectorize_status"]
        if status == ST_VECTORIZING:
            with self._take_lock:
                self._cancelled.add(path)
            self.wake()
            logger.info("取消（正在算，不打断，算完丢弃）: %s", path)
            return CAN_DEFERRED
        self.indexer.delete_file(row["file_id"])
        with self._take_lock:
            self._dirty.discard(path)
        return CAN_CANCELLED

    def purge(self, path: str) -> str:
        """撤掉并清场（无论什么状态）。用于「文件真被删除了」。

        与 `cancel` 的差别：`cancel` 对正在算的只打标记（等它自己收尾），
        `purge` 立刻删行 + 删块。删除事件用 `purge` 更彻底 —— 删掉的文件
        不该在库中留下任何痕迹（此前实测会留 completed 幽灵行，watcher
        关闭时不清理，因为清理只在全盘遍历末尾跑一次）。
        """
        row = self.db.get_file_by_path(path)
        if row is None:
            return CAN_MISSING
        if row["vectorize_status"] == ST_VECTORIZING:
            with self._take_lock:
                self._cancelled.add(path)
            self.wake()
            return CAN_DEFERRED
        self.indexer.delete_file(row["file_id"])
        with self._take_lock:
            self._dirty.discard(path)
        return CAN_CANCELLED

    def cancel_tree(self, dir_path: str) -> int:
        """目录被删 / 移出范围：撤掉该前缀下的所有行，返回行数。

        ★ 取行走 `db.files_under_dir`（逐字前缀），不用 SQL `LIKE`——目录名含
        `_`/`%` 时会命中兄弟目录（2026-10-08 实测：`a_b` 连带删掉 `axb`）。
        """
        rows = self.db.files_under_dir(dir_path)
        n = 0
        for r in rows:
            res = self.purge(r["path"])
            if res != CAN_MISSING:
                n += 1
        if n:
            logger.info("目录撤池 %d 条: %s", n, dir_path)
        return n

    # ================= 消费者侧 =================
    def take(self):
        """取一行 queued 并原子置为 vectorizing。无活返回 None。

        SELECT 与 UPDATE 必须在同一临界区内完成（配合 UPDATE 的
        `AND vectorize_status='queued'` 条件），否则两个消费者可能取到同一行。
        """
        with self._take_lock:
            row = self.db.conn.execute(
                "SELECT file_id, path FROM files WHERE vectorize_status=?"
                " ORDER BY rowid LIMIT 1",
                (ST_QUEUED,),
            ).fetchone()
            if row is None:
                return None
            cur = self.db.conn.execute(
                "UPDATE files SET vectorize_status=? WHERE file_id=? AND vectorize_status=?",
                (ST_VECTORIZING, row["file_id"], ST_QUEUED),
            )
            self.db.conn.commit()
            if not cur.rowcount:
                return None  # 竞态：刚被别人取走
            return {"file_id": row["file_id"], "path": row["path"]}

    def commit(self, path: str) -> bool:
        """消费者交活后调用。返回 True 表示结果已入库，False 表示被丢弃。

        两种收尾（对应 `cancel` 的 deferred 语义）：
          · 在算期间被取消 → 删掉 index_file 刚写入的行与块，不入库
          · 在算期间内容又变了 → 重新排队一次（不丢更新）
        """
        with self._take_lock:
            was_cancelled = path in self._cancelled
            if was_cancelled:
                self._cancelled.discard(path)
            was_dirty = path in self._dirty
            if was_dirty:
                self._dirty.discard(path)
        if was_cancelled:
            row = self.db.get_file_by_path(path)
            if row is not None:
                self.indexer.delete_file(row["file_id"])
            logger.info("取消的文件索引完成，结果已丢弃: %s", path)
            return False
        if was_dirty:
            # 内容在算期间变了：这次的结果可能已过期 → 再排一次
            try:
                self._requeue(path)
                self.wake()
            except Exception:  # noqa: BLE001
                logger.warning("重新排队失败: %s", path, exc_info=True)
        return True

    def mark_failed(self, path: str, reason: str) -> None:
        """把一行置为失败并计数（消费者侧的唯一失败落点）。

        2026-09-28：reason **落库**（files.fail_reason），不再只进内存明细。
        此前失败明细只有内存一条路（scan_errors.log + errorDetails），
        而后者每轮清空，于是「状态说 N 个失败、失败文件格 0」自相矛盾。
        """
        row = self.db.get_file_by_path(path)
        if row is None:
            return
        self.db.mark_attempt_failed(row["file_id"], "", reason)
        self.db.set_vectorize_status(row["file_id"], ST_FAILED)

    def set_status(self, path: str, status: str) -> None:
        row = self.db.get_file_by_path(path)
        if row is not None:
            self.db.set_vectorize_status(row["file_id"], status)

    def recover_interrupted(self) -> int:
        """启动/重启时把悬挂的 vectorizing 复位为 queued（进程被杀留下的）。"""
        cur = self.db.conn.execute(
            "UPDATE files SET vectorize_status=? WHERE vectorize_status=?",
            (ST_QUEUED, ST_VECTORIZING),
        )
        self.db.conn.commit()
        if cur.rowcount:
            logger.info("恢复中断残留: %d 个 vectorizing → queued", cur.rowcount)
            self.wake()
        return cur.rowcount

    # ================= 唤醒 =================
    def wake(self) -> None:
        """通知消费者「有活干了」。入池与唤醒是两件事，缺了唤醒池子就是死的。"""
        self._wake.set()

    def wait(self, timeout: float = 2.0) -> bool:
        """消费者等待唤醒；超时返回 False（兜底轮询，丢信号也能恢复）。"""
        got = self._wake.wait(timeout)
        self._wake.clear()
        return got

    def clear_wake(self) -> None:
        self._wake.clear()

    # ================= 计数 =================
    def counts(self) -> dict:
        """按状态分组计数（池子对外口径的唯一来源）。"""
        q = self.db.conn.execute(
            "SELECT vectorize_status, COUNT(*) FROM files GROUP BY vectorize_status"
        ).fetchall()
        return {r[0]: r[1] for r in q}

    def pending_count(self) -> int:
        """池中还没算完的行数（= 「待向量化」格，不再掺进程内计数）。"""
        c = self.counts()
        return int(c.get(ST_QUEUED, 0)) + int(c.get(ST_VECTORIZING, 0))

    def has_work(self) -> bool:
        return self.pending_count() > 0

    # ================= 内部 =================
    def _content_hash_same(self, row: dict, path: str) -> bool:
        h = row.get("content_hash")
        if not h:
            return False
        try:
            return h == self.indexer._content_hash(Path(path))
        except OSError:
            return False

    def _content_changed(self, row: dict, path: str) -> bool:
        return not self._content_hash_same(row, path)

    def _enqueue_new(self, path: str) -> None:
        from bibit_filesearch.metadata import extract_metadata

        meta = extract_metadata(path)
        self.db.upsert_file(meta, "", "", status=ST_QUEUED)

    def _mark_unsupported(self, path: str) -> None:
        """类型不在索引格式内：只落元数据（与 `indexer._upsert_meta_only` 一致）。"""
        from bibit_filesearch.metadata import extract_metadata

        meta = extract_metadata(path)
        self.db.upsert_file(meta, "", "", status=ST_UNSUPPORTED)

    def _requeue(self, path: str) -> None:
        from bibit_filesearch.metadata import extract_metadata

        try:
            meta = extract_metadata(path)
            size, mtime = meta["fileSize"], meta["modifiedAt"]
        except Exception:  # noqa: BLE001
            size, mtime = None, None
        if size is None:
            self.db.conn.execute(
                "UPDATE files SET vectorize_status=? WHERE path=?", (ST_QUEUED, path)
            )
            self.db.conn.commit()
            return
        self.db.conn.execute(
            "UPDATE files SET vectorize_status=?, size=?, modified_at=? WHERE path=?",
            (ST_QUEUED, size, mtime, path),
        )
        self.db.conn.commit()


class IndexWorker:
    """常驻消费者：取活 → 索引 → 提交。

    与旧 `scanner._run_worker` 的两处关键差别：
      · **池空时不退出**，等唤醒（带超时兜底）。旧实现在池空时 `return`，
        重开的唯一入口是 `start()`/`resume()` —— 于是全盘扫描跑完后新入池的
        行没人消费（实测 8 秒无人处理）。
      · **不自己制造状态**：取活时置 vectorizing，结果由 `index_file` 落终态，
        再由 `pool.commit` 做取消/去重收尾。
    """

    def __init__(self, pool: IndexPool, indexer, scanner=None):
        self.pool = pool
        self.indexer = indexer
        self.scanner = scanner
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.processed = 0

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="index-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.pool.wake()

    def _paused(self) -> bool:
        if self.scanner is None:
            return False
        ev = getattr(self.scanner, "_pause_event", None)
        return bool(ev is not None and not ev.is_set())

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._paused():
                time.sleep(0.2)
                continue
            row = self.pool.take()
            if row is None:
                self._settle()
                self.pool.wait(2.0)  # 唤醒 + 超时兜底
                self._settle()
                continue
            try:
                self._run_one(row)
            except Exception:  # noqa: BLE001 —— 单行异常绝不终止消费者
                logger.exception("索引池消费异常: %s", row.get("path"))
            self.processed += 1

    def _run_one(self, row: dict) -> None:
        path = row["path"]
        if self._paused():
            # 取活后立刻暂停：还原为 queued 等恢复
            self.pool.set_status(path, ST_QUEUED)
            return
        try:
            r = self.indexer.index_file(path)
        except Exception as e:  # noqa: BLE001
            # ★ 2026-10-08（审查 R08）：**异常出口也必须做池收尾**。
            #   原先这里直接 return，`_cancelled` / `_dirty` 标记留在集合里没人消费
            #   —— `commit()` 只在成功路径被调用。后果：取消发生在解析期间、而这次
            #   索引又失败了，标记会跨轮存活；同路径之后重新出现并**成功**索引时，
            #   `commit()` 读到上一轮的陈旧标记，把刚写好的结果删掉（实测：重试成功
            #   后库里记录为 None）。
            #   `commit()` 正是那个「消费标记并执行对应收尾」的方法：被取消 → 删行；
            #   内容又变了 → 重新排队。**注意它的返回值语义**：True = 结果已入库，
            #   False = 被丢弃（被取消），别读反（本改动第一版就读反了）。
            if not self.pool.commit(path):
                logger.info("取消的文件索引失败，结果已丢弃: %s", path)
                return
            self._record_error(path, e)
            # index_file/mark_failed 已尽力落库；确保不是悬挂的 vectorizing
            self._ensure_terminal(path)
            return

        if r.get("status") == "inflight":
            # 单飞兜底：同路径索引没拿到结果（等超时 / 同线程重入）。
            # 本行已被 take() 置成 vectorizing，不还原就会**永久**卡在那个状态
            # ——既不重试也不修正，界面「待向量化」恒 1（F8 的兜底分支，2026-09-22）。
            # `index_file` 内部已有同路径单飞；池子化后两条路径共用一行，
            # 这个分支理论上更难走到，但保留（宁可多还原一次也不留悬挂）。
            self.pool.set_status(path, ST_QUEUED)
            logger.info("同路径索引未取得结果（%s），状态还原为 queued: %s",
                        r.get("reason"), path)
            return
        if r.get("oversized"):
            logger.warning(
                "文件过大，仅索引前 %s 块（共 %s 块）：%s",
                self.indexer.max_chunks_per_file, r.get("total_chunks"), path,
            )
        if r["status"] == "skipped":
            # hash 未变而跳过：可能是中断残留（hash 已写、块为空）——
            # 按「向量在不在」兜底修正，否则那一行会永久卡在 vectorizing
            n = self.db_chunk_count(row["file_id"])
            if n > 0:
                self.pool.set_status(path, ST_COMPLETED)
            else:
                self.pool.set_status(path, ST_UNSUPPORTED)
        self.pool.commit(path)

    def db_chunk_count(self, file_id: str) -> int:
        return self.pool.db.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_id=?", (file_id,)
        ).fetchone()[0]

    def _ensure_terminal(self, path: str) -> None:
        row = self.pool.db.get_file_by_path(path)
        if row is not None and row["vectorize_status"] == ST_VECTORIZING:
            self.pool.set_status(path, ST_FAILED)

    def _record_error(self, path: str, exc: Exception) -> None:
        from bibit_filesearch.model_client import EmbedError

        if isinstance(exc, PermissionError):
            kind = "permission"
        elif isinstance(exc, EmbedError):
            kind = "embed"
        else:
            kind = "parse"
        logger.error("向量化失败 %s: %s", path, exc)
        # 2026-09-23：与 watcher/scanner 同一口径 —— 文件名编不了码给**结构化标记**，
        # 其余给原始异常消息（界面按语言出词，不再打印写死的中文）。
        reason = str(exc)
        if isinstance(exc, UnicodeEncodeError):
            import json as _json

            from bibit_filesearch.pathutil import describe_unencodable_filename

            reason = "filenameUnencodable:" + _json.dumps(
                describe_unencodable_filename(path), ensure_ascii=False)
        # 优先用构造时注入的 scanner；没有就退回池子上挂的引用（watcher 自持池的路径）
        scanner = self.scanner if self.scanner is not None else self.pool.scanner_ref
        if scanner is None:
            return
        try:
            scanner.record_error(kind, path, reason)
        except Exception:  # noqa: BLE001
            logger.warning("记录错误明细失败: %s", path, exc_info=True)

    def _settle(self) -> None:
        """空转时的收尾：空闲释放（每次都做）+ 池空落 done（要有 scanner）。

        ★ 2026-10-10 收敛（方案 A）：**删掉「池空就无条件释放」那条**（原
          `self._release_ocr_workers()`）。它认的触发条件是「索引池排空」，而这不是一个
          正确的释放理由：
            · 它不代表「不会再用了」——下一张图随时可能来（watcher 增量、慢速大目录）。
              实测两张图隔 15s 到达时，它让**每张图都重建一遍**三个 worker（各多付 1.5s）；
            · 它也不代表「内存贵」——与图片大小无关。
          于是运行期释放只剩唯一一条正确理由：**空闲 ≥ `ocr.idle_release_secs`**（下面
          第一句，判据在 `ocr_ep.should_release_when_idle`）。内存真贵时的及时归还要靠
          `ocr.worker_max_rss_mb`（按 RSS 退休，见 ocr_ep）——那是另一条独立触发点。
          代价：孤立单张（含大图）的持有时间从 ~2s 变为最多一个 TTL。

        池空落 done（`mark_idle_if_drained`）仍要 scanner，故守卫在下。
        """
        self._release_ocr_workers_when_idle()
        if self.scanner is None:
            return
        try:
            # 2026-10-10：池子静默时顺带重算「未纳入索引」的遍历侧读数（只有增量
            # 变更加过标记才真的走，且被节流）。放在这里的理由与 OCR 空闲释放相同
            # ——它是「真没活干了」的唯一出口，重算时不会与正在跑的索引抢 IO。
            self.scanner.recount_excluded()
        except Exception:  # noqa: BLE001
            logger.warning("重算未纳入索引读数失败", exc_info=True)
        try:
            self.scanner.mark_idle_if_drained()
        except Exception:  # noqa: BLE001
            logger.warning("落扫描完成状态失败", exc_info=True)

    def _release_ocr_workers_when_idle(self) -> None:
        """会话空闲超过 `ocr.idle_release_secs` 就放掉 worker（幂等，见 ocr_ep 的注释）。

        定价依据（本机实测 3 趟中位数）：释放后第一张 OCR 是 3.56s vs 热态 2.06s，
        净付 **1.50s**（重建 det/rec/cls 各 0.20/0.24/0.17s）。所以只有「与上一张图
        隔了够久」才值得做，这正是 TTL 而不是「每张都放」的原因。
        """
        try:
            from bibit_filesearch import ocr_engine

            n = ocr_engine.release_idle_when_idle()
            if n:
                logger.info("OCR 会话空闲超时，释放 %d 个 worker（省 GB 级常驻）", n)
        except Exception:  # noqa: BLE001
            logger.warning("空闲释放 OCR worker 失败", exc_info=True)
