"""SQLite 存储层。

files 表：文件元数据 + 内容 hash + 摘要 + 向量化状态
chunks 表：分块文本 + 向量 BLOB（512 维 float32 小端）
"""
from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path

# 向量 BLOB 编解码（float32 小端）
def _pack_vec(v: list[float]) -> bytes:
    return struct.pack(f"<{len(v)}f", *v)


def _unpack_vec(b: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(b)//4}f", b))


class Database:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()
        # 跨线程写锁（2026-08-28 合入 runtime 后补：watcher 补扫线程与 scanner
        # worker 线程并发 index 同一文件时，replace_chunks 的 DELETE+INSERT 非原子
        # 会 UNIQUE 冲突——用进程内互斥锁串行化整个替换）
        self._write_lock = threading.Lock()
        # 初始化 schema（用主线程连接）
        self.conn  # noqa: B018 触发创建
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        """线程本地连接（扫描遍历/worker 多线程各持一连接，避免并发 commit 冲突）。

        每个新连接统一设置 busy_timeout=5000（此前仅主线程连接设置，worker/遍历/补扫
        线程默认 busy_timeout=0，并发写偶发 database is locked）与 WAL（幂等，无害）。
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)  # autocommit，多线程可见
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            self._local.conn = conn
        return conn

    def _init_schema(self):
        c = self.conn
        # WAL：多线程读写并发（遍历写 + worker 写）不互相阻塞
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            pass
        c.execute(
            """CREATE TABLE IF NOT EXISTS files (
                file_id   TEXT PRIMARY KEY,
                path      TEXT UNIQUE NOT NULL,
                name      TEXT NOT NULL,
                file_type TEXT NOT NULL,
                size      INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT '',
                modified_at TEXT NOT NULL DEFAULT '',
                accessed_at TEXT NOT NULL DEFAULT '',
                author    TEXT NOT NULL DEFAULT '',
                directory TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                summary   TEXT NOT NULL DEFAULT '',
                pdf_images TEXT NOT NULL DEFAULT '[]',  -- PDF 图生文关联 JSON
                indexed_at TEXT NOT NULL DEFAULT '',
                vectorize_status TEXT NOT NULL DEFAULT 'idle',
                attempt_count INTEGER NOT NULL DEFAULT 0
            )"""
        )
        # 轻量迁移：老库缺列时补列（旧 schema 无 pdf_images）
        cols = [r[1] for r in c.execute("PRAGMA table_info(files)").fetchall()]
        if "pdf_images" not in cols:
            c.execute("ALTER TABLE files ADD COLUMN pdf_images TEXT NOT NULL DEFAULT '[]'")
        if "llm_summary" not in cols:
            c.execute("ALTER TABLE files ADD COLUMN llm_summary TEXT NOT NULL DEFAULT ''")
        # 2026-09-18：重试计数。此前失败文件不计数、也不落 content_hash，扫描每轮
        # 都把全部失败文件当「待重算」重排一次（实测一棵 40MB 反编译 JS 树 = 113 个
        # 文件 / 约 6.8 万个 embed 输入，每轮全量重灌 → embed 服务 OOM → 重启 →
        # 看门狗再触发一轮，正反馈）。计数 + 落 hash 后，幂等快路径对失败文件也能生效。
        if "attempt_count" not in cols:
            c.execute("ALTER TABLE files ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0")
        # 2026-09-21：分块总数（**截断前**的真实块数）。两个用途：
        #   ① 让「被 max_chunks_per_file 截断」这件事持久可见——此前它只存在于
        #      扫描期间的 errorDetails 与 scan_errors.log，扫描一结束就查不到了，
        #      于是「文件只索引了 5.3% 的内容」而界面/f健康接口都报正常。
        #   ② 抬高块数上限后能**精确找出需要重索引的文件**：判据是「已存的块数 <
        #      实际总块数」。没有这一列只能靠「块数恰好等于旧上限 32」猜，会误伤
        #      本来就 32 块的文件。
        #   老库默认 0 = 未知，不会参与重排（见 requeue_truncated 的 WHERE）。
        #   对外读取见 db.truncated_files()（供 /api/scan/progress 的 vectorizePhase）。
        if "total_chunks" not in cols:
            c.execute("ALTER TABLE files ADD COLUMN total_chunks INTEGER NOT NULL DEFAULT 0")
        # 2026-09-28：失败**原因**落库（列名 fail_reason）。
        # ★ 为什么必须持久化：`vectorizePhase.failed` 是**持久**计数（按 DB 求值），
        #   而失败明细此前只有内存里的 `scanPhase.errorDetails`（每轮 start() 清空，
        #   且只有遍历/embed 抛异常才会写）。解析失败（PDF 损坏、xlsx 缺
        #   [Content_Types].xml）从不经过那条路 → 出现「状态胶囊说 4 个失败、
        #   「失败文件」格却是 0、明细点开是空」的自相矛盾（真机实测）。
        #   落库后明细可与计数同源（见 scanner.progress），且跨轮可见。
        #   与 total_chunks 同一范式：都是「持久事实，不该塞进本轮流水」。
        if "fail_reason" not in cols:
            c.execute("ALTER TABLE files ADD COLUMN fail_reason TEXT NOT NULL DEFAULT ''")
        c.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id    TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text       TEXT NOT NULL,
                vector     BLOB,
                UNIQUE(file_id, chunk_index)
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_files_directory ON files(directory)")
        c.commit()

    # ---------- files ----------
    def upsert_file(self, meta: dict, content_hash: str, summary: str, status: str = "completed",
                    pdf_images: str = "[]", total_chunks: int = 0) -> str:
        file_id = meta["fileId"] or _gen_file_id(meta["filePath"])
        self.conn.execute(
            """INSERT INTO files (file_id, path, name, file_type, size, created_at, modified_at,
                                  accessed_at, author, directory, content_hash, summary, pdf_images, indexed_at,
                                  vectorize_status, total_chunks)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                 file_id=excluded.file_id, name=excluded.name, file_type=excluded.file_type,
                 size=excluded.size, created_at=excluded.created_at, modified_at=excluded.modified_at,
                 accessed_at=excluded.accessed_at, author=excluded.author, directory=excluded.directory,
                 content_hash=excluded.content_hash, summary=excluded.summary, pdf_images=excluded.pdf_images,
                 indexed_at=excluded.indexed_at, vectorize_status=excluded.vectorize_status,
                 total_chunks=excluded.total_chunks""",
            (
                file_id, meta["filePath"], meta["fileName"], meta["fileType"], meta["fileSize"],
                meta["createdAt"], meta["modifiedAt"], meta["accessedAt"], meta["author"],
                meta["directory"], content_hash, summary, pdf_images, _now(), status,
                # total_chunks：截断前的真实块数。0 表示无文本/未知（unsupported 走
                # _upsert_meta_only 时即为 0，语义正确——没有文本就没有块）。
                int(total_chunks or 0),
            ),
        )
        self.conn.commit()
        return file_id

    def set_llm_summary(self, file_id: str, llm_summary: str):
        """缓存 LLM 精修摘要（D-1 懒生成）。"""
        self.conn.execute("UPDATE files SET llm_summary=? WHERE file_id=?", (llm_summary, file_id))
        self.conn.commit()

    def get_file(self, file_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM files WHERE file_id=?", (file_id,)).fetchone()
        return dict(r) if r else None

    def get_file_by_path(self, path: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        return dict(r) if r else None

    def delete_file(self, file_id: str):
        self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        self.conn.execute("DELETE FROM files WHERE file_id=?", (file_id,))
        self.conn.commit()

    def set_vectorize_status(self, file_id: str, status: str):
        """改状态。成功落 completed 时顺带清零 attempt_count（成功即重置重试预算）。"""
        if status == "completed":
            # 成功即清零重试预算 + 清失败原因（否则修好的文件还挂着旧原因）
            self.conn.execute(
                "UPDATE files SET vectorize_status=?, attempt_count=0, fail_reason=''"
                " WHERE file_id=?",
                (status, file_id),
            )
        elif status == "failed":
            self.conn.execute("UPDATE files SET vectorize_status=? WHERE file_id=?",
                              (status, file_id))
        else:
            # 离开 failed（requeue / unsupported 等）同样清原因：
            # 原因只描述「上次为何失败」，状态走了它就该走。
            self.conn.execute(
                "UPDATE files SET vectorize_status=?, fail_reason='' WHERE file_id=?",
                (status, file_id),
            )
        self.conn.commit()

    def mark_attempt_failed(self, file_id: str, content_hash: str = "",
                            reason: str = "") -> int:
        """记一次失败：attempt_count += 1，并（可选）落 content_hash；返回新计数。

        2026-09-18：失败时**必须落 hash**。此前失败只改状态、hash 留在空串，
        而 `Indexer.index_file` 的幂等判据是 `existing["content_hash"] == 新hash`——
        空 hash 永不相等 → 每轮扫描都判为「需重算」，把整个失败集重灌一遍。
        落 hash 后：内容没变的失败文件可被识别（配合 attempt_count 达上限即不再重排），
        用户改了文件则 hash 变化，仍能重新排队重试。
        """
        # reason（2026-09-28）：落失败原因，供「失败文件」明细显示。
        # 与 content_hash 同理必须落库——计数是持久的、明细也必须持久，否则同一件事
        # 在界面上有两个数（详见建表处 fail_reason 的说明）。
        # 只存**首见**原因、不覆盖：重试时的异常常是「上游不可达」这类噪声，
        # 覆盖会把最初的真实原因（例如「不是 PDF」）冲掉。
        reason = (reason or "")[:300]
        if content_hash:
            self.conn.execute(
                "UPDATE files SET vectorize_status='failed', attempt_count=attempt_count+1,"
                " content_hash=?,"
                " fail_reason=CASE WHEN fail_reason='' THEN ? ELSE fail_reason END"
                " WHERE file_id=?",
                (content_hash, reason, file_id),
            )
        else:
            self.conn.execute(
                "UPDATE files SET vectorize_status='failed', attempt_count=attempt_count+1,"
                " fail_reason=CASE WHEN fail_reason='' THEN ? ELSE fail_reason END"
                " WHERE file_id=?",
                (reason, file_id),
            )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT attempt_count FROM files WHERE file_id=?", (file_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def reset_attempt(self, file_id: str):
        """清零重试计数（内容变更后允许重新尝试）。"""
        self.conn.execute("UPDATE files SET attempt_count=0 WHERE file_id=?", (file_id,))
        self.conn.commit()

    def all_files(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM files").fetchall()]

    def files_under_dir(self, dir_path: str) -> list[dict]:
        """某目录（含全部后代）下的文件行。用于目录删除 / 移出范围时的回收。

        ★ 前缀匹配用 `substr` **逐字比较**，不走 SQL `LIKE`：`LIKE` 把模式串里的
        `_` 当「任意单字符」、`%` 当「任意长度」，目录名里带这两个字符时会命中
        兄弟目录。实测（2026-10-08，临时库）：`cancel_tree("/kb/a_b")` 把
        `/kb/axb/` 下的文件一并删除；`a%b` 同理。参数化查询不会取消通配符含义，
        必须换判据。

        返回完整行（不只 path）：调用方有的按 file_id 删（watcher），有的按 path
        走池（pool），两者都需要各自那个字段，避免再查一次。
        """
        prefix = str(dir_path).rstrip("/") + "/"
        rows = self.conn.execute(
            "SELECT * FROM files WHERE substr(path, 1, ?) = ?", (len(prefix), prefix)
        ).fetchall()
        return [dict(r) for r in rows]

    def count_files(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def stored_vector_dim(self) -> int | None:
        """存量块向量的维度（float32 → 字节数 // 4）；无向量返回 None。

        2026-09-15：供启动时判断「存量索引维度是否仍与当前 embed 配置一致」。
        取一条即可——同库内维度混杂属数据损坏，由 retriever._ensure_loaded 的
        整除校验兜底。
        """
        row = self.conn.execute(
            "SELECT length(vector) FROM chunks WHERE vector IS NOT NULL LIMIT 1"
        ).fetchone()
        if not row or not row[0]:
            return None
        return int(row[0]) // 4

    def purge_vectors(self) -> int:
        """清空全部块向量，并把已处理的文件重置为待向量化；返回重置的文件数。

        索引是**派生数据**——内容全部来自知识库目录里的文件，清掉只损失重建时间。
        用于 embed 配置变更（换模型/换维度）导致存量向量不可复用时。调用方必须
        记录日志，不得静默执行。

        2026-09-15 补：**必须同时清空 `content_hash`**。否则重扫时 `Indexer.index_file`
        看到 hash 未变会直接 `skipped` 返回，而向量刚被清空 → 该文件永远拿不回向量，
        worker 还会把它的状态改判成 `unsupported`，界面上表现为「索引完成但搜不到」
        （即 docs/troubleshooting.md 现象 G 里那条「重扫报成功但向量未落库」的疑点）。
        """
        with self._write_lock:
            conn = self.conn
            conn.execute("DELETE FROM chunks")
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                "WHERE vectorize_status IN ('completed','failed')"
            )
            conn.commit()
            return cur.rowcount

    def requeue_unsupported(self, file_types) -> int:
        """把「状态 unsupported、但类型在索引格式内」的文件重新排队；返回文件数。

        这类文件是**误判**的产物：类型本来该索引，可解析器当时读不到文本（例：正文
        全在文本框里的 docx），于是落了 unsupported。解析能力升级后要救回来，但**不能
        只改状态** —— `Indexer.index_file` 的 content_hash 幂等会直接 `skipped`，块数仍
        为 0，worker 又把它标回 unsupported，新解析代码一次都跑不到。所以必须同时清
        content_hash，强制真正重解析一遍。

        只收窄到「类型在索引格式内」，避免把 285 个不在格式内的（emf/.shaders 之类）
        无故入队空转。真正的空文件（纯图片 PDF、空 txt）重解析后仍无文本 → 回到
        unsupported，代价只是一次解析，不会再被重复触发（调用方按解析能力版本只跑一次）。
        """
        types = [str(t) for t in file_types if str(t)]
        if not types:
            return 0
        placeholders = ",".join("?" * len(types))
        with self._write_lock:
            conn = self.conn
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                f"WHERE vectorize_status='unsupported' AND file_type IN ({placeholders})",
                types,
            )
            conn.commit()
            return cur.rowcount

    def requeue_ocr_text(self, image_types) -> int:
        """把「库里的文本是旧版 OCR 产出的」文件重新排队；返回文件数（PARSER_REVISION 2）。

        2026-09-24 新增。OCR 方向转正修好后，**已入库**的图片与图片型 PDF 手里还是
        旧文本：`content_hash` 是文件字节的 hash，OCR 结果变了它看不出来 → 重扫判
        `skipped` → 歪图永远停在乱码状态，「修了等于没修」。

        判据两条，都只命中「库里确实有 OCR 文本」的文件：
          · 类型 ∈ image_types —— 图片文件的文本**全部**来自 OCR，一律需要重取；
          · `pdf_images != '[]'` ——那一列**只在**「无文本层、走内嵌图 OCR」
            的分支被填充，故可当「图片型 PDF」的标记。有文本层的 PDF 文本来自
            pymupdf、与 OCR 无关，不该被重排。

        与 `requeue_unsupported` 的分工：那条管「当时 OCR 读不出文字 → 落了
        unsupported」的文件；本条管「当时读出了文字 → 落了 completed/queued」的。
        两条合起来才是这个解析版本的完整影响面，所以 `reconcile_parser_revision`
        两个都调。
        """
        types = [str(t) for t in image_types if str(t)]
        with self._write_lock:
            conn = self.conn
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                "WHERE (file_type IN (%s) OR (file_type='pdf' AND pdf_images != '[]')) "
                "  AND vectorize_status IN ('completed', 'queued', 'failed')"
                % (",".join("?" * len(types)) if types else "''"),
                types,
            )
            conn.commit()
            return cur.rowcount

    def requeue_all_pdfs(self) -> int:
        """把**全部** PDF 重新排队一次；返回文件数（PARSER_REVISION 7）。

        2026-09-29 新增。rev 7 改了 PDF 的解析**策略**（两遍 + 逐页薄页判据 +
        预算早停 + 每文件 OCR 页数上限），同一份 PDF 抽出来的文本与改前不同：
          · 混合型 PDF（封面带字 + 正文扫描）：文本**变多**（以前整份判据下
            510 页扫描页一张都不 OCR）；
          · 超预算的 PDF：文本**变少**（只处理到攒够 64 块为止，后面不再翻）。

        **为什么不能复用 `requeue_ocr_text`**：它只命中 `pdf_images != '[]'` 的
        "图片型 PDF"（那条判据是"无文本层、走了内嵌图 OCR"的标记）。而 rev 7 影响
        最大的一类恰恰是**混合型** PDF —— 它们有文本层、pdf_images 为空，旧判据
        一个都命中不到。既然影响面是"所有 PDF"，判据就直写"所有 PDF"，
        不为省几次重索引而漏掉真正的受影响面（漏掉的代价是永久留着旧文本，
        而 content_hash 看不出来，永远不会自愈）。

        只取 `vectorize_status IN ('completed','queued','failed')`：
        unsupported 的 PDF 由 `requeue_unsupported` 那条路管（它按类型收窄），
        两条职责不重叠。清 content_hash 的理由与 `requeue_unsupported` 相同 ——
        否则 index_file 的 hash 幂等会直接判 skipped，新解析代码一次都跑不到。
        """
        with self._write_lock:
            conn = self.conn
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                "WHERE file_type='pdf' "
                "  AND vectorize_status IN ('completed', 'queued', 'failed')"
            )
            conn.commit()
            return cur.rowcount

    def requeue_all_docx(self) -> int:
        """把**全部** docx 重新排队一次；返回文件数（PARSER_REVISION 9）。

        2026-10-09 新增。rev 9 给 docx 加了「内嵌图 OCR（按段落薄判据）」，同一份 docx
        抽出来的文本与改前不同，而且**变多**：
          · 纯图 docx（文本全在观测记录图里）：改前无文本 → unsupported，改后有文本；
          · 正文 + 段内插图：改前插图里的字从不进索引，改后进了。

        **为什么不能复用 `requeue_unsupported`**：它只收 `vectorize_status='unsupported'`
        的行。而影响面最大的一类恰恰是**正文 + 插图**的 docx —— 它们有正文，落的是
        completed，旧判据一个都命中不到。既然影响面是「所有 docx」，判据就直写「所有
        docx」，与 rev 7 的 `requeue_all_pdfs` 同一取舍（不为省几次重索引漏掉真正的
        受影响面，漏掉的代价是永久留着旧文本，而 content_hash 看不出来、永不自愈）。

        只取 `vectorize_status IN ('completed','queued','failed')`：unsupported 的 docx
        由 `requeue_unsupported` 那条路管（它按类型收窄），两条职责不重叠。
        清 content_hash 的理由与 `requeue_all_pdfs` 相同。
        """
        with self._write_lock:
            conn = self.conn
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                "WHERE file_type='docx' "
                "  AND vectorize_status IN ('completed', 'queued', 'failed')"
            )
            conn.commit()
            return cur.rowcount

    def requeue_truncated(self, legacy_cap: int = 0) -> int:
        """把「被块数上限截断」的文件重新排队；返回文件数。

        2026-09-21 新增，配合 `scan.max_chunks_per_file` 抬高（32→64）。为什么必须
        有这一步：content_hash 是**文件字节**的 hash，截断不改变它。所以调大上限后
        重扫会判「内容未变 → skipped」，那些文件永远停在旧的截断份数上——改配置等于
        没改。这与 0.1.9 的 `purge_vectors`、0.1.10 的 `requeue_unsupported` 是同一
        类陷阱：派生数据的**语义前提**变了，而判据看不出来。

        **主判据（精确）**：`已存块数 < total_chunks`（total_chunks 是截断前的真实
        块数，落库见 db.files.total_chunks）。因此
          · 本来就 50 块的文件：已存 50 == 总 50 → 不动；
          · 被截到 32 的 599 块文件：32 < 599 → 重排，按新上限索引。

        **老库判据（近似，仅当 legacy_cap > 0）**：`total_chunks` 这一列对**升级前**
        入库的文件一律是 0（未知），主判据匹配不到任何行——于是「抬高上限」对老用户
        等于没做，而且水位线会在同一次就被置成已见，机会一去不回。所以补一条：
        `total_chunks = 0 AND 已存块数 >= legacy_cap`。旧上限 32 之下库里不可能有
        >32 块的记录，故「存了 32 块」= 「可能被截断」。
        这条是**近似**（本来就整 32 块的文件会被多索引一次），但漏判的代价（永久
        搜不到那部分内容、界面还显示正常）远大于多算一次；且由水位线保证只跑一次。
        老库一旦被新代码重新索引过，`total_chunks` 就有值了，这条自动失效。
        """
        legacy_cap = int(legacy_cap or 0)
        with self._write_lock:
            conn = self.conn
            count_sql = "(SELECT COUNT(*) FROM chunks WHERE chunks.file_id = files.file_id)"
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                f"WHERE (total_chunks > 0 AND total_chunks > {count_sql}) "
                f"   OR (? > 0 AND total_chunks = 0 AND {count_sql} >= ?)",
                (legacy_cap, legacy_cap),
            )
            conn.commit()
            return cur.rowcount

    # ---------- chunks ----------
    def requeue_short_chunks(self, min_chars: int = 3) -> int:
        """把「含已向量化极短块」的文件重新排队；返回文件数（rev 2，配合 A）。

        为什么必须单独有一条：极短块的向量**已经在库里**，而文件内容的 content_hash
        没变 → 重扫直接 `skipped`，旧向量永远活着（== A 对存量不生效）。这与
        `requeue_truncated` / `requeue_unsupported` 是同一类陷阱：派生数据的语义前提
        变了，而幂等判据看不出来。

        判据取 `length(trim(text)) < min_chars AND vector IS NOT NULL` —— 精确到
        「真的还有旧向量」的块，只重排真正受影响的文件（当前库里只有 1 个）。
        重排后 indexer 会按新规则重切：极短块照常入库但 vector=NULL。
        """
        min_chars = int(min_chars or 0)
        if min_chars <= 0:
            return 0
        with self._write_lock:
            conn = self.conn
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                "WHERE file_id IN ("
                "  SELECT DISTINCT file_id FROM chunks "
                "   WHERE vector IS NOT NULL AND length(trim(text)) < ?)",
                (min_chars,),
            )
            conn.commit()
            return cur.rowcount

    def requeue_image_ocr_dupes(self) -> int:
        """把「因 parser/indexer 双写而多出一份 OCR 块」的文件重新排队；返回文件数（rev 3）。

        2026-09-24 新增。根因：`parser._parse_pdf` 已把图片型 PDF 的 OCR 文本拼进
        `doc.text`（前缀 `[第N页图片]`），`indexer.index_file` 又追加了一份同样内容
        （前缀 `[图N页]`）⇒ 同一页 OCR 文本入库两次、占两份向量。实测 bill.pdf 两块
        字符级相似度 0.994。

        判据取 `chunks.text LIKE '[图%页]%'` —— 这是 **indexer 侧独有的前缀**，
        parser 用的是 `[第N页图片]`。两处前缀字面不同，故该判据命中的每一个块都是
        重复块，不存在误伤；且 indexer 只在「parser 已无可读文本层」时才会走这条
        分支（`parsed.images` 只在那个分支被填充），所以也不存在「只有 indexer 那份、
        删了就搜不到」的情况。

        为什么要重排而不是就地删块：块的 `text` 与 `vector` 都在库外被引用
        （BM25 倒排、内存向量矩阵、`files.total_chunks`），就地删块会让这几处的
        计数与索引错位。重排走的是既有的「清 hash + 交给 indexer 重切」路径，
        状态与产物由同一条流水线重建。

        与 `requeue_truncated` / `requeue_short_chunks` 是同一类陷阱：派生数据的
        语义前提（该切成几块）变了，而 content_hash 只看文件字节、看不出来。
        """
        with self._write_lock:
            conn = self.conn
            cur = conn.execute(
                "UPDATE files SET vectorize_status='queued', content_hash='' "
                "WHERE file_id IN ("
                "  SELECT DISTINCT file_id FROM chunks "
                "   WHERE text LIKE '[图%页]%')"
            )
            conn.commit()
            return cur.rowcount

    def failed_files(self, limit: int = 200) -> list[dict]:
        """索引失败的文件明细（**持久**事实，按 DB 求值）。

        2026-09-28 新增。为什么不能只靠内存里的 `scanPhase.errorDetails`：
          · 计数 `vectorizePhase.failed` 本来就是按 DB 求值的**持久**值，明细却只是
            **本轮**内存流水 → 两者天生不同源；
          · 解析失败（PDF 损坏 / xlsx 缺内容表）走的是 index_file 正常返回
            `status=failed`，**从不写**那条内存流水 → 界面出现「状态说 4 个失败、
            失败文件格 0、明细空」（真机实测）。
        失败是持久属性（文件还是坏的，直到内容变化或修好），与本模块
        `truncated_files` 完全同构——那一条已用同样理由从「本轮流水」挪出来了。
        path 为 NULL 的行不参与（没有路径无法展示/定位）。
        """
        return [dict(r) for r in self.conn.execute(
            """SELECT path, file_type, fail_reason, attempt_count
                 FROM files
                WHERE vectorize_status='failed'
                  AND path IS NOT NULL
                ORDER BY path
                LIMIT ?""",
            (limit,),
        ).fetchall()]

    def truncated_files(self, limit: int = 200) -> list[dict]:
        """被块数上限截断的文件（**持久**事实，不是本轮扫描的错误记录）。

        2026-09-21 新增，配合 A 段改动：截断此前被塞进 `scanPhase.errorDetails`
        （scanner worker 调 `_record_error("oversized", ...)`）。那有两处不对：
          · 语义上它不是**错误**——这些块全部入库、文件照常可检索，只是内容不全；
          · 它是一条**持久**属性（文件内容不全，直到内容变化或重新索引），却被放进
            「本轮扫描」的明细里，扫描一结束/下一轮开始就被清空（见 scanner.start
            的 `_error_logs = []`），于是界面再也看不到「这个文件只索引了一部分」。
        改为按 DB 求值：判据 = 已知真实总块数、且已存块数更少。`total_chunks` 为 0
        （升级前入库、尚未重算）的行不参与——那是「未知」，不是「未截断」。
        """
        rows = self.conn.execute(
            """SELECT f.path                                   AS path,
                      f.total_chunks                           AS total,
                      (SELECT COUNT(*) FROM chunks c
                        WHERE c.file_id = f.file_id)           AS indexed
                 FROM files f
                WHERE f.total_chunks > 0
                  AND f.total_chunks > (SELECT COUNT(*) FROM chunks c
                                         WHERE c.file_id = f.file_id)
                ORDER BY f.total_chunks DESC
                LIMIT ?""",
            (int(limit),),
        ).fetchall()
        return [{"path": r["path"], "total": r["total"], "indexed": r["indexed"]} for r in rows]

    def replace_chunks(self, file_id: str, chunks: list[tuple[int, str, list[float] | None]]):
        """替换文件的全部块（file_id, chunk_index, text, vector）。

        2026-08-28：加 _write_lock 串行化 DELETE+INSERT（多线程并发 index 同一
        文件时原子性；INSERT OR REPLACE 在 UNIQUE(file_id,chunk_index) 上等价，
        但锁更简单且覆盖 DELETE 窗口）。

        2026-09-21：vector 允许为 None（落 NULL）—— 配合「极短块不向量化」（A）。
        这类块**保留 text**：BM25 稀疏检索仍能按字面命中（如 10 字节的
        `liuyingjie`），只是不参与语义路。语义是「不适合向量检索」，不是「不可检索」。
        """
        with self._write_lock:
            conn = self.conn
            conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            conn.executemany(
                "INSERT INTO chunks (file_id, chunk_index, text, vector) VALUES (?,?,?,?)",
                [(file_id, idx, text, _pack_vec(vec) if vec is not None else None)
                 for idx, text, vec in chunks],
            )
            conn.commit()

    def get_chunks(self, file_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT chunk_id, file_id, chunk_index, text, vector FROM chunks WHERE file_id=? ORDER BY chunk_index",
            (file_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d["vector"]:
                d["vector"] = _unpack_vec(d["vector"])
            else:
                d["vector"] = None
            out.append(d)
        return out

    def all_chunk_vectors(self) -> list[tuple[str, int, list[float]]]:
        """全量块向量（检索用）。返回 [(file_id, chunk_index, vec)]。"""
        out = []
        for r in self.conn.execute("SELECT file_id, chunk_index, vector FROM chunks WHERE vector IS NOT NULL"):
            out.append((r["file_id"], r["chunk_index"], _unpack_vec(r["vector"])))
        return out

    def close(self):
        """仅关闭调用线程的连接（其他线程连接由各自线程生命周期管理，避免 UAF）。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


def _gen_file_id(path: str) -> str:
    import hashlib

    return "f_" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
