"""索引服务：文件 → 分块 → 向量化 → 入库 + 文件聚合。

- index_file：单文件端到端（解析 → 分块 → embed → 入库）
- 内容 hash 变更检测：hash 相同则跳过
- 删除：移除向量 + 元数据 + 关联记录
- 不提供摘要（rerank / summary 不在范围）
"""
from __future__ import annotations

import hashlib
import json as _json
import logging
import threading
from pathlib import Path

from bibit_filesearch.chunker import need_chars_for_chunks, split_by_type
from bibit_filesearch.metadata import extract_metadata
from bibit_filesearch.model_client import EmbedClient
from bibit_filesearch.parser import IMAGE_TYPES, TEXT_TYPES, CODE_TYPES, OFFICE_TYPES, parse_file
from bibit_filesearch.db import Database

logger = logging.getLogger("bibit.indexer")

# 默认可索引类型（产生文本 → 向量化）。2026-08-26：从 parser.py 移出——
# 可索引性属于索引策略（indexer 消费），不属于文件解析（parser）。
# 2026-09-09：改为直接取 settings.ALL_INDEXABLE_TYPES（含 png/jpg OCR）。
# 此前是 TEXT|CODE|OFFICE，比设置页少图片——Indexer 构造后、settings 赋值前
# 的窗口内图片会被误判 unsupported。单一来源避免两处再漂移。
from bibit_filesearch.settings import ALL_INDEXABLE_TYPES  # noqa: E402

DEFAULT_INDEXABLE_TYPES = set(ALL_INDEXABLE_TYPES)

# 分块水位线：**会影响「一个文件应该有多少块」的改动**都要 +1，让存量文件按新规则
# 重索引一次。目前它覆盖 `max_chunks_per_file`（单文件块数上限）。
#
# 为什么需要：content_hash 是文件**字节**的 hash，截断/上限的变化不体现在里面。抬高
# 上限后重扫会判「内容未变 → skipped」，被截断的文件永远停在旧份数上。与
# parser.PARSER_REVISION 同源（那条管「解析器能不能读到内容」，这条管「读到之后留几块」）。
#   rev 1（2026-09-21）：max_chunks_per_file 默认 32 → 64。
#   rev 2（2026-09-21）：新增 min_embed_chars（极短块不向量化，A）——已入库的短块
#          必须重排一次才会被摘掉向量。判据见 db.requeue_short_chunks()。
#   rev 3（2026-09-24）：parser/indexer 双写导致图片型 PDF 多出一份 OCR 块——已入库的
#          重复块必须重排一次才会被摘掉。判据见 db.requeue_image_ocr_dupes()。
# 触发点在 main.py（启动时按水位线调 requeue_* 一次）。
CHUNK_REVISION = 3

# rev 3 的判据不需要参数：它匹配的是 indexer 侧独有的块前缀（`[图N页]`），
# 与 parser 侧的 `[第N页图片]` 字面不同，天然只命中重复块。见 db.requeue_image_ocr_dupes。

# rev 2 的判据：**已入库且仍有向量**的极短块。为什么不能只靠 content_hash 幂等：
# 极短块的向量已经落库，文件内容没变 → 重扫判 skipped，旧向量永远活着。
# 用「文本长度 < 阈值且 vector 非空」精确定位，只重排真正受影响的文件。
SHORT_CHUNK_REVISION_MIN_CHARS = 3

# rev 1 **之前**生效的默认单文件块数上限（32）。只服务于「老库」：
# 老库刚补上 `files.total_chunks` 列时该列一律为 0（未知），拿不到真实总块数，
# 于是只能靠「块数恰在旧上限处」来推断是否曾被截断——旧上限 32 之下，库里不可能
# 出现 >32 块的记录，所以「存了 32 块」就等价于「可能被截断」。
# 代价：本来就是整 32 块的文件会被多索引一次（一次性、由水位线兜住）；收益：不遗漏
# 任何一个真被截断的（漏掉的代价是永久搜不到那部分内容，且界面显示正常）。
# 若用户当年自己设过更高的上限，用这条判据会漏——已在 docs/configuration.md 说明。
# 后续 rev 起 total_chunks 已有值，此常量不再参与（见 db.requeue_truncated）。
CHUNK_REVISION_LEGACY_CAP = 32


class Indexer:
    def __init__(self, db: Database, embed: EmbedClient, chunk_size: int = 500, chunk_overlap: int = 100,
                 indexable_types: set[str] | None = None, max_chunks_per_file: int = 64,
                 min_embed_chars: int = 3):
        self.db = db
        self.embed = embed
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 2026-09-21（A）：短于此字符数的块**不做向量化**（0 = 关闭）。见 index_file 里
        # 那段说明：embedding 对 1~2 token 的输入会退化到「对任何查询都 0.9+」，
        # 从而稳定挤掉真实文档。判据用字符数而非估算 token 数（后者实测不可行）。
        self.min_embed_chars = int(min_embed_chars or 0)
        # 2026-09-18：单文件最多索引多少块（0 = 不限）。超限只索引前 N 块并标记 oversized。
        # 依据（实测）：一棵 40MB 反编译 JS 树共 111,249 块，其中单个 24.8MB 的 minified
        # 文件占 **83,782** 块（生产日志实测；早期「约 4.3 万」是估算，错了一倍）。
        # 为什么按块数卡：原推导是「embed 服务对每条没见过的新内容长期占内存且不归还」，
        # 实测每 600 字符稠密内容 ≈ 55~65MB（随机 base64 600 字符 50~137MB/条；600 字符
        # JS ≈ 570 token，按 ~112KB/token 折合约 64MB），故上限 = 单文件可接受内存 / 每块成本。
        #
        # ⚠️ 2026-09-21 订正：上面这条「每块成本」只在 embed 用**默认 cache-ram（8192）**
        # 时成立。`--cache-ram 0`（现已进网关 default_args，重启不再丢）实测：
        #   · 256 个全新块 → RSS **−1MB**（8 批全 +0，边际成本≈0）；
        #   · 286KB markdown 全量 599 块 → RSS **+13MB**、521s（-t 4）。
        # 内存几乎与块数无关（一次性台阶 + 平台），真正的约束变成**时间**。
        # 故默认 32 → 64：覆盖更宽的文档谱，同时仍给异常文件留下界（配合 oversized 记录，
        # 截断不再静默）。旧值 32 的遗留理由「真实文档最多 31 块因而零截断」在引入
        # 40MB 级文档目录后已不成立（实测 k1_ds.md 599 块只入 32 → 覆盖 5.3%）。
        # 注意：本上限只约束**单个文件**的冲击，约束不了总量。
        self.max_chunks_per_file = int(max_chunks_per_file or 0)
        # 可索引类型：运行时可由 settings 覆盖（前端"可索引格式"勾选）
        self.indexable_types = indexable_types if indexable_types is not None else set(DEFAULT_INDEXABLE_TYPES)
        self._on_indexed = None  # cb(file_id, vectors)
        self._on_deleted = None  # cb(file_id)

        # 2026-09-22：在飞索引计数 + 同路径占位（见 index_file 的说明）。整数 + 锁，因为
        # index_file 会被 scanner worker 与 watcher 线程**并发**调用；
        # `self._inflight += 1` 不是原子的（LOAD/BINARY_ADD/STORE 之间可被切换），
        # 丢一次自增就会让「有活在干」被误判成「管道空了」，状态在干活期间掉回 done。
        self._inflight = 0
        # 正在被索引的**路径 → 占位槽**（同路径单飞，见 index_file）。槽里放完成事件、
        # 赢家的返回值与异常：输的那一方**等赢家收尾然后复用它的结果**，而不是自己
        # 也去解析一遍，也不是空手返回 —— 理由见 index_file 里那段说明。
        self._active_paths: dict[str, dict] = {}
        self._inflight_lock = threading.Lock()
        # 等同一路径在飞索引收尾的上限（秒），仅作防挂死兜底。embed 服务本身是单槽
        # 串行的（vectorize_parallelism=1），输的那一方就算不等也抢不到算力，所以
        # 「等」几乎不付额外代价。实测单文件最长 8.7 分钟（599 块 @0.87s/块），故留 900。
        self.duplicate_wait_seconds = 900

    @property
    def inflight_indexes(self) -> int:
        """当前正在 index_file 里的文件数（进程内，含 scanner / watcher 全部路径）。

        消费者：`ScanService.progress()` 用它派生对外状态——「done」必须保证
        真的没有活在干（契约见 docs/scan-status-contract.md §3.2）。
        """
        with self._inflight_lock:
            return self._inflight

    def set_on_indexed(self, cb):
        self._on_indexed = cb

    def set_on_deleted(self, cb):
        self._on_deleted = cb

    # ---------- 主流程 ----------
    def index_file(self, path: str) -> dict:
        """索引单文件（对外唯一入口）：见 `_index_file`。

        2026-09-22：本方法承担两件事——**在飞计数**与**同路径单飞**（skip 时返回
        `{"status": "inflight"}`）。真正的实现在 `_index_file`。

        ── 为什么用包装而不是在原函数体里加逻辑 ──
        `_index_file` 有 5 条 return 路径（not_a_file / unsupported / skipped /
        no_text / 正常），要在体内包住全部路径得整段缩进、改动面大且容易漏；包装法
        把「入口登记、出口销号」收在一处，`finally` 保证异常路径也销号。

        ── 计数为什么必须在**这一层** ──
        `index_file` 是 scanner worker、watcher 事件、watcher 补扫三条路径的唯一
        公共入口（实测 grep：只有 4 处调用点，全走它），所以一个计数器就能覆盖全部
        索引来源——包括 watcher 那条**不写 queued**的路径，这正是「拷贝目录后界面
        不显示扫描中」的根因（契约见 docs/scan-status-contract.md §3.2）。
        漏减会让状态**永久**卡在 scanning（比原 bug 更糟），`finally` 是唯一能覆盖
        所有出口的写法。

        ── 同路径单飞为什么也必须在这一层（F8，2026-09-22）──
        索引有两条路径（scanner 全盘 / watcher 增量），而**两条都在索引开始之前**才
        查库问「这个文件在处理吗」，可库里那行是在 `embed_batch` **之后**才写的
        （`upsert_file`）。于是文件在整个 embed 窗口里**在库中不存在**，另一条路径的
        hash 快路径（要求 `status == 'completed'`）判不出「已在处理」→ 完整解析 +
        向量化两遍。真机实测：`Theartofmulticore.pdf` 4 秒内被 parse 两次（两条
        `bibit.indexer: 文件过大`），端到端 50s → 120s；受控实验 200 文件并发启动
        重复系数 1.99。**服务每次启动都会重演**（scanner 自动扫 + watcher 延迟补扫
        走同一批文件），不是边角案例。

        为什么用**进程内路径集合**而不是「提前写库」：
          · 库侧写占位会牵动 `progress()` 的 total/queued/unsupported 各格语义、
            `_run_worker` 的 skipped 兜底（「有向量→completed，无向量→unsupported」）、
            `_recover_interrupted`（启动时把 vectorizing 复位 queued）、以及
            `truncated_files` / `requeue_truncated` 的判据 —— 一处提前写库会让多个
            消费端同时变义，正是「改动范围失控」的典型。
          · 而本仓的两条路径**就活在同一个进程里**（`Watcher(config, db, indexer)`
            拿的是同一个 indexer 实例），进程内集合足够，且不动任何持久状态。

        为什么不直接把重复请求丢掉、而要**等赢家做完再复用它的结果**：索引窗口里
        文件仍可能被继续写（编辑器保存、正在拷贝），丢掉事件 = 这次变更彻底没人处理，
        直到下一次事件或全盘扫描才补；而赢家那次解析读的是**旧内容**，最终落库的就是
        旧内容 —— 「跳过重复」变成了「跳过更新」。等待则把两种语义分开：重复的解析被
        省掉，但结果照样送达。代价可控：embed 服务是单槽串行的
        （`vectorize_parallelism=1`），输的那一方即使不等也拿不到算力，等不会让系统
        多花时间。

        代价（诚实说明，两条）：
          · **跨进程/重启**的重复不在本判据覆盖内 —— 那种窗口落在
            `_recover_interrupted` 与 hash 快路径之间，属另一条线。本判据覆盖的是
            「进程活着时两条路径撞车」，也就是用户实测到的那一条。
          · 等待有上限（`duplicate_wait_seconds`，仅防挂死的兜底）；超时才返回
            `{"status": "inflight"}`，让调用方按「本轮没做成」处理（scanner worker
            会把状态还原，见 `_run_worker`）。正常路径下这个上限不会被触及。
        """
        me = threading.get_ident()
        with self._inflight_lock:
            slot = self._active_paths.get(path)
            if slot is None:
                slot = {"done": threading.Event(), "owner": me, "result": None, "error": None}
                self._active_paths[path] = slot
                # 登记与计数在同一临界区，保证「计数不为 0 ⟺ 集合非空」
                self._inflight += 1
                i_am_owner = True
            elif slot["owner"] == me:
                # 同一线程重入同一路径。若这里去等自己的完成事件就是**自锁**；
                # 故按「重复解析一次」放行（宁可多算一遍也不挂死），并留告警。
                # 理论上不该发生：_index_file 与它触发的回调都不会回调 index_file。
                logger.warning("同线程重入索引同一路径，按重复解析处理: %s", path)
                i_am_owner = False
            else:
                i_am_owner = False

        if not i_am_owner:
            if slot["owner"] != me:
                # 另一条路径正在索引同一文件：**不重复解析**，等它收尾后复用结果。
                # （不复用 "skipped"——那个字符串在 scanner._run_worker 里会触发
                # 「向量在不在」的兜底校正，语义是「hash 未变而跳过」，与本情形不同；
                # worker 侧对 "inflight" 的处置见 scanner._run_worker。）
                logger.debug("已在索引中，等待其收尾后复用结果: %s", path)
                if not slot["done"].wait(timeout=self.duplicate_wait_seconds):
                    logger.warning("等待同路径索引收尾超时（%.0fs），放弃本次请求: %s",
                                   self.duplicate_wait_seconds, path)
                    return {"status": "inflight", "reason": "waited_too_long", "path": path}
                if slot["error"] is not None:
                    raise slot["error"]
                return slot["result"]
            # 同线程重入：不登记、不等待，直接算一遍，出口也不摘牌（牌是外层那次登记的）
            return self._index_file(path)

        try:
            r = self._index_file(path)
            slot["result"] = r
            return r
        except BaseException as e:
            slot["error"] = e
            raise
        finally:
            with self._inflight_lock:
                self._active_paths.pop(path, None)
                self._inflight -= 1
            # 先摘牌再放行等待者：等待者被唤醒后若再发一次请求，会拿到新的一次索引
            # （旧结果可能已过期），不会误接到刚结束的这一次上。
            slot["done"].set()

    def _index_file(self, path: str) -> dict:
        """索引单文件：返回 {file_id, status, changed}。

        status: indexed（新入库）/ updated（内容变更重算）/ skipped（hash 未变）/ unsupported（不可索引类型）
        """
        p = Path(path)
        if not p.is_file():
            return {"status": "skipped", "reason": "not_a_file"}
        # 2026-09-18：失败留痕必须覆盖本函数的**全部**执行路径，包括取元数据与算 hash。
        # 此前 try 从 parse_file 才开始，读不到文件时（PermissionError/OSError）异常直接
        # 抛给调用方：worker 调用前已把状态置为 vectorizing，而记录失败的代码在 try 内，
        # 于是状态永远停在 vectorizing、attempt_count 不增——既看不到失败（UI 无明细）
        # 也永不重试（实测：root 拥有、0600 权限的文件）。
        meta = None
        try:
            meta = extract_metadata(path)
            ftype = meta["fileType"]

            # other / 纯图片无 OCR 文本 → 仅元数据，不入向量
            # 图片类型放行进向量化判断——parser 已尝试 OCR，
            # OCR 成功有文本则向量化（图片文字可检索），失败无文本则走 no_text 仅元数据
            # 2026-08-31：图片类型也受"支持的格式"约束——未勾选的图片类型不索引
            # （此前 IMAGE_TYPES 无条件放行 OCR，用户取消图片格式无效）
            if ftype not in self.indexable_types:
                self._upsert_meta_only(meta, ftype)
                return {"status": "unsupported", "file_type": ftype}

            content_hash = self._content_hash(p)
        except Exception:
            self.mark_failed(path, meta=meta, content_hash="")
            raise

        existing = self.db.get_file_by_path(path)

        # 不提供摘要：hash 未变即跳过。
        # 2026-09-18：**必须是 completed 才跳过**。上一版把 failed 也算进这条快路径，
        # 与「失败要落 hash」自相矛盾——落了 hash 之后每次重试都被判「内容未变」而
        # 跳过，失败文件永远不再尝试（随后被 worker 当作无向量的悬挂记录改成
        # unsupported）。失败文件的去重应由 attempt_count 上限负责（见 scanner._enqueue_file），
        # 不是由 hash 快路径负责。
        if (existing and existing["content_hash"] == content_hash
                and existing["vectorize_status"] == "completed"):
            return {"status": "skipped", "file_id": existing["file_id"]}

        # 解析 → 分块 → embed。2026-09-18：整段包 try —— 失败必须留痕（见 mark_failed），
        # 且必须在本层做，因为 index_file 是 scanner worker / watcher 补扫 / backfill
        # 三条路径的公共入口；只在 scanner worker 侧记，watcher 路径的失败就漏掉了。
        pdf_images = "[]"
        try:
            # 2026-09-29：把"要留几块"换算成"最少需要多少字符"传给 parser，
            # 让它能在 PDF 解析过程中早停（"填满就不再往后看"）。
            # ★ 换算住在 chunker（见 need_chars_for_chunks），parser 仍不知道"块"；
            #   max_chunks_per_file=0（不限）时换算得 0 → parser 不早停（今天的行为）。
            #   实测收益：529 页文本型书 6.3s→2.5s、511 页扫描书 43.9min→3.3min。
            parsed = parse_file(path, min_chars=need_chars_for_chunks(
                self.chunk_size, self.chunk_overlap, self.max_chunks_per_file))
            meta["author"] = parsed.author or meta.get("author", "")
            chunks_text = split_by_type(parsed.text, ftype, self.chunk_size, self.chunk_overlap)
            chunks_text = [c for c in chunks_text if c.strip()]
            # 2026-08-26 防线：超长块截断（兜底防极端输入——原超长块直接 embed 500 报错且
            # 前端无提示）。注意**这不是 token 级的保证**：字符数与 token 数不成比例
            # （压缩 JS 实测 600 字符 = 548~616 token，5 个样本），按字符预估 token 不可靠；
            # 真正的保护是
            # EmbedClient 在被服务端按物理批次拒收后按实际结果二分降级（_embed_resilient）。
            MAX_CHUNK_CHARS = 800
            chunks_text = [c[:MAX_CHUNK_CHARS] if len(c) > MAX_CHUNK_CHARS else c for c in chunks_text]

            # PDF 图生文关联：图片 page_idx + OCR 描述落 `files.pdf_images`
            # 供溯源查询。**这里不再把 OCR 文本追加成块**。
            #
            # 2026-09-24 修复（重复块）：此前这里 `chunks_text.append(img_desc)`，
            # 而 parser._parse_pdf 早已把同一份 OCR 文本拼进 `doc.text`
            # （`parts.append(f"[第{pageno+1}页图片] {ocr_text}")`）。于是同一个文件的
            # 同一页 OCR 文本入库**两次**，占两份向量。实测 bill.pdf：块0 以
            # `[第1页图片]` 开头、块1 以 `[图1页]` 开头，字符级相似度 0.994。
            #
            # 为什么不删 parser 那半：`parsed.text` 必须含 OCR 文本，它是**唯一**被
            # 分块/向量化/BM25 索引的内容来源（image-only PDF 的全部可检索内容就是它）。
            # 删掉那边等于把图片型 PDF 变成搜不到。所以删的是本处的**二次**追加。
            #
            # 残留提醒：`img_desc` 曾是「把 OCR 送进向量化内容」的唯一手段的假设来自初版
            # （parser 当时 `images` 恒空、`_parse_pdf` 返回 `(text, author, [])`），
            # 该假设在 2026-09-07 接入 OCR 后就不再成立，但追加语句一直留着。
            if parsed.images:
                # ⚠️ 2026-09-28：此处原先写 `import json as _json`（函数内导入）。
                # 那会让 `_json` 成为**整个函数的局部名**，于是本函数后半段
                #（except 里落失败原因）用 `_json` 时若走的不是这条分支，就
                # UnboundLocalError —— 被 watcher 的测试当场逮到
                #（明细 reason 变成 "cannot access local variable '_json'"）。
                # 改成模块级 import（见文件头），本处直用。
                pdf_images = _json.dumps(parsed.images, ensure_ascii=False)

            if not chunks_text:
                # ★ rev 5（2026-09-28）：**没有文本**与**文本没取到**必须分开。
                #   OCR 唯一化之后，引擎/worker/模型任一失效都会让图片解析出空文本，
                #   而空文本原先与「空文件」「纯图片 PDF」一样落 unsupported → 前端
                #   显示「未纳入索引 · 无文本」，用户看不出是故障还是本来没字。
                #   判据用 parsed.error（parser 在 OCR 异常时写入），所以：
                #     · 有 error → failed：带原因、计 attempt、可重试、前端显示「失败」
                #     · 无 error → no_text：空文件/白图，与以前一致
                if parsed.error:
                    self.mark_failed(path, meta=meta, content_hash=content_hash,
                                     pdf_images=pdf_images, reason=parsed.error)
                    rec = self.db.get_file_by_path(path)
                    return {"status": "failed", "file_id": rec["file_id"] if rec else None,
                            "error": parsed.error}
                # 无文本（如空文件/纯图片 PDF 文本层空且无 OCR）→ 仅元数据（图生文关联仍入库供溯源）
                self._upsert_meta_only(meta, ftype, content_hash, pdf_images=pdf_images)
                return {"status": "no_text", "file_id": self.db.get_file_by_path(path)["file_id"]}

            # 2026-09-18：单文件块数上限（防超大文件把 embed 灌爆）。
            # 超限**只索引前 N 块**并标记 oversized——文件仍可检索（这些块照常入库），
            # 只是内容不完全。刻意不用独立状态：检索侧只认 vectorize_status='completed'
            # 作为候选，新状态会让被截断的文件整份搜不到，与「只索引前 N 块」的意图相反。
            total_chunks = len(chunks_text)
            oversized = bool(self.max_chunks_per_file and total_chunks > self.max_chunks_per_file)
            if oversized:
                chunks_text = chunks_text[: self.max_chunks_per_file]
                logger.warning(
                    "文件过大，仅索引前 %d 块（共 %d 块）：%s",
                    self.max_chunks_per_file, total_chunks, path,
                )

            # 2026-09-21（A）：极短块**不向量化**（阈值 min_embed_chars，0 = 关闭）。
            #
            # 为什么：embedding 对 1~2 token 的输入会退化——实测同一个 Qwen3-Embedding
            # 下，任意两个 1-token 输入的余弦是 0.93~0.99，而「1-token 查询 × 500 字符
            # 文档块」只有 0.26~0.37。于是内容为 `1` 的块对**任何**查询都拿 0.90+，
            # 稳定挤掉全部真实文档（实测：24 个查询里 1~2 token 查询 12/12 全部由它
            # 占据第一；换任意字符 'a'/'x'/'好' 结论相同，与内容无关）。
            #
            # 为什么不按「估算 token 数」判：实测 tokenizer 对连续 ASCII 词是 1 token
            # （'flutter'/'test'/'README'），对数字串是逐字符（'123'→3、'12345'→5），
            # 两类无法用同一个字符启发式兼顾 —— 任何估算式都必然在一边误判。故按最保守
            # 的**字符数**判：先把最脏的一档（单字符块）挡住，且该判据跨模型稳定。
            #
            # 留 text 还是丢 text：**留**。BM25 稀疏路仍按字面命中（`liuyingjie` 这类
            # 短标识符照常搜得到），语义是「不适合向量检索」而非「不可检索」。
            # 若整个文件的所有块都低于阈值 → 该文件没有向量（语义路搜不到），
            # 状态仍记 completed（它确实索引完了，只是无向量可参与语义路）。
            emb_idx = [i for i, t in enumerate(chunks_text) if len(t.strip()) >= self.min_embed_chars]
            to_embed = [chunks_text[i] for i in emb_idx]
            vectors = self.embed.embed_batch(to_embed)
            # 断言：embed 返回数量必须等于**送进去的**数量，禁止 zip 静默截断
            # （修复前 OCR 块丢失的根因）
            if len(vectors) != len(to_embed):
                raise RuntimeError(
                    f"embed 返回 {len(vectors)} 条 != 送检 {len(to_embed)} 条（{path}），该文件索引失败待重试"
                )
            skipped_short = len(chunks_text) - len(emb_idx)
            if skipped_short:
                logger.info(
                    "跳过 %d 个极短块（< %d 字符）的向量化，保留文本供关键词检索：%s",
                    skipped_short, self.min_embed_chars, path,
                )
        except Exception as e:                        # noqa: BLE001
            # 原因一并落库（2026-09-28）：此前只落状态，重扫后原因只能在内存
            # errorDetails 里看到，而那份每轮清空。
            # 2026-09-28（test3）：结构化原因（界面按语言出词，原文进 detail）。
            # 保留 "embed:" 前缀 —— 既有测试与文档按它判 kind，不能悄悄改掉。
            self.mark_failed(path, meta=meta, content_hash=content_hash,
                             pdf_images=pdf_images,
                             reason="embed:" + _json.dumps(
                                 {"kind": "embedFailed",
                                  "errorType": e.__class__.__name__,
                                  "detail": (str(e).strip() or e.__class__.__name__)[:200]},
                                 ensure_ascii=False))
            raise

        summary = ""

        file_id = self.db.upsert_file(meta, content_hash, summary, pdf_images=pdf_images,
                                      total_chunks=total_chunks)
        # 逐块配对：送检过的块带向量，其余块 vector=None（文本照常入库）
        vec_by_idx = dict(zip(emb_idx, vectors))
        self.db.replace_chunks(
            file_id,
            [(i, t, vec_by_idx.get(i)) for i, t in enumerate(chunks_text)],
        )
        self.db.set_vectorize_status(file_id, "completed")
        if self._on_indexed:
            # 回调只带向量（内存矩阵用）；极短块不进矩阵，但会经 BM25 收进稀疏路
            self._on_indexed(file_id, vectors)

        status = "updated" if existing else "indexed"
        return {"status": status, "file_id": file_id, "chunks": len(chunks_text),
                "embedded": len(vectors), "short_skipped": skipped_short,
                "oversized": oversized, "total_chunks": total_chunks}

    def delete_file(self, file_id: str):
        """删除文件（向量 + 元数据 + 关联记录）。"""
        self.db.delete_file(file_id)
        if self._on_deleted:
            self._on_deleted(file_id)

    def delete_by_path(self, path: str):
        # 必须走 delete_file（触发 on_file_deleted 回调）——此前直接 db.delete_file 导致
        # 检索内存矩阵/BM25 倒排不更新（watcher 单文件删除路径，目录删除走 _remove_dir_index 是干净的）
        f = self.db.get_file_by_path(path)
        if f:
            self.delete_file(f["file_id"])

    def file_count(self) -> int:
        return self.db.count_files()

    # ---------- 内部 ----------
    def _content_hash(self, p: Path) -> str:
        """文件内容 hash（增量变更检测）。"""
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def mark_failed(self, path: str, *, meta: dict | None = None, content_hash: str = "",
                    pdf_images: str = "[]", reason: str = "") -> None:
        """记录一次索引失败：落 content_hash + 重试计数（2026-09-18）。

        为什么必须在**本层**：`index_file` 是 scanner worker、watcher 补扫、backfill
        三条路径的公共入口。此前只有 scanner worker 的 except 里记了 hash/计数，
        watcher 那条路径失败时只打一行日志 —— 于是失败文件 hash 为空、计数为 0，
        每轮扫描都判它「待重算」→ 重新解析 + 重新分块 + 重新 embed（实测 169：单个
        24.8MB 文件每轮重切 83,782 块、重灌前 max_chunks_per_file 块，embed 内存
        逐轮上涨，正是「什么都没做却一直忙」的成因）。

        落 hash 后：内容未变的失败文件可被识别，配合 attempt_count 达上限即不再重排；
        用户改了文件则 hash 变化，仍能重新排队重试。

        调用方（唯一落点，勿在别处另写一份改状态的实现）：
          · `index_file` 内部——解析/embed 失败、读文件失败各一次；
          · `watcher._mark_failed`——事件处理本身的异常（此时不带 meta）。
        watcher 曾另有一份「只改状态、不计数、不落 hash」的实现（2026-09-09），
        与这里的语义冲突：实测读不到的文件被写成 failed / attempt_count=0 / hash 空，
        而扫描对 failed 且未达上限的文件**每轮都重新排队** → 读不到的文件每轮重试一次、
        计数永远为 0，正是「什么都没做却一直忙」的另一条来源。此路径已并入本方法。

        meta 缺省时尽力自取（文件存在但读不了，stat 往往仍可用）；仍取不到就退化为
        「按 DB 里已有记录落一次失败」，DB 里没有记录则无处可记（从未入过索引）。

        reason（2026-09-28）：失败原因**落库**（files.fail_reason），供界面「失败文件」
        明细显示。传空则库里记空——此时原因由调用方的 _record_error 走内存明细，
        那条路只有遍历/embed 异常会走；解析失败（PDF 损坏等）走的是本方法的
        status=failed 正常返回路径，必须靠这个参数才能让原因可见。
        """
        try:
            if meta is None:
                try:
                    meta = extract_metadata(path)
                except Exception:  # noqa: BLE001 —— 取元数据本身失败不阻断留痕
                    meta = None
                if meta is not None and meta["fileType"] not in self.indexable_types:
                    return  # 不可索引类型不标 failed（与 _upsert_meta_only 的 unsupported 一致）
            if meta is None:
                row = self.db.get_file_by_path(path)
                if row:
                    self.db.mark_attempt_failed(row["file_id"], content_hash, reason)
                return
            file_id = self.db.upsert_file(meta, content_hash, "", status="failed",
                                          pdf_images=pdf_images)
            self.db.mark_attempt_failed(file_id, content_hash, reason)
        except Exception:  # noqa: BLE001 —— 记录失败状态本身不得掩盖原始异常
            logger.exception("记录失败状态出错: %s", path)

    def _upsert_meta_only(self, meta: dict, ftype: str, content_hash: str = "", pdf_images: str = "[]"):
        # 2026-08-28：不可索引类型用 'unsupported' 状态（区别于已向量化的 completed），
        # 检索筛选（vectorize_status='completed'）不会把无向量的文件当候选
        self.db.upsert_file(meta, content_hash, "", status="unsupported", pdf_images=pdf_images)
