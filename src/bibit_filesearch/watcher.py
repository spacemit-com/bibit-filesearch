"""增量扫描：inotify 监听。

- ctypes 直调 inotify（零依赖，D3 决策）
- 监听配置目录（与全盘扫描范围一致）
- 事件合并：100ms 窗口批量处理（避免事件风暴）
- 新增→index_file；修改→index_file（内部 hash 比对，内容未变跳过）；删除→清理索引
- 近实时：文件变更后索引更新延迟 < 10s
- 与全盘共用同一套索引结构/队列
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import struct
import threading
import time
from pathlib import Path

from bibit_filesearch.excludes import is_excluded
from bibit_filesearch.model_client import EmbedError

logger = logging.getLogger("bibit.watcher")

# inotify 常量
IN_ACCESS = 0x00000001
IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_IGNORED = 0x00008000
IN_Q_OVERFLOW = 0x00004000
IN_ISDIR = 0x40000000
IN_ONLYDIR = 0x01000000

# 关注事件：写入完成 / 创建 / 移动 / 删除 / 属性变更 / 队列溢出（溢出触发全量补扫兜底）
WATCH_MASK = (
    IN_CLOSE_WRITE | IN_CREATE | IN_MOVED_TO | IN_MOVED_FROM | IN_DELETE | IN_DELETE_SELF | IN_ATTRIB
)

# 合并窗口（秒）：窗口内事件合并为一次处理
MERGE_WINDOW = 0.1


class _InotifyEvent(ctypes.Structure):
    _fields_ = [
        ("wd", ctypes.c_int),
        ("mask", ctypes.c_uint32),
        ("cookie", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
    ]


class Watcher:
    """目录监听（inotify）—— 重构后是**纯生产者**：只往索引池里放/撤，不索引任何东西。

    历史上这里直接调
    `indexer.index_file(path)`，于是：

      · 增量索引的文件**在库中没有行**（状态只活在进程内存里，重启即消失）；
      · 它不写 `vectorize_status='queued'`，`progress()` 的 done 判据看不见它
        → 拷一个目录进来正在算 49 秒，界面全程显示「已完成」（9-22 修的那条）；
      · 同一路径的单飞要靠 `indexer._active_paths` 这个进程内集合。

    现在它只做 `pool.offer()`（放）与 `pool.purge()/cancel_tree()`（撤），
    去重由池中那一行天然保证（UNIQUE(path)），状态跨进程可见。
    """

    def __init__(self, config, db, indexer, on_event=None, pool=None):
        self.index_dirs = [str(d) for d in config.index_dirs]
        self.excludes = list(config.exclude_patterns)
        # 索引深度（1-based，2026-09-04）：与 scanner 一致，0=无限；
        # N=base+下 N-1 层子目录（1=仅 base 本身，限制 watch/补扫层数）
        self.max_depth = int(config.scan.get("max_depth", 0) or 0)
        self.db = db
        self.indexer = indexer
        self.on_event = on_event  # 可选外部回调（测试注入）
        # 索引池（重构）：注入时用共享池（生产：与 scanner 同一个，消费者由 scanner 持有）；
        # 未注入时自查构造一个**自带消费者**的池子，保持「单独构造 Watcher 也能工作」
        # 这一旧行为（测试与脚本常用这种构造方式）。
        self._own_worker = pool is None
        if pool is None:
            from bibit_filesearch.pool import IndexPool, IndexWorker

            pool = IndexPool(db, indexer)
            self._worker = IndexWorker(pool, indexer)
        else:
            self._worker = None
        self.pool = pool

        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            raise RuntimeError("无法定位 libc")
        self._libc = ctypes.CDLL(libc_name, use_errno=True)
        # 关键：显式声明 restype/argtypes，否则 ctypes 默认 int（riscv64 上指针/返回值截断）
        self._libc.inotify_init1.restype = ctypes.c_int
        self._libc.inotify_init1.argtypes = [ctypes.c_int]
        self._libc.inotify_add_watch.restype = ctypes.c_int
        self._libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._libc.close.restype = ctypes.c_int
        self._libc.close.argtypes = [ctypes.c_int]
        self._fd = -1
        self._wds: dict[int, str] = {}  # wd → 目录路径
        self._path_to_wd: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._backfill_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # ScanService 引用（main.py / 测试注入；`_mark_failed` 用它写错误明细）。
        # 2026-09-23（重构）：同时同步给池子一份 —— 消费者归池子管，而 watcher
        # 自持池的路径上消费者拿不到 scanner 构造参数，靠 `pool.scanner_ref` 兜底
        # （见 IndexPool.scanner_ref / IndexWorker._record_error）。
        self._scanner_ref_value = None

    # ---------- scanner 引用 ----------
    @property
    def _scanner_ref(self):
        return self._scanner_ref_value

    @_scanner_ref.setter
    def _scanner_ref(self, value):
        """赋值时同步给池子。

        调用点（`main.py` 与多个测试）都是 `watcher._scanner_ref = scanner` 这种
        直接赋值，用属性让它自动完成同步，不必改调用点。
        """
        self._scanner_ref_value = value
        try:
            self.pool.scanner_ref = value
        except Exception:  # noqa: BLE001 —— 池子可能被替换/只读，同步失败不影响主流程
            pass

    # ---------- 生命周期 ----------
    def start(self):
        """注册监听（递归子目录）→ 启动补扫线程 → 启动读线程。

        双保险（D-5 加固）：
        1. 注册 watch 后立即补扫现有目录树（已有文件全部索引）——即使全盘扫描未运行/失败，
           watcher 也自包含地"认识"目录树全貌，之后增量事件只跟变化；
        2. 目录创建时同样注册+补扫（_backfill_dir），兜底"注册前拷入"的事件丢失。
        """
        if self._thread and self._thread.is_alive():
            return
        # 修复（2026-09-04）：stop() 只 set() 不复位，start() 若不复位 _stop，
        # 每次 PUT settings 触发的 stop()+start() 后新建的读线程会在
        # _read_loop 的 `while not self._stop.is_set()` 处立即退出 —— inotify fd
        # 无人读取、增量事件永久堆积，auto_index 静默失效（watcher 假活）。
        self._stop.clear()
        # 自持消费者（未注入共享池时）：起一个，否则自建的池子没人消费
        # （重构后的 watcher 只登记不算，消费者必须有人在）。
        if self._worker is not None:
            self._worker.start()
        self._fd = self._libc.inotify_init1(os.O_NONBLOCK)
        if self._fd < 0:
            raise OSError("inotify_init1 失败")
        for d in self.index_dirs:
            self._add_watch_recursive(d)
        # 启动补扫（后台线程，不阻塞服务启动）
        # 2026-08-28：延迟补扫，避免与 scanner 首次全盘扫描同时抢 embed（K3 上
        # watcher 补扫 + scanner 向量化并发 → embed 请求超时 failed）
        delay = getattr(self, "backfill_delay", 0.0)
        scanner = getattr(self, "_scanner_ref", None)

        def _delayed_backfill():
            # 若持有 scanner 引用：等其首次扫描完成（status != scanning 则补扫）
            if scanner is not None:
                waited = 0.0
                while waited < delay:
                    if self._stop.is_set():
                        return  # 已停止（PUT settings 触发的 stop）：本轮不再补扫
                    try:
                        if getattr(scanner, "status", "idle") != "scanning":
                            break
                    except Exception:
                        pass
                    # 用 Event.wait 而非 time.sleep：stop() 能立刻打断这段等待，
                    # 不必等满 2 秒（stop() 里要 join 本线程）
                    self._stop.wait(2.0)
                    waited += 2.0
            elif delay > 0:
                self._stop.wait(delay)
            if self._stop.is_set():
                return
            self._backfill_all()

        self._backfill_thread = threading.Thread(target=_delayed_backfill, daemon=True)
        self._backfill_thread.start()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("inotify 监听启动: %s", self.index_dirs)

    def _backfill_all(self):
        """补扫全部索引目录的现有文件（watcher 启动时调用）。"""
        for d in self.index_dirs:
            if self._stop.is_set():
                return
            self._backfill_dir(d)
        # 补扫可能把一整棵含被排除子目录的树纳入视野 → 排除读数可能已变（2026-10-10）
        self._request_excluded_refresh()

    def stop(self):
        """停止监听。★ 2026-09-23：必须**等补扫真正结束**再返回。

        此前只置 `_stop` 并 close inotify fd（补扫线程 daemon，不等）。后果：
        `PUT /api/settings` 里「先回收、后 stop」的次序下，正在跑的那一轮补扫
        仍拿着**旧的** index_dirs / exclude_patterns 走目录，继续把文件 offer 回
        池子 —— 刚被回收的文件又被加回库里（用户可见：加了排除规则，文件又冒出来）。
        补扫线程必须在回收之前真的停下，本次改动才成立。

        join 超时 5 秒：`_backfill_dir` 每 offer 一个文件都查一次 `_stop`（见该
        方法），所以正常情况下几十毫秒内就返回；超时只是防死等（daemon 线程不会
        阻止进程退出，超时后不再强求）。
        """
        self._stop.set()
        # 自持消费者跟着停（共享池的消费者归 scanner 管，不在这里停）
        if self._worker is not None:
            self._worker.stop()
        if self._fd >= 0:
            try:
                self._libc.close(self._fd)
            except Exception:
                pass
            self._fd = -1
        if self._thread:
            self._thread.join(timeout=2)
        # 补扫线程：必须先于「回收」结束（调用方次序保证），这里等它落地
        if self._backfill_thread and self._backfill_thread.is_alive():
            self._backfill_thread.join(timeout=5)

    # ---------- 注册 ----------
    def _add_watch_recursive(self, path: str):
        """inotify 不递归：对目录树每个目录注册 watch（受 max_depth 限制）。"""
        for root, dirs, files in self._walk_limited(path):
            dirs[:] = [d for d in dirs if not self._excluded(d)]
            self._add_watch(root)

    def _walk_limited(self, base: str):
        """os.walk 封装：max_depth>0 时剪枝（与 scanner._walk 同语义，1-based）。

        深度语义（1-based）：base 本身（depth 0）恒全量；max_depth=N → base+下
        N-1 层子目录；depth>=N 的目录自身文件与子树都不 yield。
        max_depth=0 → 无限深。例：1=仅 base 本身；2=base+1 层子目录。
        """
        if self.max_depth <= 0:
            yield from os.walk(base)
            return
        base_depth = base.rstrip("/").count("/")
        for root, dirs, files in os.walk(base):
            depth = root.rstrip("/").count("/") - base_depth
            if depth >= self.max_depth:
                # 1-based：与 scanner._walk 同款剪枝——自身文件与子树都不索引
                dirs[:] = []
                files[:] = []
            yield root, dirs, files

    def _add_watch(self, path: str):
        try:
            wd = self._libc.inotify_add_watch(
                self._fd, path.encode("utf-8"), WATCH_MASK
            )
            with self._lock:
                self._wds[wd] = path
                self._path_to_wd[path] = wd
        except Exception as e:
            logger.warning("inotify_add_watch 失败 %s: %s", path, e)

    def _excluded(self, name: str) -> bool:
        # 判据收敛到 excludes.is_excluded（与 scanner / 索引回收共用）
        return is_excluded(name, self.excludes)

    # ---------- 读事件 ----------
    def _read_loop(self):
        while not self._stop.is_set():
            try:
                self._drain_events()
            except BlockingIOError:
                pass
            except OSError as e:
                if e.errno == 11:  # EAGAIN
                    pass
                else:
                    logger.warning("inotify 读循环异常: %s", e)
                    # fd 致命损坏（EBADF）→ 自动重建并重注册；重建失败仅记日志——
                    # scanner 全盘扫描仍是最终兜底
                    if e.errno == 9:  # EBADF
                        self._rebuild()
            except Exception as e:
                logger.warning("inotify 读循环异常: %s", e)
            # 合并窗口：每次处理后停顿，窗口内事件合并为一批
            time.sleep(MERGE_WINDOW)

    def _rebuild(self):
        """重建 inotify fd + 重新注册全部 watch（EBADF 恢复）。"""
        if self._stop.is_set():
            # ★ 2026-09-23：停止后**绝不重建**。`stop()` 会 close(fd)，而读循环可能
            # 正卡在 `os.read` 上 → 醒来拿到 EBADF(errno 9) → 走到这里重建 fd 并
            # 重新注册 watch —— 于是 watcher 在 stop() 之后复活：PUT settings 里刚
            # 被回收的文件，会被这个「复活」的 watcher 按旧配置重新 offer 回库。
            # 停止位已置时不重建，读循环随下一轮 while 判断退出。
            logger.info("已停止，跳过 inotify fd 重建")
            return
        try:
            if self._fd >= 0:
                try:
                    self._libc.close(self._fd)
                except Exception:
                    pass
            self._fd = self._libc.inotify_init1(os.O_NONBLOCK)
            if self._fd < 0:
                raise OSError("inotify_init1 失败")
            with self._lock:
                self._wds.clear()
                self._path_to_wd.clear()
            for d in self.index_dirs:
                self._add_watch_recursive(d)
            logger.info("inotify fd 已重建并重注册")
        except Exception as e:
            logger.warning("inotify fd 重建失败（退化靠 scanner 兜底）: %s", e)

    def _drain_events(self):
        """单次非阻塞读 + 解析 + 处理（每次调用最多处理一缓冲批次）。

        IN_Q_OVERFLOW（队列溢出）→ 事件已部分丢失，触发全量补扫兜底（幂等 hash 跳过），
        不静默丢变更。
        """
        try:
            data = os.read(self._fd, 65536)
        except BlockingIOError:
            return
        if not data:
            return
        pending: dict[str, set] = {}
        overflow = False
        off = 0
        while off < len(data):
            wd, mask, cookie, evlen = struct.unpack_from("iIII", data, off)
            # 事件布局: [wd 4B][mask 4B][cookie 4B][len 4B][name evlen 字节]
            # name 起点 off+16，终点 off+16+evlen；事件总长 16+evlen
            name = data[off + 16 : off + 16 + evlen].split(b"\x00", 1)[0].decode("utf-8", "replace")
            off += 16 + evlen
            if mask & IN_Q_OVERFLOW:
                overflow = True
                continue
            if not name:
                continue
            path = self._wd_path(wd, name)
            # 2026-09-18：排除判定必须用**完整路径**，不能用 basename。
            # 排除规则常按父目录名写（如 `skipme`），只拿文件名比对时规则完全失效：
            # 事件路径会把 skipme/ 下的文件重新索引回来——reconcile 刚按规则回收，
            # watcher 又加回来，排除形同虚设（B5 的同类漏洞，reconcile 侧已修，
            # 事件侧此前漏了）。is_excluded 已按路径分量匹配，传全路径即同语义。
            if not path or self._excluded(path):
                continue
            pending.setdefault(path, set()).add(mask)
        if overflow:
            logger.error("inotify 队列溢出，触发全量补扫兜底")
            self._backfill_all()
        if pending:
            self._process(pending)

    def _wd_path(self, wd: int, name: str) -> str | None:
        with self._lock:
            base = self._wds.get(wd)
        if base is None:
            return None
        return os.path.join(base, name) if name else base

    # ---------- 处理 ----------
    def _process(self, pending: dict[str, set]):
        """处理合并后的事件集。"""
        for path, masks in pending.items():
            try:
                self._handle(path, masks)
            except Exception as e:
                logger.warning("增量处理失败 %s: %s", path, e)
                self._mark_failed(path, e)

    def _mark_failed(self, path: str, exc: Exception):
        """增量索引失败：落库 failed + 记录错误明细（2026-09-09）。

        此前只 logger.warning 吞掉：文件连元数据都不入库 → 失败计数为 0、明细
        没有、embed 恢复看门狗（按 DB 的 failed/queued 计数触发）也看不见它，
        用户手动新建的文件在 embed 掉线期间被永久遗忘（实测复现）。

        2026-09-18：本方法**不再碰数据库**，只记错误明细（供 UI 的失败列表）。
        双重原因：
          · 原实现只改状态、不计数、不落 hash，与「失败要落 hash + 计数封顶」冲突
            ——实测读不到的文件被写成 failed / attempt_count=0 / hash 空，而扫描对
            「failed 且未达上限」的文件每轮重新排队 → 每轮重试、计数永远为 0；
          · 内容类失败现在由 `Indexer.index_file` 自己留痕（scanner worker / watcher /
            backfill 三条路径的公共入口）。此处再记一次会重复计数，且新建记录时会把
            index_file 刚落的 content_hash 清空，让幂等判定再次失效。
        """
        if not Path(path).is_file():
            return  # 已删除/非普通文件：无需落库
        # 2026-09-21：服务侧失败（embed 过载/超时/拒收）与解析失败分开标注——
        # 此前一律归 "parse"，界面上把「向量化服务过载」显示成「错误解析」。
        if isinstance(exc, PermissionError):
            kind = "permission"
        elif isinstance(exc, EmbedError):
            kind = "embed"
        else:
            kind = "parse"
        # 2026-09-23：**不给成句文案**——后端的对接口只给「事实 + 分类」，
        # 词在 locales 里按语言拼。此前这里（以及 scanner）把一句写死的中文塞进
        # reason，界面原样打印，用户切英文时那一行仍是中文（且很长）。
        # 文件名编不了码这一种给结构化标记；其余给原始异常消息（服务侧技术细节，
        # 无法翻译，保留原文反而是最有用的信息）。
        reason = str(exc)
        if isinstance(exc, UnicodeEncodeError):
            import json as _json

            from bibit_filesearch.pathutil import describe_unencodable_filename

            reason = "filenameUnencodable:" + _json.dumps(
                describe_unencodable_filename(path), ensure_ascii=False)
        scanner = getattr(self, "_scanner_ref", None)
        if scanner is not None:
            try:
                scanner.record_error(kind, path, reason)
            except Exception:  # noqa: BLE001
                logger.warning("记录错误明细失败 %s", path, exc_info=True)

    def _request_excluded_refresh(self):
        """通知 scanner：遍历侧「未纳入索引」读数可能要重算（2026-10-10）。

        ★ 为什么要挂在**每一个**增删移事件上：被排除的目录本身不注册 inotify watch
          （`_add_watch_recursive` 会剪掉命中规则的子树），所以「文件从被排除目录里
          移出来」的那一侧**收不到事件** —— 只能靠另一侧（目标目录有新文件进来、
          或目录被创建/移入）来推断。任一侧都可能先到，所以两边都打标记，由
          `scanner.recount_excluded()` 在池子静默时做一次全量重算收敛。
        """
        scanner = getattr(self, "_scanner_ref", None)
        if scanner is None:
            return
        fn = getattr(scanner, "request_excluded_refresh", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001 —— 标记失败不该影响增量索引
                logger.warning("通知重算未纳入索引读数失败", exc_info=True)

    def _handle(self, path: str, masks: set):
        is_dir = any(m & IN_ISDIR for m in masks)
        is_delete = any(m & (IN_DELETE | IN_DELETE_SELF | IN_MOVED_FROM) for m in masks)
        is_create = any(m & (IN_CREATE | IN_MOVED_TO | IN_CLOSE_WRITE) for m in masks)
        # 目录结构发生变化（新增/删除/移入/移出目录）→ 未纳入索引读数可能变，打标记。
        if is_dir and (is_delete or is_create):
            self._request_excluded_refresh()

        if is_delete:
            # 删除判定前复核目标是否仍存在——`mv` 覆盖/移动场景下 MOVED_FROM 不代表
            # 真删除（目标仍在），避免误删索引（文件与目录分支对称）。
            if os.path.exists(path):
                if is_dir:
                    # 目录被移动/覆盖：重新注册 watch + 补扫（当移动处理）
                    self._add_watch_recursive(path)
                    self._backfill_dir(path)
                else:
                    # 文件被覆盖：内容已变，重新入池
                    self._offer(path)
                return
            if is_dir:
                # 目录删除：从池中撤掉该目录下的全部行（重构：走池，不再直接删库）
                self.pool.cancel_tree(path)
            else:
                # 文件删除：从池中撤掉。
                # 用 purge 而非 cancel —— 文件真的没了，池中不该留任何痕迹。
                # 此前走 `indexer.delete_by_path` 只删行，但**正在算的那一个**不在
                # 池里（watcher 不写行），它算完会把行重新写回来 → 实测出现
                # 「文件已从磁盘删除，库中仍是 completed 并挂着 1 个块」的幽灵行
                # （全盘遍历不跑就永远不被清理）。
                self.pool.purge(path)
            if self.on_event:
                self.on_event("delete", path)
            return

        if is_dir:
            # 新目录创建/移入：inotify 不递归，必须为新目录（及其子目录）注册 watch，
            # 否则目录内文件事件丢失（关键：服务运行中放入的目录树）
            self._add_watch_recursive(path)
            # 注册后补扫目录内已有文件：cp -r 一次性拷入目录树时，
            # 文件创建事件发生在 watch 注册前（inotify 只在有 watch 的目录发事件）→ 丢失。
            # 补扫兜底：walk 目录内现有文件逐个入池（池中已有且 hash 相同的会被 offer 幂等跳过）。
            self._backfill_dir(path)
            if self.on_event:
                self.on_event("dir_watch", path)
            return

        if not Path(path).is_file():
            return  # 非目录非文件（设备等）：忽略
        if not is_create and not any(m & (IN_MODIFY | IN_ATTRIB | IN_CLOSE_WRITE) for m in masks):
            return

        # 跳过索引数据库自身（sqlite 文件，避免 WAL 变更风暴触发无限重算）
        ext = Path(path).suffix.lower()
        if Path(path).name.endswith(("-wal", "-shm")) or ext in (".db", ".sqlite", ".sqlite3"):
            return

        # 文件级增删移同样可能改变排除读数（例如文件被移出被排除的目录，或被移进
        # 一个命中规则的目录——后者事件收不到，由目录级/补扫那侧兜）。打标记即可。
        if is_delete or is_create:
            self._request_excluded_refresh()
        self._offer(path)

    def _offer(self, path: str):
        """把路径放进索引池（重构后 watcher 唯一的「干活」入口）。

        与 `indexer.index_file` 的关键差别：这里**不做解析、不算向量**，只是登记。
        真正的计算由常驻消费者（`pool.IndexWorker`）按池中顺序做。
        """
        try:
            result = self.pool.offer(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("增量入池失败 %s: %s", path, e)
            self._mark_failed(path, e)
            return
        # 事件回调保持旧形态（测试与外部观察用）：把 offer 结果映射回 index_file
        # 的 status 词表，避免调用方因重构而失配。
        mapped = {
            "added": "indexed",
            "refreshed": "updated",
            "noop": "skipped",
            "skipped": "skipped",
            "unsupported": "unsupported",
            "removed": "removed",
            "gone": "gone",
        }.get(result, result)
        if self.on_event:
            self.on_event("index", path, {"status": mapped, "offered": result})
        if result in ("added", "refreshed"):
            logger.info("增量索引 %s: %s", path, result)

    def _remove_dir_index(self, dir_path: str):
        """目录删除清理。逐字前缀（`db.files_under_dir`），不用 SQL `LIKE`——
        目录名含 `_`/`%` 时会命中兄弟目录（2026-10-08，同 pool.cancel_tree）。"""
        rows = self.db.files_under_dir(dir_path)
        for r in rows:
            self.indexer.delete_file(r["file_id"])
        logger.info("目录删除清理 %d 条: %s", len(rows), dir_path)

    def _backfill_dir(self, path: str):
        """补扫目录内已有文件（注册 watch 后调用；修复一次性拷入目录树事件丢失）。

        2026-09-18：排除判定改用**完整路径**（原来对文件只比对 basename）。
        规则写父目录名时 basename 比对不中，补扫会把刚被 reconcile 按规则回收的
        文件重新索引回来；目录自身命中规则时也应整体跳过（不进入 walk）。

        2026-09-23（重构）：改为逐个 `pool.offer()` —— 补扫**只登记，不计算**。
        此前这里同步调 `indexer.index_file(fp)`，一个目录几百个文件时会占住
        watcher 的读循环几十秒到几分钟，期间**删除事件全部排队**
        （实测：4.65MB PDF 移出知识库路径后，清理延迟 31.2 秒才完成）。
        """
        if self._excluded(path):
            return
        offered = 0
        try:
            for root, dirs, files in self._walk_limited(path):
                if self._stop.is_set():
                    # 停止期间不再继续 offer：`stop()` 要 join 本线程，而调用方
                    # （PUT settings）在这之后立刻按新规则回收 —— 若这里还拿旧
                    # 配置继续入池，刚回收的文件会被原样加回来。
                    logger.info("补扫中断（已停止）: %s", path)
                    return
                dirs[:] = [d for d in dirs if not self._excluded(os.path.join(root, d))]
                for fname in files:
                    if self._stop.is_set():
                        return
                    fp = os.path.join(root, fname)
                    if self._excluded(fp):
                        continue
                    try:
                        if self.pool.offer(fp) in ("added", "refreshed"):
                            offered += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning("补扫入池失败 %s: %s", fp, e)
                        # 2026-09-09：与事件路径一致——失败记明细，
                        # 否则补扫失败的文件对失败计数/看门狗完全不可见
                        self._mark_failed(fp, e)
            logger.info("目录补扫完成: %s（入池 %d 个）", path, offered)
        except Exception as e:
            logger.warning("目录补扫异常 %s: %s", path, e)
