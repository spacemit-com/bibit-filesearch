# -*- coding: utf-8 -*-
"""把 PP-OCR 的推理搬到 SpaceMIT NPU（SpaceMITExecutionProvider）——Python 侧客户端。

背景（2026-09-24，实测）
========================

本机（K3 / riscv64）有 SpaceMIT 的 NPU 推理加速：
  · spacemit-onnxruntime 2.0.7+2      → /usr/lib/libonnxruntime.so.1.24.2+spacemit.a1
  · /usr/lib/libspacemit_ep.so.2.0.7  → SpaceMITExecutionProvider（NPU）
  · /dev/aidma_list、/dev/ai_dma      → 加速设备节点
  · onnxruntime_perf_test -e spacemit → 实测可用

但**Python 侧用不了**，实测 5 种写法全部段错误（子进程退出码 -11）：

  import spacemit_ort; ort.InferenceSession(m, providers=["SpaceMITExecutionProvider"])  → SIGSEGV
  providers=[("SpaceMITExecutionProvider", {})]                                          → SIGSEGV
  先建 CPU 会话再建 EP 会话                                                                 → SIGSEGV
  ort.register_execution_provider_library("spacemit", "/usr/lib/libspacemit_ep.so")
      → Fail: Failed to get symbol CreateEpFactories (undefined symbol)

根因 = ABI 不匹配：
  libspacemit_ep.so 导出的是**旧版 shared-provider ABI** `GetSpaceMITSharedProviderFactory`，
  而 onnxruntime 1.24.2 的 register_execution_provider_library() 只认**新版 hosted EP**
  的 `CreateEpFactories`。`import spacemit_ort` 做的只是猴子补丁（把 EP 名字塞进
  get_available_providers() 的返回列表并 hook initialize_session），所以
  `get_ep_devices()` 里其实**只有 CPUExecutionProvider** ——「列表里有 EP 名字」是假象。

可用入口 = C API `OrtSessionOptionsSpaceMITEnvInit()`（声明 /usr/include/spacemit_ort_env.h，
符号在 libspacemit_ep.so）。它在 env init 阶段把 EP 装进 ORT 工厂注册表，之后建会话即可用
NPU；**不要再 AppendExecutionProvider("SpaceMITExecutionProvider")**，会抛
`Unknown provider name`。

因此本模块用一个常驻 C++ 进程（`packaging/ocr-ep/ep_worker`，源码同目录）承接推理：
Python 只做前处理/后处理，张量按 stdin/stdout 二进制协议转发给 worker。

用法
====

    from bibit_filesearch import ocr_ep
    ocr_ep.patch_rapidocr()                 # 让 RapidOCR 的会话走 NPU
    from rapidocr import RapidOCR
    eng = RapidOCR()                        # 之后所有推理都在 NPU

或直接看可用性：

    ocr_ep.available()   # → (True, "/opt/.../ep_worker") 或 (False, "原因")

回退：worker 不存在 / 启动失败 / 推理报错，一律**自动退回纯 CPU**（不抛给调用方），
所以 OCR 功能不会因为 NPU 不可用而挂掉。
"""
from __future__ import annotations

import logging
import os
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger("bibit.parser")

# worker 可执行文件位置
WORKER_CANDIDATES = (
    Path("/opt/bibit-filesearch/ocr-ep/ep_worker"),          # deb 安装位置
    Path(__file__).resolve().parents[2] / "ocr-ep" / "ep_worker",          # 源码树（构建产物）
    Path(__file__).resolve().parents[2] / "packaging" / "ocr-ep" / "ep_worker",  # 源码树（仓库内）
    Path("/usr/lib/bibit-filesearch/ep_worker"),
)

# ---------------------------------------------------------------------------
# ★ 2026-09-24：worker 生命周期 —— 为什么必须「用完就退」
#
# 实测（本机 riscv64）ONNX Runtime 的 arena 是**高水位、不归还**的：
#
#   LOAD det               →  RSS    45 MB   （det 模型本体才 9.5 MB）
#   RUN (1,3, 320, 320)    →  RSS   214 MB
#   RUN (1,3, 640, 640)    →  RSS   700 MB
#   RUN (1,3, 960, 960)    →  RSS  1545 MB
#   RUN (1,3,1280,1280)    →  RSS  2913 MB   ← 随张量面积近似平方增长
#   同尺寸连跑 30 次        →  RSS  恒定 1477 MB（**不是泄漏**，是池子撑开后不缩）
#   跑完静置 6s            →  2913 MB 一点不降
#
# 换成真实图，喂给 det 的张量尺寸（rapidocr 默认 Global.max_side_len=2000 只压长边）：
#
#   id_card_2.jpg  335x597   → det (1,3, 736,1312) → det worker  1534 MB
#   bill.jpg      1153x770   → det (1,3,1152, 768) → det worker  1874 MB
#   id_card_1.png 1532x957   → det (1,3,1536, 960) → det worker  2807 MB
#   id_rotate.png 2298x1496  → det (1,3,1984,1312) → det worker  4600 MB
#
# 即「**一张大图就把 det worker 永久抬到 4.6 GB**」，之后所有小图都白占着。
# 三个 worker（det/cls/rec）实测峰值合计约 6.1 GB，在 16 GB 的 K3 上 = 31% × 3。
#
# ★ 为什么不是「给 worker 设个内存上限」：ORT 的 arena 没有总量上限参数，
#   SpaceMIT EP 那侧也没有公开的 cap 接口（读过 libspacemit_ep.so.2.0.7 的字符串，
#   只有 reserve_os_memory / "memory check failed" / "Workspace buffer is null"）。
#   能做的只有「缩输入」和「用完就退」，前者见 ocr_engine 的 MAX_SIDE_LEN。
#
# 所以这里给两条释放路径：
#
#   ① 空闲超时释放（主）：`pool.IndexWorker._settle()` 每次空转都会问「距上次真正 OCR 是否
#      已满 `WORKER_IDLE_RELEASE_SECS`」，满了就调 `release_idle()` —— 关掉全部 worker。
#      ⚠️ 2026-10-10 方案 A：原「一次扫描把索引池跑空就释放」那条**已删**（它认的是「池空」，
#      既不代表不会再用了、也不代表内存贵，实测让零星到达的相邻图每张都重建）。现在运行期
#      只剩这一条时间判据。
#      重建代价实测：进程启动 + LOAD 一个模型 = 0.22~0.32s、空载 RSS 45 MB；
#      三个模型全建 ≈ 1s。拿 1s 换 GB 级常驻内存。
#   ② 次数上限释放（兜底）：单个 worker 跑满 `WORKER_MAX_RUNS` 次 RUN 就退休。
#      防的是「一次扫描里有几千张图」，高水位会一直停在最贵那张的水平上。
#
# ★ 实现上的坑：EPSession 在 RapidOCR **构造期**就把 worker 抓住并长期持有，
#   所以「关掉 worker」必须让持有者知道自己手上的 worker 已经死了 —— 否则下一次
#   run() 会往一个已关闭的管道里写（BrokenPipeError）。
#   做法：每个新建 EPSession 记下当时的 `_EPOCH`；release 时 `_EPOCH += 1`，
#   旧 session 的 epoch 失配 → 下次 run() 自动重建 worker 并重打 epoch。
# ---------------------------------------------------------------------------
_EPOCH = 0

# 单个 worker 的 RUN 次数上限（0 = 不限制）。run 满即退休重建，防高水位钉在最高点。
WORKER_MAX_RUNS = 200

# 单个 worker 的 RSS 上限（MB，0 = 不限制）。跑完一次 RUN 就查一次自己的常驻内存，
# 超了即退休，下次取用时重建。
#
# ★ 为什么需要它（2026-09-28 实测，本机 4 张真实图）：ONNX Runtime 的 arena 随**输入
#   张量面积**上涨且不归还，一张 2298x1496 的图就把 det worker 永久抬到 4.6 GB——
#   而空闲超时释放（`release_idle` 那条）要等「距上次 OCR 满 TTL 秒」才触发，在那之前
#   这 4.6 GB 一直占着：
#
#     构造后 未推理        det   45 MB   cls 34 MB   rec 55 MB   合计  134 MB
#     id_card_2.jpg       det 1534 MB              rec 1015 MB   合计 2648 MB
#     bill.jpg            det 1875 MB              rec 1039 MB   合计 3013 MB
#     id_card_1.png       det 2809 MB              rec 1048 MB   合计 3956 MB
#     id_rotate.png       det 4616 MB              rec 1051 MB   合计 5766 MB  ← 峰值
#
#   det 占 80%。`WORKER_MAX_RUNS` 只能按**次数**兜（且要跑满 200 次才动），对
#   「一张大图」这种情形完全无感 —— 这张图只贡献 1~4 次 RUN。
#
# ★ 默认 1500 MB 的由来：rec 稳定在 ~1050 MB（跟最长一行走）、cls 恒定 99 MB，
#   只有 det 会越过 1500。所以这个阈值**只挑大图的 det**，小图批与小模型都不触发重建。
# ★ 代价是重建（进程 + LOAD ≈ 0.3s/模型），只在跨谷时付；换来的是「大图处理完就归还」，
#   而不是「整批都占着」。
# ★ 它不改变识别文本：重建的是进程，输入缩放（`ocr.max_side_len`）没变。
WORKER_MAX_RSS_MB = 0

# 空闲多久就放掉全部 worker（秒；0 = 关，即只靠 worker_max_rss_mb 与 worker_max_runs）。
#
# ★ 2026-09-28 新增，为什么需要它：释放原本只有一个触发点 —— `pool.IndexWorker._settle()`
#   判「索引池排空 ∧ 遍历线程结束」。这是**批次级**判据，与「内存什么时候贵」无关：
#     · 峰值出现在这批最大的那张图跑完的瞬间，而释放要等整个池排空；
#     · 大目录 walk 慢、watcher 增量一次只来一张 —— 后者**永远不排空**，于是高水位一直占着。
#   改成「与空闲时长绑定」就能覆盖这两种情形：最后一张图处理完 N 秒后归还。
#
# ★ 2026-10-10（方案 A 收敛）：那条「池空即释放」的批次路径**已删**（见 pool._settle）。
#   它认的是「池空」，既不代表不会再用了、也不代表内存贵，实测会让零星到达的相邻图
#   （间隔 < TTL）每张都重建一遍 worker。于是本值成为**运行期唯一的释放触发点**：
#   只有「距上次真正 OCR 满 TTL 秒」才放。内存真贵时的及时归还交给 worker_max_rss_mb。
#   出厂值由 60 降到 **10**：池空不再抢先释放后，TTL 要自己兜住「孤立单张 / 增量后无活」
#   这类尽快归还；10s 是「不把每张图都重建（相邻图间隔通常 < 10s）」与「尽快归还」的折中。
#
# ★ 代价（本机实测，3 趟中位数，riscv64 + NPU）：
#     warm_ocr（引擎热着）            2.06s
#     release_idle() 本身             0.64s
#     release 之后第一张 OCR          3.56s
#     ⇒ 净代价 rebuild_overhead       **1.50s**（≈ 1.5× 单次 OCR 的 70%）
#     逐模型 spawn+LOAD：det 0.20s / rec 0.24s / cls 0.17s（三者 ≈ 0.61s，其余是引擎侧重建）
#   所以本值不能太小：扫一个几千张图的大目录时，若设成几秒且相邻两张间隔又大于它，
#   每张图都要多付 1.5s。出厂 10s 的依据：同一批图的相邻两张间隔通常 < 10s（不会命中），
#   而「扫描结束 / watcher 偶发单张」这类孤立事件会在 10s 后归还 —— 兼顾及时与不churn。
WORKER_IDLE_RELEASE_SECS = 0

_DTYPES = {
    "float32": ("f", 4), "float64": ("d", 8), "float16": ("e", 2),
    "int64": ("q", 8), "int32": ("i", 4), "int16": ("h", 2),
    "uint8": ("B", 1), "int8": ("b", 1),
}
_NP_OF = {"float32": "float32", "float64": "float64", "float16": "float16",
          "int64": "int64", "int32": "int32", "int16": "int16",
          "uint8": "uint8", "int8": "int8"}

_WORKERS: dict[str, "Worker"] = {}
# TTL 的去重：记下**上次释放时的 `_LAST_USE`**，只有「自那以后又真的用过一次」才再释放。
# ★ 为什么不用世代号（`_EPOCH`）去重：重建 worker 并不会抬 `_EPOCH`（只有统一释放才抬），
#   所以「释放过一次」之后同一世代里 TTL 再也武装不起来 —— 实测就是这样：
#   test_idle_release_is_deduped_across_heartbeat 第二次判真失败。
#   改成按使用时刻去重后，语义也更直白：**没新用它就不必再放。**
_RELEASED_AT = 0.0
_LOCK = threading.Lock()


def worker_path() -> Path | None:
    for p in WORKER_CANDIDATES:
        if p.is_file() and os.access(str(p), os.X_OK):
            return p
    found = shutil.which("ep_worker")
    return Path(found) if found else None


def available() -> tuple[bool, str]:
    """NPU 路径是否可用。返回 (可用, 说明)。"""
    w = worker_path()
    if w is None:
        return False, "未找到 ep_worker（deb 应装到 /opt/bibit-filesearch/ocr-ep/ep_worker）"
    if not Path("/usr/lib/libspacemit_ep.so").exists():
        return False, "缺少 libspacemit_ep.so（spacemit-onnxruntime 未安装）"
    return True, str(w)


class Worker:
    """常驻 ep_worker 的客户端：**一个模型一个进程**。

    ★ 为什么不能一个进程装三个模型：worker 的 LOAD 会把会话替换掉，而 PP-OCR 的
      det/cls/rec 是三个独立会话（RapidOCR 依次构造 3 个 InferenceSession）。
      共用一个进程时，最后 LOAD 的那个会顶掉前两个，det 推理被喂进 rec 模型 →
      `Got invalid dimensions for input: x ... Get 768 Expected 48`（实测踩过）。
      故按模型路径各起一个进程（3 个），与同事文档第 12 节的观察一致。
    """

    def __init__(self, path: Path):
        self.path = path
        self.proc: subprocess.Popen | None = None
        self.loaded: str | None = None
        self.providers = ""
        self.lock = threading.Lock()
        self.runs = 0          # 已完成的 RUN 次数（退休判据，见 WORKER_MAX_RUNS）
        self.retired = False   # 已判决退休：下次取用时不复用，直接关掉重建

    def _ensure(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        # stderr 用线程抽干：既保留诊断又不让管道写满把 worker 卡死
        self.proc = subprocess.Popen(
            [str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        self.loaded = None
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        logger.info("OCR NPU worker 已启动 pid=%d", self.proc.pid)

    def _rss_mb(self) -> int:
        """本 worker 进程的常驻内存（MB）。读不到就返回 0（当作不超限）。"""
        if self.proc is None:
            return 0
        try:
            with open("/proc/%d/status" % self.proc.pid) as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0

    def _drain_stderr(self):
        try:
            for raw in self.proc.stderr:
                logger.debug("ep_worker: %s", raw.decode(errors="replace").rstrip())
        except Exception:                            # noqa: BLE001
            pass

    def load(self, model_path: str) -> None:
        with self.lock:
            self._load_locked(model_path)

    def _load_locked(self, model_path: str) -> None:
        """真正做 LOAD 握手。★ 调用方**必须已经持有 self.lock**。

        为什么单独拆出来（2026-09-28 实测踩坑）：`run()` 在持有 self.lock 的区间里
        检测到 worker 已死需要重载模型，若直接调 `load()`，`load` 会再次
        `with self.lock` —— threading.Lock 不可重入，于是**自锁死**
        （现象：进程停住、CPU 0、wchan=futex_do_wait，pytest 挂死）。
        """
        self._ensure()
        self.proc.stdin.write(("LOAD\t%s\n" % model_path).encode())
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().decode(errors="replace").rstrip("\n")
        if not line.startswith("OK"):
            raise RuntimeError("ep_worker load 失败: %s" % line[:200])
        self.providers = line.split("\t", 1)[1] if "\t" in line else ""
        self.loaded = model_path

    def run(self, arr):
        """arr: numpy 数组（float32 等）→ 返回 numpy 输出数组。"""
        import numpy as np

        dtype = str(arr.dtype)
        if dtype not in _DTYPES:
            raise RuntimeError("ep_worker 不支持 dtype=%s" % dtype)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        with self.lock:
            # ★ 2026-09-28（rev 5 冒烟实测暴露）：worker 进程可能已经**死了但客户端不知道**
            #   ——release_idle() 只清 _WORKERS 并抬 _EPOCH，而已经被 RapidOCR 构造期抓住的
            #   EPSession 手上那个 Worker 对象仍在，其 self.proc 指向已关闭的管道。
            #   实测现象：整批索引跑完（其间空闲释放过 worker）之后，**同进程内再索引第一张图**
            #   会在 stdin.write 上抛 BrokenPipeError，整张图被判失败。
            #   修法：写之前确认管道还活着，不活就重新拉起并 LOAD（一次 ~0.3s）。
            # ★ 必须先把模型路径存下来：_ensure() 会把 self.loaded 置 None（新进程没模型），
            #   第一版补丁就栽在这里——_ensure() 之后 `if self.loaded:` 恒假，于是重建出来
            #   的 worker 没 LOAD 任何模型，run 直接回 `ERR no model loaded`（冒烟实测）。
            model = self.loaded
            if self.proc is None or self.proc.poll() is not None:
                logger.info("ep_worker 已不在，重新拉起并重载模型 %s", model)
                self._ensure()
                if model:
                    self._load_locked(model)
            shape = ",".join(str(d) for d in arr.shape)
            try:
                self.proc.stdin.write(("RUN\t%s\t%s\n" % (shape, dtype)).encode())
                self.proc.stdin.write(arr.tobytes())
                self.proc.stdin.flush()
                hdr = self.proc.stdout.readline().decode(errors="replace").rstrip("\n")
            except (BrokenPipeError, OSError) as e:
                # 竞态兜底：检查与写入之间进程死了。重建一次，仍失败才抛给调用方
                # （调用方是 parser，会把它记成 failed；不静默）。
                logger.warning("ep_worker 管道断（%s），重建后重试一次", str(e)[:80])
                self._ensure()
                if model:
                    self._load_locked(model)
                self.proc.stdin.write(("RUN\t%s\t%s\n" % (shape, dtype)).encode())
                self.proc.stdin.write(arr.tobytes())
                self.proc.stdin.flush()
                hdr = self.proc.stdout.readline().decode(errors="replace").rstrip("\n")
            if not hdr.startswith("OK"):
                raise RuntimeError("ep_worker run 失败: %s" % hdr[:300])
            _, shp, odt = hdr.split("\t")
            oshape = [int(x) for x in shp.split(",") if x]
            n = 1
            for d in oshape:
                n *= d
            raw = self.proc.stdout.read(n * _DTYPES[odt][1])
            out = np.frombuffer(raw, dtype=_NP_OF[odt]).reshape(oshape)
            # ★ 跑满上限就标记退休：**不在这里 close**（此刻 self.lock 还握着，
            #   而 "用完就退" 的语义允许它先把手头这次结果交出去）。
            #   真正的回收发生在下次 worker_for()，见那里的说明。
            self.runs += 1
            limit = int(WORKER_MAX_RUNS or 0)
            if limit > 0 and self.runs >= limit and not self.retired:
                self.retired = True
                logger.debug("ep_worker 跑满 %d 次，标记退休（下次取用时重建）%s",
                             limit, self.loaded)
            # 内存判据（见 WORKER_MAX_RSS_MB 的实测说明）。同样只标记，不在锁内关进程。
            rss_limit = int(WORKER_MAX_RSS_MB or 0)
            if rss_limit > 0 and not self.retired:
                rss = self._rss_mb()
                if rss >= rss_limit:
                    self.retired = True
                    logger.info("ep_worker RSS %d MB ≥ 上限 %d MB，标记退休（下次取用时重建）%s",
                                rss, rss_limit, self.loaded)
            return out

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:                        # noqa: BLE001
                self.proc.kill()


def worker_for(model_path: str) -> Worker:
    """取（或建）指定模型的 worker（整个进程内每模型一个）。

    到手的 worker 若已退休，**先关掉再重建** —— 这是「用完就退」的回收点。
    回收放在这里而不是 Worker.run() 里，是为了不把 close() 的等待塞进推理路径：
    RapidOCR 一次识别要串行调 det→cls→rec 多次 run()，中途关进程会让这一张图的
    推理整体多等一个进程退出（实测 QUIT + wait 约 0.1~0.3s）。放到取用点后，
    代价落在「下一张图的第一次 run」，且只有跨批次时才付。
    """
    with _LOCK:
        w = _WORKERS.get(model_path)
        if w is not None and w.retired:
            w.close()
            w = None
        if w is None:
            ok, why = available()
            if not ok:
                raise RuntimeError(why)
            w = Worker(worker_path())
            w.load(model_path)
            _WORKERS[model_path] = w
    return w


_LAST_USE = 0.0          # 最后一次真正跑过推理的单调时刻（TTL 判据）


def mark_used() -> None:
    """记下「刚刚用过引擎」。调用点：`ocr_engine.recognize_ppocr` 成功跑到引擎那一步。

    放在 ocr_engine 侧而不是 Worker.run()，是因为要覆盖**每一条**使用路径：
    CPU 路径没有 Worker，但它同样提供得到「这个进程最近还在 OCR」这个事实。
    """
    global _LAST_USE
    _LAST_USE = time.monotonic()


def should_release_when_idle(now: float | None = None) -> bool:
    """空闲超过 `WORKER_IDLE_RELEASE_SECS` 且手上真有 worker → 该释放了。

    三条前置，缺一条都返回 False：
      · TTL > 0（默认 0 = 关，不改变现有行为）
      · 自上次释放以来又真的用过一次（否则 2s 心跳会把同一件事反复通知上来）
      · 手上确实还有 worker（没有就不必通知任何人）
    另外要求 `_LAST_USE > 0`：从没跑过 OCR 时不释放（构造引擎本身不算「用过」，
    但那种情形下有 worker 且没跑过，说明正在第一次构造的路上，别去打断它）。
    """
    if not WORKER_IDLE_RELEASE_SECS or WORKER_IDLE_RELEASE_SECS <= 0:
        return False
    if not _WORKERS:
        return False
    if _LAST_USE <= 0:
        return False
    if _LAST_USE <= _RELEASED_AT:
        return False        # 自上次释放以来没有新的使用 —— 再放一次没有意义
    now = time.monotonic() if now is None else now
    return (now - _LAST_USE) >= WORKER_IDLE_RELEASE_SECS


def worker_count() -> int:
    """当前活着的 worker 进程数（TTL 生效与否的观测点，经 /api/meta 暴露）。"""
    with _LOCK:
        return sum(1 for w in _WORKERS.values() if w.proc is not None and w.proc.poll() is None)


def idle_age() -> float:
    """距上次真正用过引擎过了多少秒（从没跑过则返回 -1）。"""
    if _LAST_USE <= 0:
        return -1.0
    return time.monotonic() - _LAST_USE


def release_idle() -> int:
    """关掉全部 worker 并让在飞 session 失效。返回释放的进程数。

    ★ 为什么要动 `_EPOCH`：worker 是被 RapidOCR 持有的 EPSession 抓住的，
      只关进程不通知持有者，下一次 run() 就写进已关闭的管道。置高 epoch 后，
      每个旧 session 在下次 run() 时发现 epoch 失配 → 自己重新取一个 worker
      （重建约 0.3s/模型），所以调用方不需要做任何事。

    两条调用路径：
      · 空闲超时（TTL）——`pool.IndexWorker._settle` → `_release_ocr_workers_when_idle`
        → `should_release_when_idle` 判真后才调；**运行期唯一的释放触发点**（2026-10-10
        方案 A：原「池空即释放」那条已删）。
      · 停机——`main._release_on_shutdown`（lifespan shutdown + 信号兜底，幂等）。
    另有两条退休判据绕过本函数、在**下次取用**时生效：`WORKER_MAX_RUNS`（按次数）、
    `WORKER_MAX_RSS_MB`（按内存）。见 `EPSession._worker`。
    """
    global _EPOCH, _RELEASED_AT
    with _LOCK:
        n = len(_WORKERS)
        for w in _WORKERS.values():
            w.close()
        _WORKERS.clear()
        _EPOCH += 1
        _RELEASED_AT = _LAST_USE      # TTL 去重：记下「放到哪一次使用为止」，见 should_release_when_idle
    if n:
        logger.info("OCR NPU worker 全部释放（%d 个），下次 OCR 时按需重建", n)
    return n


def close_all():
    """同 `release_idle()`（保留旧名）。**真正的调用者**在 `main._release_on_shutdown`。

    ⚠️ 2026-09-28 订正：此前这个函数没有调用者（docstring 写着「供服务停止时调用」，
    但全仓 grep 不到），服务停机只靠 `ep_worker` 读到管道 EOF 后自己退 —— 中间有窗口，
    systemd `KillMode=mixed` 下那几个进程还能占着 GB 级 arena。
    现在停止路径有两道：FastAPI lifespan 的 shutdown（主）+ `main()` 的信号处理（兜底），
    都经 `main._release_on_shutdown()`，幂等。
    """
    return release_idle()


# ---------------------------------------------------------------------
# RapidOCR 会话 shim：模仿 onnxruntime.InferenceSession 的接口，
# 但把 run() 转发给 worker。
# ---------------------------------------------------------------------
class _Meta:
    def __init__(self, custom_metadata_map):
        self.custom_metadata_map = custom_metadata_map


class _IO:
    def __init__(self, name):
        self.name = name


class EPSession:
    """替代 InferenceSession。run() 走 NPU worker；元数据（字表）从 CPU 会话读一次。

    ★ 为什么字表要 CPU 读：rec 模型的 `character` 内嵌在 ONNX 元数据里，worker 协议
      只回张量。CPU 建一次元数据会话（不推理）开销很小，与同事文档第 7.2 节
      观察到的「metadata EP: CPU」是同一件事。
    """

    def __init__(self, model_path: str, sess_options=None, providers=None, **kwargs):
        import onnxruntime as ort

        self.model_path = model_path
        # 元数据（输入输出名 + rec 模型内嵌的 6623 字表）用 CPU 会话读一次，不做推理。
        # 开销很小；worker 协议只回张量，字表必须从 ONNX 元数据拿。
        cpu = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._inputs = [_IO(i.name) for i in cpu.get_inputs()]
        self._outputs = [_IO(o.name) for o in cpu.get_outputs()]
        self._meta = _Meta(dict(cpu.get_modelmeta().custom_metadata_map))
        del cpu

        # 记下建会话时的世代号 —— 见 release_idle 的说明
        self._epoch = _EPOCH
        self._w = worker_for(model_path)

    def _worker(self) -> "Worker":
        """取当前可用的 worker：世代号变了（被释放过）或手上这个已退休就重新取一个。

        ★ 2026-09-28 实测修 **死旋钮**：此前这里只比 `_EPOCH`，而 `_EPOCH` 只在
          `release_idle()` 里 +1。`WORKER_MAX_RUNS` 把 `w.retired` 置真之后，
          消费这个标记的唯一位置是 `worker_for()` —— 而 `worker_for()` 在本方法
          之外没人调。于是 worker 一旦被 EPSession 持有就**永不退休**：
          真机探针把上限设成 2、连识别 4 次同一张图，det/cls/rec 三个 worker 的
          pid 全程一个都没变（`retired` 全是 True，runs 涨到 16）。
          即代码注释里承诺的「防一次扫描几千张图」这条兜底**当时并不存在**。
          现在多一个判据：手上的 worker 退休了就换（换的代价 = 重建 + LOAD ≈ 0.3s）。
        """
        global _EPOCH
        if self._epoch != _EPOCH or getattr(self._w, "retired", False):
            self._w = worker_for(self.model_path)
            self._epoch = _EPOCH
        return self._w

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_modelmeta(self):
        return self._meta

    def get_providers(self):
        return ["SpaceMITExecutionProvider(cpp worker)"]

    def run(self, output_names, input_feed):
        arr = None
        for v in input_feed.values():
            arr = v
            break
        # 每次都走 _worker()：它负责「世代号失效 / worker 已退休」时换一个新的
        out = self._worker().run(arr)
        return [out]


def patch_rapidocr() -> tuple[bool, str]:
    """让 `rapidocr` 用我们的 NPU 会话。返回 (是否打上, 说明)。

    手法：替换 `rapidocr.inference_engine.onnxruntime.main.InferenceSession` 这个名字。
    只在 rapidocr 真的 import 成功时动手；打不上不影响功能（退回 CPU）。
    """
    ok, why = available()
    if not ok:
        return False, why
    try:
        from rapidocr.inference_engine.onnxruntime import main as ort_main
    except Exception as e:                          # noqa: BLE001
        return False, "rapidocr 不可用: %s" % str(e)[:120]
    if getattr(ort_main, "_bibit_ep_patched", False):
        return True, "已打过"
    ort_main.InferenceSession = EPSession            # type: ignore[assignment]
    ort_main._bibit_ep_patched = True                # type: ignore[attr-defined]
    return True, str(worker_path())
