"""OCR 引擎：PP-OCR（RapidOCR + onnxruntime），唯一引擎（rev 5，2026-09-28）。

背景：原实现直接调 tesseract（pytesseract）。tesseract 出厂 psm=3 **不做方向判断、
也不读 EXIF**，所以手机拍的歪图/倒图会被 OCR 成整段乱码，parser 里为此长出一整套
「EXIF 转正 + OSD 判角 + 双向打分 + 置信度门槛」的补救。换成 PP-OCR 后这些补救不再
需要（PP-OCR 自带方向分类器，按行判方向并摆正）。实测：

    样本         tesseract(psm3)      PP-OCR(cls)
    0°           命中                  命中
    180°         乱码                  命中（自动摆正）
    90°          乱码                  命中（自动摆正）
    270°         乱码                  命中（自动摆正）
    EXIF=6 jpg   乱码（不读 EXIF）      命中（按像素判方向）

rev 5 把 tesseract 整个删掉（含回退与救援）。理由与代价：保留「换引擎回退」只把
「读不出字」换成「读得出字但可能不准」，却让 OCR 故障在用户看来与「这张图本来没字」
无法区分。现在的契约是**失败就抛异常**，由 parser 记进 `ParsedDocument.error`，
indexer 据此落 failed（版本速查见 `docs/troubleshooting.md` 的 0.1.24 三行）。

推理设备：auto 时若 ep_worker 与 libspacemit_ep.so 都在，则经 SpaceMIT NPU 推理
（ep_worker 是常驻 C++ 进程，原因见 `ocr_ep` 模块注释：Python 侧建 EP 会话必然段错误，
唯一入口是 C API OrtSessionOptionsSpaceMITEnvInit）。NPU 只是加速，不是依赖：
不可用时自动跑纯 CPU，实测差异（4 张真实图，各 3 次取中位数）：

    图                CPU      NPU      提速
    bill.jpg        9.25s    3.16s    2.93x
    id_card_1.png   4.71s    1.99s    2.37x
    id_card_2.jpg   5.49s    2.39s    2.30x
    id_rotate.png   7.55s    4.41s    1.71x
    均值            6.75s    2.99s    2.26x
    识别文本：四张图 CPU 与 NPU **逐字一致**
（CPU 那轮测于 load≈3.4，非空载；NPU 不是依赖，缺了就是慢一倍。）

模型：rapidocr wheel 自带 PP-OCRv6 det/rec small + cls（32MB），**不需联网**；
实测断网（http(s)_proxy 指向黑洞端口）构造成功 2.5s，单图 3.13s，峰值 RSS 425MB。

内存：常驻约 700MB；输入长边上限 `MAX_SIDE_LEN` 的代价是**近平方级**的（见其注释
的逐尺寸与逐档实测表）。注意 det 的 arena **只涨不缩**，所以峰值取决于「这轮遇到过
的最大那张图」，而与图片张数无关。
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("bibit.parser")

# 推理设备：auto（有 NPU 用 NPU，否则 CPU）/ npu / cpu
# 注意这是**设备**级回退，不是引擎回退：换的是算力，不是识别器，文本不会因此变样。
# ★ 2026-09-24 实测（同一批真实图，NPU vs CPU）：文本逐行 100% 一致，提速 1.34~2.47×
#   走 NPU 需要 packaging/ocr-ep/ep_worker（C++ 常驻进程），原因见 ocr_ep 模块注释：
#   Python 侧的 spacemit_ort shim 建 EP 会话必然段错误（ABI 不匹配），
#   唯一可用入口是 C API OrtSessionOptionsSpaceMITEnvInit()。
OCR_DEVICE = "auto"

# 低置信度阈值（0~100）。仅用于「记录可疑」，不触发重读（见 recognize_ppocr）。
# parser.OCR_LOW_CONF 同值；两处独立定义是为了避免 parser ←→ ocr_engine 循环导入。
OCR_LOW_CONF = 50.0

# 输入长边上限（rapidocr 的 Global.max_side_len）。
# ★ 2026-09-24 首测（单图单尺寸）：这个值的**内存代价近平方级**——喂给 det 的张量
#   面积决定 det worker 的 arena。当时记下的四个点：
#
#     det 张量            det worker RSS     来源图
#     (1,3,  736,1312)    1534 MB            id_card_2.jpg  335x597
#     (1,3, 1152, 768)    1874 MB            bill.jpg       1153x770
#     (1,3, 1536, 960)    2807 MB            id_card_1.png  1532x957
#     (1,3, 1984,1312)    4600 MB            id_rotate.png  2298x1496  ← 旧默认 2000 的后果
#
#   （数字与下表差 ±20MB：arena 高水位有批间抖动。）
#
#   ⚠️ 2026-09-28 订正：那版还写着「压到 1200 后 det 约 1.5GB、推理 8.12s→4.80s」
#   与「1200/1600 的相似度 0.9508/0.9497」——那是单图单趟读数，已被下表的
#   4 图 × 8 档扫描取代，这里不再重复，以免两套数字打架。
#
# ★ 2026-09-28：出厂默认由 2000 改为 **1400**。依据是 4 张真实图 × 8 档的扫描
#   （每档独立进程，PID 级归因；合计 = det+cls+rec 三个 worker 的 VmRSS 之和）：
#
#     side  合计峰值  平均字符集重合  最差字符集   丢字并集
#     2000  4916      —              —           —          ← 旧出厂值
#     1800  4185      0.9864         0.9455      问
#     1600  3438      0.9907         0.9630      问
#     1400  2983      0.9907         0.9630      问         ← 新出厂值（−39%）
#     1200  2648      0.9861         0.9444      ×问
#     1000  2648      0.9737         0.8947      ×□
#      800  2648      0.9692         0.8929      ;×□问；
#      600  2648      0.9326         0.8448      %;×□仟务局市征整销问；
#
#   选 1400 而不是更低的档，两条判据：
#     ① 合计峰值在 1200 就触底（2648 MB = 最小的 id_card_2 自身的 1534+1015），
#        1000/800/600 一点都不再降；
#     ② 1400 与 1600 的识别质量相同（平均/最差都是 0.9907/0.9630，只丢一个字），
#        而 1200 起开始丢第二个字符、600 档把「税务局/征收」认成「仟务局市征整销」。
#   影响面：4 张里只有 id_rotate.png（2298x1496）会变，其余三张长边本就 ≤1532，
#   全程逐字不变 —— 所以这是一次**低风险**的降峰，代价可量化到「丢 1 个字符」。
#   改这个值必须配 PARSER_REVISION +1（存量图片的文本会变，content_hash 看不出）。
MAX_SIDE_LEN = 1400

_PPOCR = None            # 缓存的 RapidOCR 实例（构造一次，常驻）
_PPOCR_TRIED = False
_PPOCR_ERROR = ""
_PPOCR_DEVICE = ""       # 实际用上的设备（"npu" / "cpu"）
_PPOCR_BUILT = None      # 构造该实例时的 (OCR_DEVICE, MAX_SIDE_LEN) 快照；None = 未构造
_LOW_CONF_COUNT = 0      # 低置信度识别次数（进程级，供 describe 暴露；只看本次运行）
_LOCK = threading.Lock()


def apply_settings(ocr_cfg: dict | None = None) -> None:
    """把 `ocr.*` 配置写进本模块的全局旋钮（含 ocr_ep / parser 的几项）。

    ★ 为什么要有这个入口（2026-10-08，审查 R03）：这些全局量原先只在
      `create_app()` 里赋值，而生产启动顺序是 `main() → ocr_selfcheck() → uvicorn
      → create_app()`。自检**先**构造并缓存了 OCR 引擎（`_PPOCR_TRIED=True`），
      create_app 随后改的全局量只影响「下次构造」——引擎已经是老参数了，于是
      `ocr.device: cpu` 与自定义 `max_side_len` 在生产路径上都不生效，`/api/meta`
      还会如实报出那个陈旧值。启动路径与工厂路径必须共用同一份赋值逻辑。

    ★ 参数变了要**重建**：`_PPOCR_BUILT` 记住构造时的取值，`apply_settings` 发现
      与当前配置不符就丢掉缓存实例（下次用时按新参数重建）。否则「改了配置重启」
      仍然跑旧参数——那正是本项要修的现象。

    幂等：同样的配置重复调用无副作用（不重建、不记日志）。
    """
    global OCR_DEVICE, MAX_SIDE_LEN
    global _PPOCR, _PPOCR_TRIED, _PPOCR_ERROR, _PPOCR_DEVICE, _PPOCR_BUILT

    cfg = ocr_cfg or {}

    device = str(cfg.get("device", "auto") or "auto").strip().lower()
    if device not in ("auto", "npu", "cpu"):
        logger.warning("ocr.device=%r 不是 auto/npu/cpu，按 auto 处理", device)
        device = "auto"
    OCR_DEVICE = device

    try:
        MAX_SIDE_LEN = int(cfg.get("max_side_len", 1400) or 0)
    except (TypeError, ValueError):
        logger.warning("ocr.max_side_len=%r 不是整数，按 1400 处理", cfg.get("max_side_len"))
        MAX_SIDE_LEN = 1400

    # 推理侧的几个上限住在别的模块（ep 生命周期 / PDF 页闸），一并从同一份配置赋值。
    # 汇总在此处是为了让「配置 → 生效值」只有一条路径：create_app 与 main() 都调它。
    try:
        from bibit_filesearch import parser as _parser
        _parser.PDF_MAX_OCR_PAGES = _int_or(
            cfg.get("pdf_max_ocr_pages"), 20, "ocr.pdf_max_ocr_pages")
    except Exception as e:                           # noqa: BLE001
        logger.warning("应用 ocr.pdf_max_ocr_pages 失败：%s", str(e)[:120])
    try:
        from bibit_filesearch import ocr_ep as _ep
        _ep.WORKER_MAX_RUNS = _int_or(cfg.get("worker_max_runs"), 0, "ocr.worker_max_runs")
        _ep.WORKER_MAX_RSS_MB = _int_or(cfg.get("worker_max_rss_mb"), 0, "ocr.worker_max_rss_mb")
        _ep.WORKER_IDLE_RELEASE_SECS = _int_or(
            cfg.get("idle_release_secs"), 10, "ocr.idle_release_secs")
    except Exception as e:                           # noqa: BLE001
        logger.warning("应用 ocr_ep 旋钮失败：%s", str(e)[:120])

    if _PPOCR is not None and _PPOCR_BUILT != (OCR_DEVICE, MAX_SIDE_LEN):
        logger.info(
            "OCR 配置已变（设备 %s→%s，长边 %s→%s）：丢弃已构造的引擎，下次按新参数重建",
            (_PPOCR_BUILT or ("?", "?"))[0], OCR_DEVICE,
            (_PPOCR_BUILT or ("?", "?"))[1], MAX_SIDE_LEN,
        )
        _PPOCR = None
        _PPOCR_TRIED = False
        _PPOCR_ERROR = ""
        _PPOCR_DEVICE = ""
        _PPOCR_BUILT = None


def _int_or(val, default: int, name: str) -> int:
    """配置值 → int，非法值告警并回落默认（保证 apply_settings 永不抛）。"""
    try:
        return int(val if val is not None else default) or 0
    except (TypeError, ValueError):
        logger.warning("%s=%r 不是整数，按 %s 处理", name, val, default)
        return int(default) or 0


def _try_ppocr():
    """惰性构造 PP-OCR 引擎；失败只记一次，之后不再重试。"""
    global _PPOCR, _PPOCR_TRIED, _PPOCR_ERROR, _PPOCR_DEVICE, _PPOCR_BUILT
    if _PPOCR_TRIED:
        return _PPOCR
    with _LOCK:
        if _PPOCR_TRIED:
            return _PPOCR
        _PPOCR_TRIED = True
        try:
            # ① NPU 优先：打上 shim 后 RapidOCR 的会话由 ep_worker 承接推理
            _PPOCR_DEVICE = "cpu"
            if OCR_DEVICE in ("auto", "npu"):
                try:
                    from bibit_filesearch import ocr_ep
                    ok, why = ocr_ep.patch_rapidocr()
                    if ok:
                        _PPOCR_DEVICE = "npu"
                        logger.info("OCR 推理设备：NPU（SpaceMIT EP via ep_worker %s）", why)
                    elif OCR_DEVICE == "npu":
                        logger.warning("ocr.device=npu 但 NPU 不可用：%s（仍按 CPU 跑）", why)
                    else:
                        logger.info("OCR 推理设备：CPU（NPU 不可用：%s）", why)
                except Exception as e:               # noqa: BLE001
                    logger.warning("NPU 路径装载失败，用 CPU：%s", str(e)[:160])

            from rapidocr import RapidOCR

            # ★ 必须显式传：rapidocr 的出厂值就是 2000，但它是**包内 config.yaml**
            #   的值，不从我们这边读；只有构造参数能覆盖，所以做成显式传入。
            #   传 0/None 表示不覆盖（用 rapidocr 自己的默认）。
            _lim = int(MAX_SIDE_LEN or 0)
            _PPOCR = RapidOCR() if _lim <= 0 else RapidOCR(
                params={"Global.max_side_len": _lim})
            # ★ 构造成功即记下当时的参数：apply_settings 靠它判断「配置变了要重建」。
            _PPOCR_BUILT = (OCR_DEVICE, MAX_SIDE_LEN)
            # ★ 必须在构造**之后**再压日志等级：RapidOCR 在构造过程中会重建自己的
            #   logger/Handler（实测构造前设 WARNING 无效，构造后又回到 INFO）。
            for _n in ("RapidOCR", "rapidocr", "onnxruntime"):
                logging.getLogger(_n).setLevel(logging.WARNING)
            logger.info("OCR 引擎：PP-OCR（RapidOCR + onnxruntime）已就绪，设备=%s，"
                        "输入长边上限=%s", _PPOCR_DEVICE,
                        MAX_SIDE_LEN if int(MAX_SIDE_LEN or 0) > 0 else "rapidocr 默认(2000)")
        except Exception as e:                     # noqa: BLE001
            _PPOCR_ERROR = str(e)[:200]
            logger.warning("PP-OCR 不可用：%s", _PPOCR_ERROR)
            _PPOCR = None
    return _PPOCR


def mark_used() -> None:
    """告诉引擎「刚用它跑过一次推理」——TTL 判据（`ocr.idle_release_secs`）的输入。

    放在这一层而不是 `ocr_ep.Worker.run()`：CPU 路径没有 Worker，但它同样能提供
    「这个进程最近还在 OCR」这个事实，而这个事实才是 TTL 要的东西。
    """
    from bibit_filesearch import ocr_ep

    ocr_ep.mark_used()


def release_idle_when_idle(now: float | None = None) -> int:
    """TTL 判据成立就释放 worker（返回释放的进程数；没释放则 0）。

    与 `release_idle()` 分开，是为了让**策略**（多久算空闲）独立于**无条件释放** ——
    后者仍被停机路径（`main._release_on_shutdown`）使用，语义不变。
    判据与前置条件见 `ocr_ep.should_release_when_idle`。
    """
    from bibit_filesearch import ocr_ep

    if not ocr_ep.should_release_when_idle(now):
        return 0
    return release_idle()


def release_idle() -> int:
    """释放推理引擎占的内存。返回释放的 worker 进程数。

    调用方：空闲超时（TTL）与**停机**（`main._release_on_shutdown`）。2026-10-10 方案 A
    起运行期只有 TTL 这一条（原「池空即释放」已删）。

    只对 NPU 路径有意义：NPU 的推理在 ep_worker 子进程里，关掉子进程即归还内存
    （实测 det worker 能到 4.6 GB）。CPU 路径的会话在**本进程**内，本进程内存
    不会因此下降，所以只记一条日志、不动会话（重建 CPU 会话要多花 1s 且无收益）。

    ★ 释放后 RapidOCR 对象仍然可用：ocr_ep 用世代号让旧会话在下次 run() 时
      自动重建 worker，调用方不需要做任何事。
    """
    if _PPOCR is None:
        return 0
    if _PPOCR_DEVICE != "npu":
        logger.debug("release_idle：CPU 路径无子进程可释放（本进程会话保留）")
        return 0
    try:
        from bibit_filesearch import ocr_ep

        return ocr_ep.release_idle()
    except Exception as e:                           # noqa: BLE001
        logger.warning("释放 OCR worker 失败：%s", str(e)[:160])
        return 0


def available() -> tuple[bool, str]:
    """引擎是否真的可用（构造成功即算可用）。返回 (可用, 说明)。

    ★ 与「import 成功」不同：这里真的走一次构造。postinst 与启动自检都用它，
      否则会出现「报了就绪、实际跑不动」的谎报（实测踩过：ep_worker 不在时仍报就绪）。
    """
    eng = _try_ppocr()
    if eng is not None:
        return True, "PP-OCR 就绪（设备=%s）" % (_PPOCR_DEVICE or "?")
    return False, _PPOCR_ERROR or "PP-OCR 未构造"


def describe() -> dict:
    """引擎自述（供 /api/meta 暴露，避免只能翻日志判断「到底在跑什么」）。"""
    eng = _try_ppocr()
    info = {"engine": "ppocr", "device": _PPOCR_DEVICE or "?", "ready": eng is not None,
            "max_side_len": MAX_SIDE_LEN, "error": _PPOCR_ERROR,
            "low_conf_count": _LOW_CONF_COUNT,
            # 2026-09-28：TTL 空闲释放的三格观测 —— 「worker 到底放没放掉」看这里，
            # 不然只能翻日志（ttl 相关排查见 docs/troubleshooting.md 现象 J）。
            "idle_release_secs": 0,
            "worker_count": 0,
            "idle_age_secs": -1.0}
    # TTL 三格：CPU 路径也要报（那时 worker_count 恒 0，正是实情 —— 没有子进程）
    try:
        from bibit_filesearch import ocr_ep as _ep

        info["idle_release_secs"] = int(_ep.WORKER_IDLE_RELEASE_SECS or 0)
        info["worker_count"] = _ep.worker_count()
        info["idle_age_secs"] = round(_ep.idle_age(), 1)
    except Exception as e:                           # noqa: BLE001
        info["idle_release_secs"] = 0
        info["worker_count"] = 0
        info["idle_age_secs"] = -1.0
        logger.debug("TTL 观测格填充失败：%s", str(e)[:120])
    if _PPOCR_DEVICE == "npu":
        try:
            from bibit_filesearch import ocr_ep
            ok, why = ocr_ep.available()
            info["npu_worker"] = why if ok else ""
            info["npu_ok"] = ok
        except Exception as e:                       # noqa: BLE001
            info["npu_ok"] = False
            info["npu_worker"] = "探测失败：%s" % str(e)[:120]
    return info


def recognize_ppocr(img) -> tuple[str, float]:
    """PP-OCR 识别。返回 (文本, 平均置信度 0~100)。失败**抛异常**，由调用方处理。

    ★ rev 5：这里不再有 `disable()`。「引擎崩了就整个停用、后续换 tesseract」是双引擎
      时代的语义；单引擎下没有可换的引擎，把异常抛出去才是诚实的——parser 会把它记进
      `ParsedDocument.error`，indexer 据此把文件落 `failed`（可查、可重试），
      而不是静默成「无文本」。

    低置信度不是错误：只计数并在日志留痕（`_LOW_CONF_COUNT` 经 describe 暴露）。
    旧方案在低 conf 时会进救援分支自动重读，那套随 tesseract 一起删除了。
    """
    global _LOW_CONF_COUNT
    eng = _try_ppocr()
    if eng is None:
        raise RuntimeError("PP-OCR 不可用：%s" % _PPOCR_ERROR)
    res = eng(img)
    mark_used()          # TTL 判据：这一步跑完就算「刚用过」（见 ocr.idle_release_secs）
    txts = list(getattr(res, "txts", None) or [])
    scores = list(getattr(res, "scores", None) or [])
    text = "\n".join(str(t) for t in txts)
    conf = 0.0
    if scores:
        try:
            conf = float(sum(float(s) for s in scores) / len(scores)) * 100.0
        except Exception:                           # noqa: BLE001
            conf = 0.0
    if text and conf < OCR_LOW_CONF:
        _LOW_CONF_COUNT += 1
        logger.info("OCR 置信度偏低（%.1f < %.1f）：%d 字，已记录不重读",
                    conf, OCR_LOW_CONF, len(text))
    return text, conf
