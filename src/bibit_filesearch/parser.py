"""文件解析：按类型提取文本与作者信息。

类型覆盖（与前端契约 fileTypes 枚举对齐）：
- 文本类: txt md docx pdf
- 代码类: py js java cpp c cs go rs ts sql sh json
- 图片类: png jpg jpeg gif webp bmp svg（PP-OCR 取图中文字，见 `_ocr_image`）
- 表格/演示: xlsx pptx
- 其他: 其余扩展名（仅元数据，无文本）
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("bibit.parser")

# 解析能力版本：解析器新增「能读到以前读不到的内容」时 +1。
#
# 用途：重扫时识别「按旧能力落成 unsupported、但类型其实在索引格式内」的文件并强制
# 重解析一次。**必须有这个标记**——indexer 的 content_hash 幂等会让这类文件永远停在
# unsupported：hash 未变 → 判定 skipped → 块数为 0 → 又被标回 unsupported，新解析
# 代码一次都跑不到（2026-09-17：docx 文本框修复后，老库里误判的文件不会自愈）。
#
# 只在「解析器能读的东西变多了」时 +1；纯重构/改错别字不动它。
# 1 = 起算点（0 表示从未跑过，即升级前落库的数据）
# 2 = OCR 预处理（2026-09-24）：方向转正（EXIF + tesseract OSD）——歪着的图片/图片型 PDF
#     以前 OCR 出的是倒过来的乱码，等于「索引里没有那些字」。
# 3 = OCR 判据修正 + 二级救援（2026-09-24 当天，与 rev 2 同批未发版）。
# 4 = OCR 引擎换成 PP-OCR（2026-09-24，见 `ocr_engine`）：换的是**识别器本身**。
#     已入库的图片都是 tesseract 读出来的文本，content_hash 看不出差异 → 必须重排一次。
# 5 = 引擎唯一化（2026-09-28）：**移除 tesseract** 与 rev 2/3 那套方向补救
#     （EXIF 转正 + OSD 判角 + 双向打分 + 置信度门槛）。理由：PP-OCR 自带方向分类器，
#     rev 2/3 补的是「tesseract psm=3 既不判方向也不读 EXIF」这个 tesseract 专属缺陷。
#     ★ 实测（GATE-0.1，2026-09-28）：在真实 _ocr_image 上摘掉 EXIF 转正，四张真图
#       （id_card_1/2、bill.jpg、id_rotate.png）文本逐张完全一致、且都不需要救援 →
#       引擎路径与 EXIF 转正**等价**，删掉不损失任何文本。
#     ★ 但合成样本指出边界：Orientation=3（180°）时引擎按行判方向会重排行序
#       （文本集合相同、行序不同），且 EXIF 只是被忽略、不是被应用。所以
#       **EXIF 转正保留**（成本实测 0.35s/2298px 一张），它才是方向标记的权威解释。
#     rev 5 的重排面与 rev 4 相同（都是「已入库图片的文本要重取」），
#     仍靠 `reconcile_parser_revision` + `db.requeue_ocr_text(image_types)`。
# 6 = OCR 输入长边上限 2000 → 1400（2026-09-28，`ocr_engine.MAX_SIDE_LEN`）。
#     ★ 为什么改个缩放要重排：这个常量直接决定送进 det 的张量尺寸，检测框会动、
#       行切分与个别小字会变（实测 4 图 × 8 档：3 图逐字不变，1 张 2298x1496 的
#       图丢 1 个字符）。content_hash 只按文件算，看不出文本变了 → 存量图片必须重取，
#       否则老库里那张图永远留着 2000 时代的文本、与新库不一致。
#     重排面与 rev 4/5 相同（已入库图片），仍靠 `reconcile_parser_revision` +
#     `db.requeue_ocr_text(image_types)`。
# 7 = PDF 解析改为两遍 + 逐页判据 + 预算早停（2026-09-29）。
#     ★ 为什么必须重排：解析**策略**变了，同一份 PDF 抽出来的文本不一样了 ——
#       ① 判据从"整份无文本层才 OCR"改成"逐页薄页 OCR"：混合型 PDF
#          （封面带字 + 正文扫描）以前一张图都不 OCR，现在会 OCR，文本**变多**；
#       ② 预算早停 + 每文件 OCR 页数上限（20）：超出预算的 PDF 现在**只处理前若干页**，
#          文本**变少**（但被截掉的部分本来也进不了索引，因为 indexer 只留前 64 块）。
#     content_hash 是文件字节的 hash，两处变化它都看不出来 → 不重排的话存量 PDF
#     永远停在旧文本上。重排面：类型为 pdf 的**全部**文件（不只是"图片型"）——
#     混合型 PDF 在旧代码下 pdf_images 为空、`requeue_ocr_text` 的判据命中不到它，
#     所以 rev 7 走 `db.requeue_all_pdfs()`（见那里的说明）。
# 8 = 文本快速路径加了「预算窗口内不得有带图薄页」的闸（2026-10-08，审查 R04）。
#     ★ 为什么必须重排：改前只按**全篇字数**判「够预算就零 OCR」，于是前置扫描页
#       （正文全在图里、文本层为空）在后面的文本页把总字数撑够时被整页放过 ——
#       那些字从来没进过索引。改后这类 PDF 会 OCR 窗口内的带图薄页，文本**变多**。
#       实测（真文件 Theartofmulticore.pdf）：封面页 1125x1388 的图里是书名与作者
#       （`THEART / MULTIPROCESSOR / PROGRAMMING / Maurice Herlihy & Nir Shavit`），
#       改前 0 次 OCR、那行字不在索引里；改后 1 次 OCR，文本含书名与作者，耗时
#       2.85s → 9.24s（多付那次 OCR 的 7.09s，本就够不到 64 块而白跑）。
#       纯文本 PDF（窗口内无带图薄页）仍然零 OCR，529 页的书整体仍是秒级。
#     重排面与 rev 7 相同（全部 PDF），仍走 `db.requeue_all_pdfs()`。
# 9 = docx 内嵌图 OCR（2026-10-09）。与 PDF「逐页薄页判据」同构，粒度换成**段落**：
#     一张内嵌图所在段落没有文字时，对该图提 OCR，把文本内联到该段位置。判据是
#     「和 PDF 用相同的方式处理」——PDF 按页判薄、docx 按段判薄，都不看整份文档。
#     ★ 为什么必须重排：改前 docx 只要 body 抬不出文字就落 unsupported（例：一份
#       只有一张观测记录图的 docx，文本全在图里），而 content_hash 是文件字节的 hash、
#       看不出 OCR 结果 → 不重排的话这些文件永远停在 unsupported，新解析代码一次都
#       跑不到（与 rev 7 的坑同源）。
#     重排面 = 全部 docx（不只那些落 unsupported 的：一份「正文 + 段内插图」的 docx
#       此前根本没有段内图文本，重排后文本**变多**，故走 `db.requeue_all_docx()`，
#       与 rev 7 对 PDF 的处理同理——判据直写「所有 docx」而不为省几次重索引漏掉
#       混合型这一最大受影响面）。
PARSER_REVISION = 9

# 低置信度阈值（0~100）：低于它在 raw_meta 里标记可疑，不再自动重读。
# 旧方案这里是「健康闸 + 两级救援」，已在 rev 5 删除——引擎自带方向分类器后，
# 那套机器治的是 tesseract 专属的毛病。单引擎下 conf 是唯一质量信号，只做记录。
OCR_LOW_CONF = 50.0

# 单个 PDF 最多对多少页做 OCR（0 = 不限）。2026-09-29。
#
# 为什么需要：逐页判据（"这页文本薄 → 对这页 OCR"）把"整本扫描书"从"不 OCR"翻成
# "每页都 OCR"。实测一份 511 页的扫描书（Python源码剖析.pdf）单页 5.2~7.5s，
# 不做闸就是 43~64 分钟的 NPU 占用，后面所有文件排队。
#
# 为什么是 20：早停（min_chars，见 `_parse_pdf`）通常在后 10~40 页内就攒够 64 块，
# 所以正常情况下根本碰不到这个闸；它只在**病态输入**上生效 —— 前面大量页是
# 无文字图页/空白页，OCR 也读不出东西，于是要一直往后翻。20 页 ≈ 2 分钟封顶，
# 与 `ocr.idle_release_secs` 同量级，不会把 OCR worker 长时间钉在 GB 级。
#
# 注：这是**每文件**的闸，不约束总量（与 `scan.max_chunks_per_file` 同一性质）。
PDF_MAX_OCR_PAGES = 20

# 单个 docx 最多对多少张内嵌图做 OCR（0 = 不限）。2026-10-09。
#
# 与 `PDF_MAX_OCR_PAGES` 同性质、同量级，只是粒度从「页」换成「图」：docx 没有页，
# 每张内嵌图落在自己的段落里，一张图 = PDF 里的一页。
# 为什么需要：段落薄判据（"这图所在段落没字 → 对这张图 OCR"）把「纯图 docx」从
# 「不 OCR」翻成「逐图 OCR」。一份 100 张插图的文档不做闸就是 10 分钟起的 NPU 占用，
# 后面所有文件排队。20 张 × ~3s(NPU) ≈ 1 分钟封顶。
DOCX_MAX_OCR_IMAGES = 20

# 单页文本少于此字符数 → 判为"薄页"，对该页提内嵌图 OCR（2026-09-29）。
#
# 为什么是 40：实测标定（本机 5 份真实 PDF、1240 页的逐页字数分布）。
#   · 正常正文页：中位 2035 字/页（《The Art of Multiprocessor Programming》528 页）
#     —— 离 40 有 50 倍余量，绝不会被误判；
#   · 需要判成"薄"的页：封面（0~37 字）、章节扉页（34）、"本页留白"（12）、
#     图表题注页（<40）；
#   · 关键反例（为什么不能更高）：《Python源码剖析》的扫描页抽出来是 0 字，
#     而它的版权页 99 字 —— 阈值只要 < 99 就能把"有 99 字的版权页"与
#     "0 字的扫描页"分开，这正是 rev 7 要修的漏检。
# 这个阈值只影响"要不要对某页 OCR"，不改变任何已有文本。
PDF_THIN_PAGE_CHARS = 40

# 前端契约枚举（开发计划.md 2.1 补充语义）
TEXT_TYPES = {"txt", "md", "docx", "pdf"}
CODE_TYPES = {"py", "js", "java", "cpp", "c", "cs", "go", "rs", "ts", "sql", "sh", "json"}
IMAGE_TYPES = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"}
OFFICE_TYPES = {"xlsx", "pptx"}


@dataclass
class ParsedDocument:
    """解析结果。"""

    path: str
    file_type: str
    text: str = ""
    author: str = ""
    is_image: bool = False
    images: list[dict] = field(default_factory=list)  # PDF 图生文关联
    headings: list[str] = field(default_factory=list)  # 结构标题（docx Heading 样式 / xlsx sheet 名 / pptx 幻灯片标题）
    code_symbols: list[str] = field(default_factory=list)  # 代码签名（def/class 行）
    error: str = ""
    raw_meta: dict = field(default_factory=dict)


def file_type_of(path: str | Path) -> str:
    """扩展名小写；无扩展名或未知 → other。"""
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if not ext:
        return "other"
    if ext in TEXT_TYPES | CODE_TYPES | IMAGE_TYPES | OFFICE_TYPES:
        return ext
    return "other"


def parse_file(path: str | Path, *, min_chars: int = 0) -> ParsedDocument:
    """按类型解析文件 → 文本 + 作者。

    合入 runtime 后（2026-08-28）：PDF 只用 pymupdf（mineru 不搬），
    图片不做 OCR（仅元数据）。原 use_mineru 参数与图片 OCR 分支移除。

    `min_chars`（2026-09-29，仅 PDF 用）："抽到这么多字符就不必再往后看了"的上界，
    由调用方（indexer）按自己留几块算出（见 `chunker.need_chars_for_chunks`）。
    默认 0 = 不早停，**行为与改动前逐字一致** —— 这条保证了本参数之前的调用点
    （测试、其它入口）不受影响。

    ★ 注意它不是"块数"：parser 依然不知道块、不认识 chunk_size/overlap。
    传进来的是字符量，换算的权威在 chunker。
    """
    p = Path(path)
    ftype = file_type_of(p)
    doc = ParsedDocument(path=str(p), file_type=ftype)
    errors: list[str] = []      # 子步骤失败原因（目前只有 OCR 用），汇总进 doc.error

    try:
        if ftype in ("txt", "md"):
            doc.text = _read_text(p)
            doc.author = _author_from_doc(p, None)
        elif ftype == "docx":
            doc.text, doc.author, doc.headings, doc.images, doc.raw_meta["docx_scan"] = (
                _parse_docx(p, errors))
        elif ftype == "pdf":
            doc.text, doc.author, doc.images, doc.raw_meta["pdf_scan"] = _parse_pdf(
                p, errors, min_chars=min_chars)
        elif ftype == "xlsx":
            doc.text, doc.headings = _parse_xlsx(p)
        elif ftype == "pptx":
            doc.text, doc.headings = _parse_pptx(p)
        elif ftype in CODE_TYPES:
            doc.text = _read_text(p)
            doc.code_symbols = _extract_code_symbols(doc.text)
        elif ftype in IMAGE_TYPES:
            # 图片走 PP-OCR；有文本则供索引，失败/无文本 → 仅元数据（is_image 标记）。
            # ★ rev 5：失败写进 doc.error（errors 列表），让 indexer 能把它落成 failed，
            #   而不是与「这张图本来没字」一起静默成 unsupported。
            doc.is_image = True
            try:
                doc.text, conf = _ocr_image(p)
                doc.raw_meta["ocr_conf"] = conf
                if conf and conf < OCR_LOW_CONF:
                    doc.raw_meta["ocr_low_conf"] = True
            except Exception as e:                   # noqa: BLE001
                # 2026-09-28（test3）：结构化原因（见 describe_ocr_failure），
                # 不再拼 `"ocr: %s"` —— 那句英文会原样露在中文界面上。
                doc.error = describe_ocr_failure(e)
                errors.append(doc.error)
                logger.warning("OCR 失败 %s: %s", p, e)
        else:
            # other：仅元数据
            pass
    except Exception as e:
        logger.warning("解析失败 %s: %s", p, e)
        # 2026-09-28（test3）：同上 —— 结构化标记代替裸异常文本。
        errors.append(describe_parse_failure(e))

    # 汇总：单个子步骤报的错（如 OCR）优先保留，否则用解析期异常的摘要。
    if errors:
        doc.error = "; ".join(errors)[:300]
    return doc


# ── 失败原因的结构化标记（2026-09-28）──────────────────────────
#
# 用户报（真机）：界面语言选了中文，「失败文件」明细里那一行
# 右侧的原因仍是英文 —— `Failed to open file '/…/mm02-fake.pdf' as type pdf.`。
#
# 与 `pathutil.describe_unencodable_filename` 同一套做法、同一个理由：
#   **后端只给事实，文案在前端按语言拼**（当年那条中文写死后端、切英文时又
#   冒出中文，是同一个坑的另一面）。`kind` 早就这么做了，`reason` 一直漏。
#
# 为什么还要把英文原文带上（detail）：它是排障用的**技术细节**（库/pymupdf 的
# 原始报错），翻译会把信息丢掉。前端渲染成「解析失败：<原文>」—— 中文壳 + 可查证的
# 细节，既不是纯英文、也不丢线索。
PARSE_FAILED_PREFIX = "parseFailed:"
OCR_FAILED_PREFIX = "ocrFailed:"


def describe_parse_failure(exc: BaseException) -> str:
    """解析期异常 → 结构化原因标记（界面按语言出词，原文进 detail）。

    形如 `parseFailed:{"kind":"parseFailed","errorType":"FileDataError","detail":"…"}`。
    与 `filenameUnencodable:` 同一格式约定：标记 + JSON，前端认前缀、认 kind。
    """
    import json as _json

    detail = str(exc).strip() or exc.__class__.__name__
    return PARSE_FAILED_PREFIX + _json.dumps(
        {"kind": "parseFailed",
         "errorType": exc.__class__.__name__,
         "detail": detail[:200]},
        ensure_ascii=False,
    )


def describe_ocr_failure(exc: BaseException) -> str:
    """OCR 异常 → 结构化原因标记（同上；覆盖 `ocr: …` 那批）。

    这一类此前是 `"ocr: %s"` —— 中文界面下同样会露出英文原文，且
    `indexer` 与文档都在按 `ocr:` 前缀判据，一并换成结构化标记。
    """
    import json as _json

    detail = str(exc).strip() or exc.__class__.__name__
    return OCR_FAILED_PREFIX + _json.dumps(
        {"kind": "ocrFailed",
         "errorType": exc.__class__.__name__,
         "detail": detail[:200]},
        ensure_ascii=False,
    )


def _ocr_image(p: Path | str) -> tuple[str, float]:
    """PP-OCR 识别图片 → (文本, 平均置信度)。失败**抛异常**（由调用方决定怎么记）。

    2026-09-07：图片/图片型 PDF 支持（用户需求：pdf 能力完善 + png/jpg 可检索）。
    入参既可以是路径，也可以是 PIL Image（PDF 内嵌图），两条入口共用本函数。

    ## rev 5：只走 PP-OCR，不再回退 tesseract

    删掉的东西与理由（不是「简化」，是那套机器治的是别人的病）：
      · **两级救援**（OSD 判角 + 0°/180° 双向打分 + 置信度门槛）—— 治的是
        「tesseract psm=3 不做方向判断」，PP-OCR 自带方向分类器，等价物在引擎里。
      · **引擎回退**（PP-OCR 不可用 → tesseract）—— 它只把「读不出字」换成
        「读得出字但可能不准」，而代价是静默：OCR 故障在用户看来和「这张图本来没字」
        一模一样。改为「失败就抛出去、由调用方记成失败」，故障可见比多读几个字重要。
      · **健康闸**（字数 + 置信度）—— 它的作用原本是决定要不要进救援分支，救援没了它
        也就没有出口。conf 仍然要算出来，但用途改为**记录可疑**（见 OCR_LOW_CONF），
        不再触发自动重读。

    ## 保留：EXIF 转正（**只在真有方向标记时才动图像对象**）

    实测依据（GATE-0.1，2026-09-28，合成样本 + 四张真实图）：
      · 四张真实图摘掉转正后文本逐张一致 → 转正不是识别质量的来源；
      · 合成 Orientation=3 样本显示引擎会重排行序（集合相同），说明它按**行方向**判，
        不是按 EXIF；
      · 结论：EXIF 是方向标记的权威解释，转正与引擎互补，保留。
    条件是 `getexif()[0x0112]` 存在且 != 1 —— 无标记时连 `exif_transpose`（内部 copy、
    触发完整解码）都不做，避免把所有图片的解码路径都换掉。
    """
    from PIL import Image, ImageOps

    from bibit_filesearch import ocr_engine

    img = Image.open(str(p)) if isinstance(p, (str, Path)) else p   # PDF 内嵌图已是 PIL Image

    try:
        orientation = (img.getexif() or {}).get(0x0112)
    except Exception:                                # noqa: BLE001
        orientation = None
    if orientation and orientation != 1:
        img = ImageOps.exif_transpose(img)

    txt, conf = ocr_engine.recognize_ppocr(img)      # 失败抛异常，不在这里吞
    return _ocr_clean(txt), conf


def _ocr_clean(txt: str) -> str:
    return (txt or "").strip()


def _read_text(p: Path) -> str:
    """读文本文件（自动检测编码）。"""
    raw = p.read_bytes()
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _docx_embedded_images(doc) -> list[tuple[str, bytes]]:
    """按**文档顺序**给出 docx 的内嵌图 → `[(rId, blob), …]`。

    为什么按 blip 的出现顺序而不是 `part.rels`：rels 是一张**无序**的映射，而 OCR 文本
    必须内联回它所在段落的位置（见 `_parse_docx`）。`w:drawing` 里的 `a:blip/@r:embed`
    是文档顺序的权威来源，一个 rId 出现几次就代表同一张图被引用几次（例如页眉 logo
    可能被多段共用），每次都返回同一份 blob。

    取 blob 的路径：`document.part.related_parts[rId].blob`（python-docx 已把 media
    解好），不自己解 zip —— 与「用库的公开 API」这条一致。
    """
    from docx.oxml.ns import qn

    rels = doc.part.related_parts
    out: list[tuple[str, bytes]] = []
    for blip in doc.element.body.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed"))
        if not rid:
            continue
        part = rels.get(rid)
        # 只认图片部：blip 也可能指向 emf/wmf 等（OCR 读不了，跳过）
        ctype = getattr(part, "content_type", "") if part is not None else ""
        if part is None or not ctype.startswith("image/"):
            continue
        out.append((rid, part.blob))
    return out


def _paragraph_images(para, rels) -> list[bytes]:
    """某段落**直属**（含其内联 drawing）的内嵌图 blob 列表。

    只取该段落自己引用的图 —— 段落顺序遍历时每张图正好归到它所在的段落。
    `w:drawing` 嵌在 run 里，故直接在该段落的 XML 内找 `a:blip`。
    """
    from docx.oxml.ns import qn

    out: list[bytes] = []
    for blip in para._p.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed"))
        if not rid:
            continue
        part = rels.get(rid)
        ctype = getattr(part, "content_type", "") if part is not None else ""
        if part is not None and ctype.startswith("image/"):
            out.append(part.blob)
    return out


def _parse_docx(p: Path, errors: list[str]) -> tuple[str, str, list[str], list[dict], dict]:
    """解析 docx → (text, author, headings, images, scan)。

    ## 2026-10-09（rev 9）：内嵌图 OCR，与 PDF 的「逐页薄页判据」同构

    用户诉求原话：「对于 docx 或者 doc 中如果有图片而没有文字的情况：假设图片中能够
    OCR 出文字呢？我觉得应该和 pdf 用相同的处理方式」。

    PDF 的做法（`_parse_pdf`，rev 7/8）：逐页看文本厚薄 → **薄页**提该页内嵌图 OCR →
    OCR 文本**内联**在该页文本之后。docx 没有页，但每张内嵌图落在自己的**段落**里，
    段落就是 docx 的页，于是判据一一对应：

        PDF                      docx
        逐页                     逐段落（按文档顺序）
        页文本 < 40 字算「薄页」   段文本为空算「薄段」
        薄页才 OCR 该页的图         薄段才 OCR 该段的图
        OCR 文本内联在该页之后       OCR 文本内联在该段之后
        每文件最多 20 页           每文件最多 20 张图（DOCX_MAX_OCR_IMAGES）

    ★ 判据用「段落空」而不是「全篇空」。用全篇判会漏掉「正文 + 段内插图」的 docx
      （正文有字 → 全篇不空 → 插图一次都不 OCR），与 PDF 改前「整份判据」漏掉
      混合型 PDF 的扫描页是同一个错误。既然要求「和 PDF 相同」，就按段落判。

    ★ 为什么必须内联（不能统一追加到文末）：与 rev 7 同因 —— 追加到尾部时，正文
      先把 64 块额度占满，OCR 出来的字落到第 64 块之后，等于白跑。

    ★ 成本闸按「**尝试** OCR 的张数」算，不是「出字的张数」（与 PDF 同）：一份全空白
      插图的 docx 每张都 OCR 却读不出字，按出字数计闸就永远拦不住，正是它要防的病态输入。

    `images` 的形态与 PDF 一致：`[{page_idx, ocr_text}]`，page_idx 这里记**段落序号**，
    供溯源（`files.pdf_images` 列复用同一结构，见 indexer 的说明）。
    """
    import docx
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(p))
    parts: list[str] = []
    headings: list[str] = []
    images: list[dict] = []
    ocr_attempts = 0
    ocr_images = 0            # 实际读出文字并可用上的张数（上报用）
    capped = False

    rels = d.part.related_parts

    def _take(para) -> str:
        """段落 → 文本；Heading 样式同时记为标题。"""
        t = para.text.strip()
        if not t:
            return ""
        if para.style and para.style.name and para.style.name.lower().startswith("heading"):
            headings.append(t)
        return t

    def _para_ocr(para, seq: int) -> str:
        """段落无文字时，对该段的图 OCR → 内联文本（空串 = 无可读文本 / 已到闸）。"""
        nonlocal ocr_attempts, ocr_images, capped
        blobs = _paragraph_images(para, rels)
        if not blobs:
            return ""
        limit = int(DOCX_MAX_OCR_IMAGES or 0)
        got: list[str] = []
        for blob in blobs:
            if limit and ocr_attempts >= limit:
                capped = True
                break
            ocr_attempts += 1
            try:
                from PIL import Image
                import io as _io
                img = Image.open(_io.BytesIO(blob))
                txt, _conf = _ocr_image(img)
            except Exception as e:                    # noqa: BLE001
                errors.append(describe_ocr_failure(e))
                logger.warning("docx 内嵌图 OCR 失败 %s (para=%s): %s", p, seq, e)
                continue
            if txt:
                got.append(txt)
                images.append({"page_idx": seq, "ocr_text": txt})
        if got:
            ocr_images += 1                           # 计「段」，一段多图只算一段
            return "\n".join(got)
        return ""

    handled = set()
    seq = 0
    # 正文直属段落：逐段走，段内无文字且有图 → OCR 内联。
    for para in d.paragraphs:
        handled.add(para._p)
        t = _take(para)
        if not t:
            t = _para_ocr(para, seq)
        if t:
            parts.append(t)
        seq += 1
    # 表格文本
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
        # 单元格段落已合并为上面的行，登记以免下面重复收录
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    handled.add(para._p)
    # 文本框 / 图形等嵌套容器里的段落。d.paragraphs 只给 body 直属段落，正文放在
    # w:txbxContent 里的文档（WPS / PPT 导出的版式 docx 常见）会被整体读成空文本，
    # 进而被判「无文本」→ 状态 unsupported → 检索永远搜不到
    # （2026-09-15 用户实测：AI.docx 1464 字全在 32 个文本框内）
    for el in d.element.body.iter(qn("w:p")):
        if el in handled:
            continue
        para = Paragraph(el, d._body)
        t = _take(para)
        if not t:
            t = _para_ocr(para, seq)
        if t:
            parts.append(t)
        seq += 1
    author = ""
    try:
        cp = d.core_properties
        author = (cp.author or "").strip()
    except Exception:
        pass
    total_images = len(_docx_embedded_images(d))
    scan = {
        "strategy": "paragraph",
        "total_paragraphs": seq,
        "total_images": total_images,
        "ocr_images": ocr_images,
        "ocr_attempts": ocr_attempts,
        "ocr_images_capped": capped,
    }
    return "\n".join(parts), author, headings, images, scan


def _first_thin_page_with_image(doc, page_texts: list[str], window: int) -> int | None:
    """预算窗口（前 window 页）内第一张「薄页 + 带内嵌图」的页号；没有则 None。

    专供 `_parse_pdf` 的文本快速路径（2026-10-08，审查 R04）。判据与逐页路径
    保持一致：页文本 < `PDF_THIN_PAGE_CHARS` 视为薄页；只有真带内嵌图才有
    OCR 的价值（无图薄页 OCR 也只能读出同样的空白）。

    返回 None 表示「窗口内没有需要 OCR 的页」，调用方可以放心走零 OCR 的快速
    返回。这个探针自己很便宜：只 `load_page().get_images()`，不做任何推理；
    而且是**短路**的——遇到第一张就返回，纯文本 PDF 的开销是每页一次
    `get_images()`（实测 529 页的文本型书仍在秒级）。
    """
    for pageno in range(min(window, len(page_texts))):
        if len(page_texts[pageno].strip()) >= PDF_THIN_PAGE_CHARS:
            continue
        try:
            if doc.load_page(pageno).get_images(full=True):
                return pageno
        except Exception:                            # noqa: BLE001 —— 探测失败不阻断解析
            continue
    return None


def _pixmap_to_image(pix):
    """pymupdf Pixmap → PIL Image（供 OCR 直接用内存，不再落临时 PNG）。

    2026-09-24：旧版把 pix 存成临时 PNG 再让 `_ocr_image` 重新读盘。那一步既慢又要
    一个 fd；OCR 预处理（OSD 判角 / 转正）本来就要求手上有一张 PIL 图，顺手一并去掉。
    """
    from PIL import Image

    if pix.colorspace is None or pix.n == 1:
        return Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("RGB")
    if pix.n == 3:
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # CMYK / 带 alpha 等：pymupdf 已在上游转成 sRGB
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _parse_pdf(p: Path, errors: list[str],
               min_chars: int = 0) -> tuple[str, str, list[dict], dict]:
    """PDF 解析（2026-09-29 起为**两遍 + 逐页判据 + 预算早停**）。

    ## 返回
    `(text, author, images, scan)`
      · text   可索引文本。**OCR 文本内联在该页位置**（见下"内联"一段）
      · images 图生文关联 `[{page_idx, ocr_text}]`，供溯源
      · scan   本次扫描的结构化事实（`meta["pdf_scan"]`），见文件末尾 "scan 字段"

    ## 两遍结构（改前是"一遍 + 整份判据"）
      pass 0  逐页 `page.get_text("text")`，只为**累加字数**（实测 4.6ms/页，恒廉价）。
              这一步决定两件事：这份 PDF 是不是"文本型"，以及文本是否已够 min_chars。
      pass 1  按需翻页：
                · 文本型且 pass 0 已够预算 → **一次 OCR 都不做**（实测 529 页的书 2.5s）
                · 否则逐页走：文本厚 → 直接用；文本薄（< PDF_THIN_PAGE_CHARS）→
                  提该页内嵌图 OCR，并把结果**内联到该页文本之后**
              累进到 len(text) >= min_chars 即停（"填满就不再往后看"）。

    ## 为什么必须是"两遍"而不是"边读边判"（实测对照，2026-09-29）
    "边读边判"会把文本型书籍里的**封面/章节扉页/本页留白**（文本 <40 字、带图）
    误判成扫描页而去 OCR：实测《The Art of Multiprocessor Programming》多跑一次
    3.9s 的无用 OCR（该页 OCR 结果对索引毫无贡献）。两遍模式由 pass 0 的**全文字数**
    一眼判定"这是文本型书"，同书 OCR 次数 1 → 0，总耗时 6.3s → 2.5s。
    pass 0 的代价有界且便宜（实测 4.6ms/页；纯扫描书更便宜，511 页仅 232ms，
    因为抽不出字就没什么可做），且它顺带产出"全文字数"这个**文档级事实**，
    正是 `scan` 里"是否被预算截断"要用的。

    ## 判据从"整份"改成"逐页"（rev 7 的另一半）
    改前：`if not text.strip():` —— 判的是**整份文档**有没有文本层。
      ⇒ 实测 `Python源码剖析.pdf`（29MB / 511 页）：
        第 511 页（版权页）有 99 字文本层，其余 510 页是扫描图。整份判据不为空 →
        **510 页扫描内容一张图都没 OCR**，索引里只剩那 99 个字的版权页。
    改后：逐页判据 —— 页面文本 < PDF_THIN_PAGE_CHARS 就算"薄"，对该页提图 OCR。
      ⇒ 混合型 PDF（正文文字 + 插图/扫描页）不再漏掉扫描页那一半。

    ## OCR 文本**内联**在页位置（不是统一追加到全文尾部）
    改前是 `parts.append(f"[第{pageno+1}页图片] {ocr_text}")` —— 所有 OCR 文本
    都排在**全文末尾**。这在"整份无文本层"时无害（反正本来就没别的文本），
    但逐页判据下会出事：混合型 PDF 的正文文本会先把前面几十块占满，
    排在尾部的 OCR 文本落到第 64 块之后 → **花了 OCR 时间却一块都进不了索引**。
    内联（每页文本后面紧跟该页的 OCR 文本）保证 OCR 结果与所在页一起参与切块。

    ## 不做整页渲染（沿用原结论）
    实测把整页 rasterize 再 OCR，在"图只占页面一小块"的 PDF 上（bill.pdf：
    A4 页面里一块 387x516 的发票截图，暗像素仅 1.5%）各 dpi 都返回空文本，
    比抽内嵌图更差；且整页渲染会丢掉"一图一条 pdf_images"的粒度。
    """
    author = ""
    import pymupdf

    ocr_pages_limit = int(PDF_MAX_OCR_PAGES or 0)
    want = int(min_chars or 0)

    d = pymupdf.open(str(p))
    try:
        npages = len(d)
        # ── pass 0：逐页抽文本，只累加字数（廉价、无副作用）──
        page_texts = [pg.get_text("text") for pg in d]
        total_chars = sum(len(t) for t in page_texts)
        author = (d.metadata or {}).get("author", "") or ""

        # 文本型 + 已够预算 ⇒ 直接用，一次 OCR 都不做。
        #
        # ★ 2026-10-08（审查 R04）：但**预算窗口内不能有「薄页 + 内嵌图」**。
        #   混合型 PDF 的前置扫描页文本为 0、只有一张图，全篇字数由后面的文本页
        #   撑起来时，这条快速路径会直接返回文本层 —— 那些扫描页的内容一次都没
        #   OCR，等于「索引里没有那些字」（实测：前置图页 + 后置文本页，budget 小时
        #   OCR 0 次、图页文本丢失；budget 调大走逐页路径才保留）。
        #   窗口**之后**的薄页不用管：它们本来就会因预算被切在 64 块之外。
        #   判据写在这里而不是整份 PDF 上，是为了保住「纯文本书零 OCR」这个实测收益
        #   （529 页的书 2.5s）——只有窗口内真有带图薄页时才退回逐页路径。
        if want > 0 and total_chars >= want:
            # 只取"够预算"的那些页：pass 0 已经读完全部页（成本已付），但交给下游的
            # 文本必须裁到预算之内 —— 否则 indexer 会把整本切成 2690 块再丢掉 97%，
            # 白付切块/embed 前的那段成本，且 `pages_consumed` 会虚报成"整本都用上了"。
            kept, acc_fast = [], 0
            for t in page_texts:
                kept.append(t)
                acc_fast += len(t) + 1
                if acc_fast >= want:
                    break
            thin_with_image = _first_thin_page_with_image(d, page_texts, len(kept))
            if thin_with_image is None:
                text = "\n".join(kept)
                scan = {
                    "strategy": "twopass",
                    "is_text_layer": True,
                    "total_pages": npages,
                    "total_chars": total_chars,
                    "pages_consumed": len(kept),
                    "ocr_pages": 0,
                    "ocr_pages_capped": False,
                    "stopped_by_budget": True,
                    "min_chars": want,
                }
                return text, author.strip(), [], scan
            logger.info(
                "PDF 预算窗口内有带图薄页（第 %d 页，%d 字）：转入逐页路径对该页 OCR（%s）",
                thin_with_image + 1, len(page_texts[thin_with_image].strip()), p,
            )

        # ── pass 1：逐页取文本；薄页提图 OCR；累进到够预算即停 ──
        parts: list[str] = []
        images: list[dict] = []
        acc = 0
        used = 0
        ocr_pages = 0        # 实际 OCR 出字并可用上的页数（上报用）
        ocr_attempts = 0     # **尝试**过 OCR 的页数 —— 闸按这个算
        capped = False
        for pageno in range(npages):
            t = page_texts[pageno].strip()
            used += 1
            # 薄页 → 提该页内嵌图 OCR（逐页判据；整份判据的替代）
            if len(t) < PDF_THIN_PAGE_CHARS:
                page = d.load_page(pageno)
                page_ocr: list[str] = []
                imgs = page.get_images(full=True)
                # ★ 闸按"尝试页数"算，不是按"出字页数"：一本空白扫描书每页都 OCR
                #   却读不出字（ocr_pages 恒 0），若按出字页数计闸就永远拦不住，
                #   正是它要防的病态输入。这一条是测试
                #   test_blank_scan_book_is_capped 抓出来的真 bug。
                if imgs and ocr_pages_limit and ocr_attempts >= ocr_pages_limit:
                    capped = True
                    if t:
                        parts.append(t)
                        acc += len(t) + 1
                    break
                if imgs:
                    ocr_attempts += 1
                for img in imgs:
                    xref = img[0]
                    try:
                        pix = pymupdf.Pixmap(d, xref)
                        if pix.colorspace and pix.colorspace.n > 3:
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        ocr_text, _conf = _ocr_image(_pixmap_to_image(pix))
                        del pix
                        if ocr_text:
                            images.append({"page_idx": pageno, "ocr_text": ocr_text})
                            page_ocr.append(ocr_text)
                    except Exception as e:               # noqa: BLE001
                        errors.append(describe_ocr_failure(e))
                        logger.warning("PDF 提图 OCR 失败 %s (xref=%s): %s", p, xref, e)
                if page_ocr:
                    ocr_pages += 1                       # 计"页"，一页多图只算一页
                    ocr_join = "\n".join(page_ocr)
                    # 内联：该页文本（若有）之后紧跟该页 OCR 文本
                    t = (t + "\n" + ocr_join).strip() if t else ocr_join
                if ocr_pages_limit and ocr_attempts >= ocr_pages_limit:
                    # OCR 页闸已满：本页之后不再尝试 OCR（已有的文本照常计入）。
                    # 这不是"失败"——超出的页本来也进不了索引（indexer 只留前 64 块），
                    # 闸只是把"白跑 NPU"挡掉。scan["ocr_pages_capped"] 会如实上报。
                    capped = True
                    if t:
                        parts.append(t)
                        acc += len(t) + 1
                    break
            if t:
                parts.append(t)
                acc += len(t) + 1                            # +1 ≈ 拼接的换行
            # 预算早停：攒够就不再往后看（这页 OCR 出来的字也算进去）
            if want > 0 and acc >= want:
                break

        text = "\n".join(parts)
        scan = {
            "strategy": "twopass",
            "is_text_layer": total_chars > 0,
            "total_pages": npages,
            "total_chars": total_chars,
            "pages_consumed": used,
            "ocr_pages": ocr_pages,
            "ocr_attempts": ocr_attempts,
            "ocr_pages_capped": capped,
            "stopped_by_budget": bool(want > 0 and acc >= want),
            "min_chars": want,
        }
        return text, author.strip(), images, scan
    finally:
        d.close()


def _parse_xlsx(p: Path) -> tuple[str, list[str]]:
    import openpyxl

    wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
    parts = []
    sheet_names: list[str] = []
    for ws in wb.worksheets:
        sheet_names.append(ws.title)
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                parts.append(" | ".join(cells))
    wb.close()
    return "\n".join(parts), sheet_names


def _parse_pptx(p: Path) -> tuple[str, list[str]]:
    from pptx import Presentation

    prs = Presentation(str(p))
    parts = []
    slide_titles: list[str] = []
    for slide in prs.slides:
        # 幻灯片标题占位符（title 布局）
        title = ""
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text.strip()
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(run.text for run in para.runs).strip()
                    if t:
                        parts.append(t)
        if title:
            slide_titles.append(title)
    return "\n".join(parts), slide_titles


def _extract_code_symbols(text: str) -> list[str]:
    """代码签名提取：def/class 行（摘要要点用）。"""
    syms = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith(("def ", "class ")) or s.startswith(("async def ", "public ", "private ", "func ", "fn ")):
            syms.append(s[:100])
        if len(syms) >= 10:
            break
    return syms


def _author_from_doc(p: Path, doc: object | None) -> str:
    """docx/pdf 的 author 已在各解析函数处理；txt/md 无作者。"""
    return ""
