"""PDF 两遍解析 + 逐页判据 + 预算早停（PARSER_REVISION 7，2026-09-29）。

## 改前的行为（rev 6）

    _parse_pdf: 逐页 get_text → 全文拼起来 → `if not text.strip():` 才走 OCR
                整份判据 + 无预算概念 ⇒ ① 混合型 PDF 全漏；② 超预算也全量解析

## 改后（rev 7）

    pass 0  逐页抽文本只为累加字数（廉价）→ 判定"文本型"与"够不够预算"
    pass 1  文本型且够预算 ⇒ **0 次 OCR**；否则逐页：薄页(<40 字)提内嵌图 OCR，
            结果**内联**在该页位置；累进到 >= min_chars 即停
    闸      PDF_MAX_OCR_PAGES（出厂 20）：防"大量无字图页"把 NPU 占住

## 为什么这些用例不打真 PDF / 不跑真 OCR

与 `test_image_pdf_ocr_dupe.py` 同一考量：真 pymupdf + 真 PP-OCR 会把全量套件的
fd 峰值推过 1024，排在后面的 `test_t3_parser` 真 OCR 用例拿不到 fd 而失败、
整个会话作废。此处用假 pymupdf 模块 + 打桩 `_ocr_image`，判别力不变（被测的正是
"哪一页会被送进 OCR、什么时候停"这套决策），只是不花 fd。
真 pymupdf 端到端仍由 `test_t3_parser.test_parse_pdf_image_only_ocr` 守。
"""
import sys
import types

import pytest

from bibit_filesearch import parser as P
from bibit_filesearch.chunker import chunk_text, need_chars_for_chunks


# ────────────────────────── 假 pymupdf ──────────────────────────

class _FakePage:
    def __init__(self, text, images=()):
        self.text = text
        self.images = list(images)          # xref 列表

    def get_text(self, kind="text"):
        # ★ 与真 pymupdf 一致：**保留**原始空白（真引擎返回的是排版文本，含空格/换行）。
        #   v1 夹具这里返回的就是"无空白短串"，恰好让生产代码的 `t.strip()` 把它清成 ""
        #   → 薄页判定误命中（明明是正文页）。夹具失真会造出假 bug，故显式还原。
        return self.text

    def get_images(self, full=False):
        return [(x, 0, 0, 0, 0, 0, 0) for x in self.images]


class _FakePixmap:
    def __init__(self, doc, xref):
        self.xref = xref
        self.colorspace = types.SimpleNamespace(n=3)
        self.width = self.height = 10


class _FakeDoc:
    def __init__(self, pages, author=""):
        self.pages = pages
        self.metadata = {"author": author}
        self.closed = False
        self.loaded = []

    def __len__(self):
        return len(self.pages)

    def __iter__(self):
        return iter(self.pages)

    def load_page(self, i):
        self.loaded.append(i)
        return self.pages[i]

    def close(self):
        self.closed = True


def _install_fake_pymupdf(monkeypatch, pages, author="", ocr_text="OCR字"):
    """把 parser 里 `import pymupdf` 拿到的东西换掉；返回 (doc, ocr_calls)。"""
    doc = _FakeDoc(pages, author=author)
    mod = types.ModuleType("pymupdf")
    mod.open = lambda *a, **k: doc
    mod.Pixmap = _FakePixmap
    mod.csRGB = object()
    monkeypatch.setitem(sys.modules, "pymupdf", mod)

    calls = []

    def fake_ocr(img):
        calls.append(getattr(img, "xref", None))
        # xref=0 约定为"读不出字"的图页（模拟纯装饰图/空白扫描页）
        if getattr(img, "xref", None) == 0:
            return "", 0.0
        return "%s%d" % (ocr_text, getattr(img, "xref", 0)), 95.0

    monkeypatch.setattr(P, "_ocr_image", fake_ocr)
    monkeypatch.setattr(P, "_pixmap_to_image", lambda pix: pix)
    return doc, calls


# ─────────────────── 换算：need_chars_for_chunks ───────────────────

def test_need_chars_matches_chunk_text_exactly():
    """换算公式必须与 chunk_text 的实际切块**逐字一致**（否则早停会停早/停晚）。

    判据：对一批长度，len(chunk_text(text[:need(n)])) == n。
    这是"早停阈值"这层唯一的正确性来源 —— 公式错了不会报错，只会静默少收内容。
    """
    for n in (1, 2, 5, 64, 65, 200):
        need = need_chars_for_chunks(500, 100, n)
        text = ("汉字八" * 40000)          # 无空白，走硬切（不受"词内回退"影响）
        if len(text) < need:
            text = text * 3
        assert len(chunk_text(text[:need], 500, 100)) == n, \
            "need_chars_for_chunks(%d)=%d 时 chunk_text 未产出 %d 块" % (n, need, n)
        assert len(chunk_text(text[:need - 1], 500, 100)) == n - 1, \
            "少一个字符就该只产出 %d 块（阈值必须紧）" % (n - 1)


def test_need_chars_zero_means_unlimited():
    assert need_chars_for_chunks(500, 100, 0) == 0
    assert need_chars_for_chunks(500, 100, -3) == 0


# ─────────────────── min_chars=0：行为与改前一致 ───────────────────

def test_min_chars_default_keeps_old_behaviour(monkeypatch):
    """min_chars=0（默认）⇒ 不早停：翻完所有页，且薄页照样 OCR。

    这条守的是**兼容性**：parse_file 的老调用点（测试、其它入口）不传参时，
    必须与 rev 6 一样"全量解析"。早停是 opt-in，不是行为替换。
    """
    pages = [_FakePage("", images=[1]) for _ in range(5)]
    _install_fake_pymupdf(monkeypatch, pages, ocr_text="第")
    text, _author, images, scan = P._parse_pdf("/x.pdf", [], min_chars=0)
    assert scan["pages_consumed"] == 5, "min_chars=0 应翻完全部页"
    assert scan["ocr_pages"] == 5, "min_chars=0 时薄页全部 OCR"
    assert scan["stopped_by_budget"] is False
    assert len(images) == 5


# ─────────────────── 文本型 + 够预算 ⇒ 0 次 OCR ───────────────────

def test_text_layer_pdf_does_zero_ocr(monkeypatch):
    """文本型书（每页正文远超 40 字）+ 预算窗口内**没有带图薄页** ⇒ 一次 OCR 都不做。

    实测对照（真文件，2026-09-29）：《The Art of Multiprocessor Programming》
    529 页文本型，单遍模式因封面/扉页"薄"而多跑 1 次无用 OCR（3.9s）；
    两遍模式 0 次，总耗时 6.3s → 2.5s。这条就是这个收益的守门测试。

    ⚠️ 2026-10-08（审查 R04）订正：窗口内若出现**带图的薄页**，这条快速路径
    不再成立——那种页正是混合型 PDF 的前置扫描页，放过它等于「索引里没有那些字」。
    见 test_leading_scanned_page_is_not_skipped_by_budget 与
    test_thin_page_with_image_falls_back_to_per_page_ocr。本用例改为纯文本页，
    守住「窗口内没有带图薄页时仍然零 OCR」这一半。
    """
    body = "这是一段真实的正文段落。 " * 300        # 单页 ~3600 字，远超阈值
    pages = [_FakePage(body) for _ in range(20)]
    pages[0] = _FakePage("封面")                 # 封面"薄"但**不带图** → 无需 OCR
    _install_fake_pymupdf(monkeypatch, pages)
    want = need_chars_for_chunks(500, 100, 3)   # 3 块就够 → 第 1 页抽完就够
    text, _a, images, scan = P._parse_pdf("/x.pdf", [], min_chars=want)
    assert scan["ocr_pages"] == 0, "文本型且窗口内无带图薄页时不该有任何 OCR"
    assert scan["is_text_layer"] is True
    assert scan["stopped_by_budget"] is True
    assert images == []


def test_leading_scanned_page_is_not_skipped_by_budget(monkeypatch):
    """前置扫描页 + 后续足够文本 ⇒ 前置页仍要被 OCR（审查 R04 的核心用例）。

    改前：pass 0 看到全篇字数够预算，直接返回文本层，一次 OCR 都不做 ——
    第 1 页（正文内容全在图里）的内容**永远进不了索引**。实测（合成 PDF +
    mock OCR）：min_chars=100 时 OCR 调用 0 次、该页文本缺失；调大到 1000
    才走逐页路径并保留。这条守住「预算不能成为漏页的理由」。
    """
    pages = [
        _FakePage("", images=[7]),                                  # 第 1 页：扫描页
        _FakePage("这是一段真实的正文段落。 " * 100),                  # 第 2 页：正文
    ]
    _install_fake_pymupdf(monkeypatch, pages, ocr_text="扫描")
    # want 小到第 2 页单页就够 —— 改前会因此直接走快速路径
    want = need_chars_for_chunks(500, 100, 2)
    text, _a, images, scan = P._parse_pdf("/x.pdf", [], min_chars=want)

    assert scan["ocr_pages"] == 1, (
        "预算窗口内的扫描页必须被 OCR；改前这条快速路径会返回 0 次"
    )
    assert "扫描7" in text, "OCR 文本必须落在返回文本里（否则索引里没有那些字）"
    assert images and images[0]["page_idx"] == 0
    assert scan["stopped_by_budget"] is True, "仍然要在攒够预算后停下"


def test_thin_page_with_image_after_window_is_not_ocrd(monkeypatch):
    """窗口**之后**的带图薄页不必 OCR（它本来就会被 64 块上限切掉）。"""
    body = "这是一段真实的正文段落。 " * 300
    pages = [_FakePage(body) for _ in range(3)]
    pages.append(_FakePage("", images=[9]))      # 第 4 页：预算窗口之外
    _install_fake_pymupdf(monkeypatch, pages, ocr_text="晚")
    want = need_chars_for_chunks(500, 100, 4)    # 4 块 ≈ 1300 字 → 1 页就够
    _t, _a, images, scan = P._parse_pdf("/x.pdf", [], min_chars=want)
    assert scan["ocr_pages"] == 0, "窗口之外的薄页不该触发 OCR"
    assert images == []


def test_pure_image_pdf_still_ocrs_everything(monkeypatch):
    """退化保障：全扫描件（总字数 0，够不到预算）仍走逐页 OCR，行为不变。"""
    pages = [_FakePage("", images=[i + 1]) for i in range(3)]
    _install_fake_pymupdf(monkeypatch, pages, ocr_text="页")
    _t, _a, images, scan = P._parse_pdf(
        "/x.pdf", [], min_chars=need_chars_for_chunks(500, 100, 64))
    assert scan["ocr_pages"] == 3
    assert len(images) == 3


def test_first_thin_page_with_image_probe(monkeypatch):
    """探针本身：只在窗口内找、命中即短路、纯文本页返回 None。"""
    doc = _FakeDoc([_FakePage("正文" * 100), _FakePage("", images=[1]), _FakePage("", images=[2])])
    assert P._first_thin_page_with_image(doc, [p.text for p in doc.pages], 3) == 1
    assert P._first_thin_page_with_image(doc, [p.text for p in doc.pages], 1) is None, \
        "窗口只到第 1 页时不该看到第 2 页"
    doc2 = _FakeDoc([_FakePage("正文" * 100), _FakePage("正文" * 100)])
    assert P._first_thin_page_with_image(doc2, [p.text for p in doc2.pages], 2) is None


# ─────────────────── 逐页判据：混合型 PDF 不再全漏 ───────────────────

def test_mixed_pdf_ocrs_thin_pages(monkeypatch):
    """混合型（版权页 99 字 + 510 页扫描页）⇒ 扫描页走到 OCR。

    这是 rev 7 修的实机坑（真文件 /home/user/pdf/Python源码剖析.pdf，29MB/511 页）：
    第 511 页版权页有 99 字文本层 → 旧判据 `if not text.strip()` 判"整份有文本" →
    **510 页扫描内容一张图都没 OCR**，索引里只剩那 99 个字的版权页。
    """
    body = "这是一段真实的正文段落。 " * 200
    pages = [_FakePage("", images=[i + 1]) for i in range(4)]   # 无文本层的扫描页（各带 1 图）
    pages.append(_FakePage("[General Information] 书名=Python源码剖析 99 字"))  # 版权页
    pages.extend(_FakePage(body) for _ in range(5))    # 后面的正文页
    _install_fake_pymupdf(monkeypatch, pages, ocr_text="扫描")
    # 不传 min_chars ⇒ 不早停，走完所有页
    text, _a, images, scan = P._parse_pdf("/x.pdf", [], min_chars=0)
    assert scan["ocr_pages"] == 4, "4 个无文本层的扫描页都该 OCR"
    assert len(images) == 4
    assert "扫描" in text
    assert all(img["page_idx"] < 4 for img in images)


def test_ocr_text_inlined_at_its_page(monkeypatch):
    """OCR 文本必须**内联在该页位置**，不能统一追加到全文尾部。

    为什么这条是硬要求（2026-09-29）：改前的写法是
    `parts.append(f"[第{n}页图片] {ocr_text}")` —— 所有 OCR 文本排在全文末尾。
    整份判据下无害（本来就没别的文本）；逐页判据下会出事：混合型 PDF 的正文
    会先占满前面几十块，排尾部的 OCR 文本落到第 64 块之后 ⇒ 花了 OCR 时间
    却一块都进不了索引。
    """
    pages = [
        _FakePage("", images=[1]),                        # 第 1 页：扫描页 → OCR 出 AAA
        _FakePage("这是一段真实的正文段落。 " * 100),        # 第 2 页：正文
        _FakePage("", images=[2]),                        # 第 3 页：扫描页 → OCR 出 BBB
    ]
    _install_fake_pymupdf(monkeypatch, pages, ocr_text="")
    # 打桩：第 1 页出 AAA、第 3 页出 BBB
    seen = []

    def fake_ocr(img):
        seen.append(img.xref)
        return ("AAA" if img.xref == 1 else "BBB"), 95.0

    monkeypatch.setattr(P, "_ocr_image", fake_ocr)
    text, _a, _imgs, _scan = P._parse_pdf("/x.pdf", [], min_chars=0)
    i_aaa, i_body, i_bbb = text.index("AAA"), text.index("这是一段真实"), text.index("BBB")
    assert i_aaa < i_body < i_bbb, \
        "OCR 文本应内联在原页位置（AAA 在正文前、BBB 在正文后），实际顺序索引 %s" % (
            (i_aaa, i_body, i_bbb),)


# ─────────────────── 预算早停 + OCR 页闸 ───────────────────

def test_stops_when_budget_reached(monkeypatch):
    """攒够 min_chars 就停，不再翻后面的页（"填满就不干了"）。"""
    pages = [_FakePage("这是一段真实的正文段落。 " * 300) for _ in range(30)]
    _install_fake_pymupdf(monkeypatch, pages)
    want = need_chars_for_chunks(500, 100, 4)      # 4 块 ≈ 1300 字 → 1 页就够
    _t, _a, _i, scan = P._parse_pdf("/x.pdf", [], min_chars=want)
    assert scan["pages_consumed"] == 1, "第 1 页抽完就够 4 块，不该翻第 2 页"
    assert scan["stopped_by_budget"] is True
    assert scan["total_pages"] == 30


def test_ocr_page_cap_blocks_pathological_file(monkeypatch):
    """病态输入：前面全是"OCR 也读不出字"的图页 → OCR 页闸（20）拦停。

    为什么必须有这道闸：逐页判据把"整本扫描书"从"不 OCR"翻成"逐页 OCR"，
    实测单页 5.2~7.5s，不设闸就是 43~64 分钟的 NPU 占用（真文件实测）。
    正常书碰不到这个闸（预算早停通常 10~40 页内就攒够），它只治病态。
    """
    pages = [_FakePage("", images=[5]) for _ in range(100)]   # 每页 1 张图，OCR 出字
    _doc, calls = _install_fake_pymupdf(monkeypatch, pages, ocr_text="字")
    monkeypatch.setattr(P, "PDF_MAX_OCR_PAGES", 5)
    _t, _a, _i, scan = P._parse_pdf("/x.pdf", [], min_chars=need_chars_for_chunks(500, 100, 64))
    assert scan["ocr_pages"] == 5, "该在 5 页处被闸拦下"
    assert scan["ocr_pages_capped"] is True
    assert scan["stopped_by_budget"] is False, "停的是闸，不是预算"
    assert len(calls) == 5, "第 6 页起不该再调用 OCR（闸的用途就是不白跑 NPU）"


def test_blank_scan_book_is_capped(monkeypatch):
    """空白扫描书（每页都 OCR 但读不出字）也必须被闸停住 —— 否则无限往后翻。

    这是本文件第二个夹具抓出来的**真 bug**：闸原先按"OCR 出字的页数"计，
    而病态输入恰恰是"每页都试、每页都读不出字"（ocr_pages 恒 0）→ 闸永远不触发
    → 100 页空白扫描书会把 100 页全翻完。改为按"**尝试**页数"计。
    """
    pages = [_FakePage("", images=[0]) for _ in range(100)]   # xref=0 ⇒ OCR 出空
    _doc, calls = _install_fake_pymupdf(monkeypatch, pages)
    monkeypatch.setattr(P, "PDF_MAX_OCR_PAGES", 6)
    _t, _a, _i, scan = P._parse_pdf("/x.pdf", [], min_chars=0)   # 不早停，只能靠闸
    assert len(calls) == 6, "试过 6 页就该停，实际试了 %d 页" % len(calls)
    assert scan["ocr_attempts"] == 6
    assert scan["ocr_pages_capped"] is True


def test_cap_zero_means_unlimited(monkeypatch):
    """闸设 0 = 不限（老行为）：不因页数被拦。"""
    pages = [_FakePage("", images=[0]) for _ in range(8)]
    _install_fake_pymupdf(monkeypatch, pages)
    monkeypatch.setattr(P, "PDF_MAX_OCR_PAGES", 0)
    _t, _a, _i, scan = P._parse_pdf("/x.pdf", [], min_chars=0)
    assert scan["ocr_pages_capped"] is False
    assert scan["pages_consumed"] == 8


# ─────────────────── 短件：不能因为"不满一块"就产出 0 块 ───────────────────

def test_short_pdf_still_yields_one_chunk(monkeypatch):
    """一页短扫描件（OCR 出 38/366 字）必须**仍产出 1 块**。

    实机依据：`K3-GPU调试笔记-含图.pdf` OCR 出 38 字、`invoice.pdf` 出 366 字，
    两者都 < chunk_size(500)。chunk_text 对 `len(text) <= size` 返回 `[text]`
    （不是 []），所以这类文件必须能索引到 —— 发票、截图笔记全是这个形态，
    若在解析/早停这一层把它们打没，等于整类文件搜不到。
    """
    pages = [_FakePage("", images=[1])]
    _install_fake_pymupdf(monkeypatch, pages, ocr_text="发票号码 24932000000038888512")
    text, _a, _i, scan = P._parse_pdf(
        "/x.pdf", [], min_chars=need_chars_for_chunks(500, 100, 64))
    assert len(text) < 500, "这个夹具本就短于一块"
    assert len(chunk_text(text, 500, 100)) == 1
    assert scan["stopped_by_budget"] is False       # 只有一页，翻完为止


# ─────────────────── parse_file 的接线 ───────────────────

def test_parse_file_passes_min_chars_and_exposes_scan(monkeypatch):
    """parse_file 要把 min_chars 传到 _parse_pdf，并把 scan 事实放进 raw_meta。"""
    got = {}

    def fake_parse_pdf(path, errors, min_chars=0):
        got["min_chars"] = min_chars
        return "文本", "作者", [], {"strategy": "twopass", "pages_consumed": 7}

    monkeypatch.setattr(P, "_parse_pdf", fake_parse_pdf)
    monkeypatch.setattr(P, "file_type_of", lambda p: "pdf")
    doc = P.parse_file("/x.pdf", min_chars=12345)
    assert got["min_chars"] == 12345
    assert doc.raw_meta["pdf_scan"]["pages_consumed"] == 7
    assert doc.text == "文本" and doc.author == "作者"


def test_parse_file_min_chars_defaults_to_zero(monkeypatch):
    """不传 min_chars ⇒ 0（不早停）。老调用点零改动、行为不变。"""
    got = {}
    monkeypatch.setattr(P, "_parse_pdf",
                        lambda path, errors, min_chars=0: (got.setdefault("mc", min_chars), "", [], {})[1:])
    monkeypatch.setattr(P, "file_type_of", lambda p: "pdf")
    P.parse_file("/x.pdf")
    assert got["mc"] == 0
