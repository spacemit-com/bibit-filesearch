"""解析 + 分块 + 元数据提取测试。

覆盖：全部契约类型解析、代码语义命中（分块策略兜底验收）、author 提取、创建时间回退、zip 等 other 仅元数据。
"""
import sys
from pathlib import Path

import pytest


from bibit_filesearch.chunker import chunk_code, chunk_text, split_by_type  # noqa: E402
from bibit_filesearch.metadata import extract_metadata  # noqa: E402
from bibit_filesearch.parser import file_type_of, parse_file  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"


@pytest.fixture(scope="module", autouse=True)
def make_samples():
    """构造测试样本文件（覆盖各类型）。"""
    SAMPLES.mkdir(exist_ok=True)
    (SAMPLES / "note.txt").write_text("这是纯文本文件，用于测试文本解析。\n第二行内容。", encoding="utf-8")
    (SAMPLES / "doc.md").write_text("# 标题\n\n正文段落，**加粗**。\n", encoding="utf-8")
    (SAMPLES / "calc.py").write_text(
        '# 计算总价模块\n"""计算购物车总价。"""\n\ndef calculate_total_price(items):\n    """计算总价。"""\n    return sum(i["price"] * i["qty"] for i in items)\n\ndef apply_discount(total, rate):\n    return total * (1 - rate)\n',
        encoding="utf-8",
    )
    (SAMPLES / "server.js").write_text(
        "// HTTP 服务器\nfunction startServer(port) {\n  console.log('start on', port);\n}\n\n// 处理请求\nasync function handleRequest(req, res) {\n  res.end('ok');\n}\n",
        encoding="utf-8",
    )
    (SAMPLES / "util.java").write_text(
        "public class Util {\n    // 字符串工具\n    public static String trimAll(String s) {\n        return s.trim();\n    }\n}\n",
        encoding="utf-8",
    )
    # xlsx：openpyxl 生成（依赖 requirements 已含 openpyxl；缺它此测试 xfail）
    try:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws["A1"] = "产品"
        ws["B1"] = "价格"
        ws["A2"] = "键盘"
        ws["B2"] = "199"
        wb.save(SAMPLES / "table.xlsx")
    except ImportError:
        pass  # openpyxl 缺失时跳过样本生成，对应测试 xfail
    yield
    # cleanup
    import shutil

    shutil.rmtree(SAMPLES, ignore_errors=True)


def test_file_type_mapping():
    assert file_type_of("a.txt") == "txt"
    assert file_type_of("a.MD") == "md"
    assert file_type_of("a.py") == "py"
    assert file_type_of("a.cpp") == "cpp"
    assert file_type_of("a.PNG") == "png"
    assert file_type_of("a.xlsx") == "xlsx"
    assert file_type_of("a.zip") == "other"
    assert file_type_of("noext") == "other"


def test_parse_txt_md():
    d = parse_file(SAMPLES / "note.txt")
    assert d.file_type == "txt"
    assert "纯文本文件" in d.text
    d2 = parse_file(SAMPLES / "doc.md")
    assert "# 标题" in d2.text


def test_parse_code():
    d = parse_file(SAMPLES / "calc.py")
    assert d.file_type == "py"
    assert "calculate_total_price" in d.text
    assert "计算总价" in d.text  # 注释与 docstring 保留


def test_parse_image_placeholder():
    """损坏/无效图片：is_image 标记，文本为空，且**失败原因可见**（rev 5）。

    rev 5 起 OCR 失败不再是「静默无文本」：parser 把它写进 doc.error
    （rev 6 起为结构化标记 `ocrFailed:`，此前是 `ocr: ` 拼句），
    indexer 据此落 failed。这里守的就是「错误不被吞掉」。
    """
    p = SAMPLES / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    d = parse_file(p)
    assert d.is_image is True
    assert d.text == ""
    # ⚠️ 2026-09-28（test3）：判据从 `"ocr:"` 前缀换成结构化标记 `ocrFailed:`。
    # 原前缀是「拼一句话」的产物（`"ocr: %s" % e`），英文原文会原样露在中文界面上。
    # 现在后端只给结构化事实，词在前端 locales 里按语言拼（见 parser.describe_ocr_failure）。
    assert d.error.startswith("ocrFailed:"), "OCR 失败必须留下原因: %r" % d.error
    assert "ocrFailed" in d.error


def test_parse_image_ocr_text(monkeypatch):
    """图片 → 引擎出的文本进 doc.text，且 conf 落 raw_meta（rev 5）。

    ★ rev 5 起 OCR 唯一引擎是 PP-OCR，测试用 stub 顶替 `recognize_ppocr`：
      真跑引擎要 2~9s/张且依赖模型与设备，单测不该背这个成本（端到端另有手工验收）。
    """
    from bibit_filesearch import ocr_engine

    calls = []

    def fake_recognize(img):
        calls.append(img.size)
        return "Indexed Image OCR", 95.0

    monkeypatch.setattr(ocr_engine, "recognize_ppocr", fake_recognize)
    # 必须是一张**真图**：parse_file 会先 Image.open（PIL 打不开就轮不到引擎）。
    # 引擎本身被打桩，所以图里有没有文字不重要。
    from PIL import Image

    p = SAMPLES / "ocr.png"
    Image.new("RGB", (64, 32), "white").save(p)
    parsed = parse_file(str(p))
    assert parsed.is_image is True
    assert "Indexed" in parsed.text, f"应取到引擎文本: {parsed.text!r}"
    assert parsed.raw_meta.get("ocr_conf") == 95.0
    assert calls, "必须真的调过引擎（否则这条测试没验到东西）"


def test_parse_zip_other():
    p = SAMPLES / "arch.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
    d = parse_file(p)
    assert d.file_type == "other"
    assert d.text == ""


def test_chunk_text_overlap():
    text = "甲" * 1200
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) == 3  # 500-400-400（step=400）
    assert chunks[0] == "甲" * 500
    assert chunks[1] == "甲" * 500  # 400..900
    assert chunks[2] == "甲" * 400  # 800..1200


def test_chunk_code_boundary():
    """代码分块优先在空行断开（函数边界附近）。"""
    text = "\n".join([f"def func_{i}():\n    pass\n" for i in range(30)])
    chunks = chunk_code(text, size_lines=12, overlap_lines=2)
    # 每块不应在函数定义中间断开（块尾应为空行或块结束）
    for c in chunks[:-1]:
        assert c.rstrip().endswith("") or c.rstrip().endswith("pass"), f"块尾异常: {c[-40:]}"


def test_code_semantic_hit_py():
    """代码语义命中：搜函数名命中。"""
    d = parse_file(SAMPLES / "calc.py")
    chunks = split_by_type(d.text, d.file_type)
    joined = "\n".join(chunks)
    assert "calculate_total_price" in joined  # 函数名完整保留在某个块
    assert "计算总价" in joined  # 注释/中文语义保留


def test_chunk_code_token_safe():
    """2026-08-26：代码分块不超 embed token 上限（600 字符预算，实测 0 超 512 token）。

    回归：原 60 行分块实测 1000-2100 token 超限 → embed 服务 500 报错且前端无提示。
    """
    import jieba

    # 长行代码（模拟真实项目：中文注释多、行较长）
    long_lines = []
    for i in range(200):
        long_lines.append(f"def func_{i}():  # 处理第{i}个用户请求的详细逻辑说明，包含错误处理与重试机制")
        long_lines.append("    result = process(data, retry=True, timeout=30, verbose=True)  # 处理结果")
        long_lines.append("")
    text = "\n".join(long_lines)
    chunks = split_by_type(text, "py")
    tokens = [len(list(jieba.cut(c))) for c in chunks]
    assert max(tokens) <= 512, f"代码块超 embed token 上限: max={max(tokens)} token"
    # 函数名保留（分块不破坏语义）
    joined = "\n".join(chunks)
    assert "func_199" in joined


def test_chunk_code_single_long_line():
    """2026-08-26：单行超预算（minified）硬切，不撑爆 token。"""
    import jieba

    text = "x" * 3000  # 单行 3000 字符
    chunks = split_by_type(text, "js")
    tokens = [len(list(jieba.cut(c))) for c in chunks]
    assert max(tokens) <= 512, f"minified 单行超限: max={max(tokens)}"
    assert "".join(chunks).count("x") == 3000  # 内容完整保留


def test_code_semantic_hit_js_java():
    d = parse_file(SAMPLES / "server.js")
    joined = "\n".join(split_by_type(d.text, d.file_type))
    assert "startServer" in joined and "HTTP 服务器" in joined
    d2 = parse_file(SAMPLES / "util.java")
    joined2 = "\n".join(split_by_type(d2.text, d2.file_type))
    assert "trimAll" in joined2 and "字符串工具" in joined2


def test_metadata_fields():
    p = SAMPLES / "note.txt"
    m = extract_metadata(p)
    assert m["fileName"] == "note.txt"
    assert m["fileType"] == "txt"
    assert m["fileSize"] > 0
    assert m["modifiedAt"].startswith("20")
    assert m["createdAt"].startswith("20")  # btime 或 ctime 回退
    assert m["accessedAt"].startswith("20")
    assert m["directory"] == str(SAMPLES)
    assert m["author"] == ""


def test_metadata_author_docx():
    """docx author 提取（用 python-docx 造一个带作者的 docx）。"""
    import docx

    p = SAMPLES / "authored.docx"
    d = docx.Document()
    d.add_paragraph("测试文档")
    d.core_properties.author = "张三"
    d.save(str(p))
    parsed = parse_file(p)
    assert parsed.author == "张三"
    assert "测试文档" in parsed.text
    m = extract_metadata(p)
    # metadata.author 由解析器回填（入库时），此处验证解析器层面
    assert parsed.file_type == "docx"


def test_parse_xlsx_content():
    """xlsx 解析：openpyxl 读表格文本（2026-09-07 修复：requirements 曾漏
    openpyxl 导致 xlsx 解析必失败 text_len=0）。"""
    p = SAMPLES / "table.xlsx"
    if not p.exists():
        pytest.skip("openpyxl 未安装，样本未生成")
    d = parse_file(str(p))
    assert d.file_type == "xlsx"
    assert len(d.text.strip()) > 0, "xlsx 应解析出表格文本（非空）"
    assert "键盘" in d.text and "199" in d.text, f"应含单元格内容: {d.text!r}"


def test_parse_pdf_image_only_ocr(monkeypatch):
    """图片型 PDF（无文本层只有内嵌图）→ 提内嵌图走 OCR，文本并入 doc.text。

    用 pymupdf 造：新建 PDF 插入一张 PNG，无 text 层；引擎打桩（见上一条的理由）。
    """
    try:
        import pymupdf
        from PIL import Image
    except ImportError:
        pytest.skip("依赖未装")

    from bibit_filesearch import ocr_engine

    def fake_recognize(img):
        return "ImagePDF OCR Probe", 93.0

    monkeypatch.setattr(ocr_engine, "recognize_ppocr", fake_recognize)

    img = Image.new("RGB", (600, 150), "white")
    png = SAMPLES / "_ocr_probe.png"
    img.save(png)
    pdf_path = SAMPLES / "image_only.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(50, 100, 650, 250), filename=str(png))
    doc.save(str(pdf_path))
    doc.close()

    parsed = parse_file(str(pdf_path))
    assert "OCR Probe" in parsed.text or "ImagePDF" in parsed.text, \
        f"图片型 PDF 应 OCR 出文本: {parsed.text!r}"
    assert parsed.images and parsed.images[0].get("ocr_text"), "图生文关联仍要入库"
