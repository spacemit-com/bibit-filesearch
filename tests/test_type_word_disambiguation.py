"""类型词消歧回归（2026-10-08）：类型词是 **keyword** 还是 **fileType filter**。

背景：原实现只认 `_TYPE_EN_WORDS` 白名单——"X语法"里 X 在表就当类型筛、不在就当内容词，
于是 "python源码剖析" 返回 0 条而 "java源码" 正常。用户实测暴露后定版规则：

  格式词（jpg/pdf/markdown/excel/表格/幻灯片/配置文件…）
      独立出现 / 句末 / 后跟"文件/文档/脚本/表格…" → fileTypes（中心语）
      后面还有别的内容词（"excel公式"）          → keywords（修饰语）
  语言名/数据格式名（python/js/c/cpp/java/rust/sql/json…）
      一律当 keywords；只有后跟"文件/脚本"才转 fileTypes

不变量：同一句式对不同类型一视同仁（不再按白名单分档）。
"""
import pytest

import bibit_filesearch.nl_parser as nl

KNOW = ["py", "js", "ts", "java", "cpp", "c", "cs", "go", "rs", "sql", "sh", "json"]


@pytest.mark.parametrize("lang", ["python", "py", "javascript", "js", "java", "cpp",
                                  "c++", "c", "cs", "go", "rust", "rs", "sql",
                                  "shell", "sh", "json"])
def test_lang_word_alone_is_keyword(lang):
    """语言名独立成词 → keyword（不再抢成 fileType）。"""
    p = nl.parse_nl_query(lang, index_dirs=["/root/Documents"])
    assert "fileTypes" not in p.filters, f"{lang}: 不应产生 fileTypes（实际 {p.filters}）"
    assert p.query == lang, f"{lang}: kw 应为 {lang!r}（实际 {p.query!r}）"


@pytest.mark.parametrize("query,ext", [
    ("python文件", "py"), ("py文件", "py"), ("python脚本", "py"),
    ("javascript文件", "js"), ("java文件", "java"), ("cpp文件", "cpp"),
    ("c文件", "c"), ("rust文件", "rs"), ("shell脚本", "sh"), ("json文件", "json"),
])
def test_lang_word_with_limit_is_filter(query, ext):
    """语言名后跟"文件/脚本"（限定词）→ fileType 筛选。"""
    p = nl.parse_nl_query(query, index_dirs=["/root/Documents"])
    assert p.filters.get("fileTypes") == [ext], \
        f"{query}: 应只筛 {ext}（实际 {p.filters.get('fileTypes')}）"


@pytest.mark.parametrize("query", [
    "python源码剖析", "python多线程编程实践", "python教程", "py爬虫",
    "java源码", "cpp教程", "go语法", "rust入门", "sql优化", "javascript语法",
])
def test_lang_word_with_content_is_keyword(query):
    """"语言名 + 内容词" → 整句当 keyword（这是用户报的 bug 形态）。"""
    p = nl.parse_nl_query(query, index_dirs=["/root/Documents"])
    assert "fileTypes" not in p.filters, f"{query}: 不应筛类型（实际 {p.filters}）"
    assert p.query == query, f"{query}: 整句应留在 kw（实际 {p.query!r}）"


def test_same_pattern_same_treatment():
    """不变量：同一句式（X + 内容词），格式词与语言名行为一致——都当 keyword。

    原实现下 "python源码剖析"（python 在 _TYPE_EN_WORDS）→ fileTypes=[py] + 0 条，
    而 "java源码"（java 不在表）→ keyword，同一句式两种待遇。
    """
    for x in ("python", "java", "cpp", "excel", "jpg", "markdown", "pdf"):
        p = nl.parse_nl_query(x + "语法", index_dirs=["/root/Documents"])
        assert "fileTypes" not in p.filters, \
            f"{x}语法: 与其它类型词应同待遇（实际 {p.filters}）"


@pytest.mark.parametrize("query,exts", [
    ("jpg", ["jpg"]), ("pdf", ["pdf"]), ("markdown", ["md"]), ("excel", ["xlsx"]),
    ("发票的jpg", ["jpg"]), ("有哪些pdf", ["pdf"]), ("pdf文档", ["pdf"]),
    ("markdown文档", ["md"]), ("excel表格", ["xlsx"]), ("表格", ["xlsx"]),
    ("表格类", ["xlsx"]), ("幻灯片", ["pptx"]), ("配置文件", ["json"]),
    ("文本", ["txt"]), ("word", ["docx"]), ("ppt", ["pptx"]),
])
def test_format_word_stays_filter(query, exts):
    """格式词独立/句末/后跟限定词 → 仍是 fileType 筛选（用户实测要保留的能力）。"""
    p = nl.parse_nl_query(query, index_dirs=["/root/Documents"])
    for e in exts:
        assert e in (p.filters.get("fileTypes") or []), \
            f"{query}: 应筛 {e}（实际 {p.filters.get('fileTypes')}）"


def test_image_word_keeps_all_exts():
    """"图片"一个词映射多个扩展名，中心语判定不能把它压成单个。"""
    p = nl.parse_nl_query("图片", index_dirs=["/root/Documents"])
    fts = p.filters.get("fileTypes") or []
    assert {"png", "jpg", "jpeg"} <= set(fts), f"图片: 扩展名不全（实际 {fts}）"
    p2 = nl.parse_nl_query("视频", index_dirs=["/root/Documents"])
    assert "mp4" in (p2.filters.get("fileTypes") or []), "视频: 应含 mp4"


def test_tail_type_word_not_stolen_by_cn_phrase():
    """句尾类型词不被中文长词组抢占：\"文档markdown\" 应是 md，不是 docx。"""
    assert nl._extract_types("文档markdown") == ["md"]
    assert nl._extract_types("资料excel") == ["xlsx"]
    assert nl._extract_types("笔记md") == ["md"]


# ---------- 并列多类型（2026-10-09 修 C：0.1.29 回归，0.1.28 行为）----------
#
# 0.1.29 第一版把「最后一个词的后缀是不是限定词」当中心语判据，
# 于是并列里只有最后一个类型词算中心语：`pdf、md、txt 文档` 只剩 txt。
# 0.1.28 是全收。下面钉住"全收"这个不变量。

@pytest.mark.parametrize("query,exts", [
    # 限定词后缀只挂在最后一个并列项 —— 三种都要收
    ("pdf、md、txt 文档", ["pdf", "md", "txt"]),
    ("pdf，md，txt 文档", ["pdf", "md", "txt"]),
    ("pdf、md 文档", ["pdf", "md"]),
    ("pdf或md文件", ["pdf", "md"]),
    ("excel和word文件", ["xlsx", "docx"]),
    ("ppt和excel表格", ["xlsx", "pptx"]),
    # 两个并列段各带限定词
    ("python文件和java文件", ["py", "java"]),
    ("pdf、md、txt 文档和表格", ["pdf", "md", "txt", "xlsx"]),
])
def test_parallel_types_with_limiter_all_collected(query, exts):
    """并列类型 + 限定词后缀 → 全部当中心语（不只最后一个）。"""
    got = nl._extract_types(query)
    assert set(got) == set(exts), f"{query}: 应={exts}（实际 {got}）"


def test_parallel_types_stop_at_content_word():
    """并列传播遇到内容词即断：'Python源码剖析和另一个word文件' 只收 docx。

    "源码剖析"是内容词（非连接词），它把 python 和 word 隔开 → 不算并列。
    """
    assert nl._extract_types("Python源码剖析和另一个word文件") == ["docx"]


def test_bare_parallel_types_do_not_all_become_filters():
    """裸并列（无限定词）仍按位置判据收**最后一个**（0.1.29 定稿行为，本轮不变）。

    'excel和word' 没有限定词，属"句末类型词=中心语"，只收 word。
    这条是钉子不是缺陷：想两个都要，写 'excel和word文件'。
    """
    assert nl._extract_types("excel和word") == ["docx"]
    assert nl._extract_types("pdf和md") == ["md"]


def test_limiter_inside_span_detected():
    """限定词落在类型词**跨度内部**（中文表条目 "word文件"）也要判出中心语。

    这是并列传播能否触发的前提：'excel和word文件' 里 word 的跨度是
    "word文件"（6 字），限定词在跨度内而非其后。
    """
    q = "excel和word文件".lower()
    hits = nl._type_matches(q)
    assert nl._followed_by_limiter(q, hits[-1][0], hits[-1][1]), \
        "word文件: 限定词在跨度内，应判出后跟限定词"


def test_limiter_after_trailing_space_detected():
    """'txt 文档'（限定词隔了空格）也算后跟限定词。"""
    q = "pdf、md、txt 文档".lower()
    hits = nl._type_matches(q)
    assert nl._followed_by_limiter(q, hits[-1][0], hits[-1][1]), \
        "txt 文档: 应判出后跟限定词"


# ---------- 修 1（2026-10-09）：英文文件性名词 ----------
#
# 原表 _TYPE_HEAD_LIMITERS 只有中文，于是中文并列收全三种、同一句式的英文只收最后一个：
#   pdf、md、txt 文档        -> ['pdf','md','txt']
#   pdf, md, txt documents  -> ['txt']        ← 判据不分语种，表却分
# 补 _TYPE_HEAD_LIMITERS_EN 后两侧一致。

@pytest.mark.parametrize("query,exts", [
    ("pdf, md, txt documents", ["pdf", "md", "txt"]),
    ("pdf and md files", ["pdf", "md"]),
    ("pdf or md files", ["pdf", "md"]),
    ("excel and word documents", ["xlsx", "docx"]),
    ("python and java files", ["py", "java"]),
])
def test_parallel_types_english_all_collected(query, exts):
    """英文并列 + 文件性名词 → 全部收（与中文侧一致）。"""
    assert set(nl._extract_types(query)) == set(exts), \
        f"{query}: 应={exts}（实际 {nl._extract_types(query)}）"


@pytest.mark.parametrize("query,exts", [
    ("pdf documents", ["pdf"]),
    ("word file", ["docx"]),
    ("python files", ["py"]),
])
def test_single_type_english_noun_is_filter(query, exts):
    """单个英文类型词 + 文件性名词 → 筛类型。"""
    assert nl._extract_types(query) == exts


def test_english_noun_needs_word_boundary():
    """英文名词按整词匹配：'file' 不得命中 'filetype' / 'profile'。"""
    assert nl._extract_types("filetype") == []
    assert nl._extract_types("profile") == []
    # 类型词自己的命中也不因这两词改变角色
    p = nl.parse_nl_query("filetype", index_dirs=["/root/Documents"])
    assert p.query == "filetype"


# ---------- 修 4（2026-10-09）：残留连接词吞召回 ----------
#
# 「pdf或md文件」净化后 kw='或' → 从纯筛选退化成带内容词的语义检索，同 filters 下
# 召回 3 掉到 1；「pdf和md文件」因"和"本就在停用词表而正常。补齐按整词丢弃。

@pytest.mark.parametrize("query", [
    "pdf或md文件", "pdf及md文件", "pdf和md文件", "pdf或者md文件", "pdf and md files",
])
def test_connector_residue_does_not_leak_into_keywords(query):
    """类型词之间只剩连接词时，净化后 keywords 必须为空（纯筛选）。"""
    p = nl.parse_nl_query(query, index_dirs=["/root/Documents"])
    assert p.query == "", f"{query}: kw 应为空（实际 {p.query!r}）"


@pytest.mark.parametrize("query,keep", [
    ("普及率", "普及率"),            # "及" 不是独立词 → 不得按 substring 拆
    ("涉及内存的问题", "涉及内存"),   # "及" 同上
    ("world map", "world map"),     # 英文 "or" 按整词丢 → 不得切 "world"
    ("information retrieval", "information"),  # "or" 在词内，不切
])
def test_connector_purify_spares_non_word_substrings(query, keep):
    """连接词净化只丢独立成词者，不切内容词的 substring。"""
    p = nl.parse_nl_query(query, index_dirs=["/root/Documents"])
    assert keep in p.query, f"{query}: {keep!r} 应保留（实际 {p.query!r}）"


def test_english_connector_gap_ignores_spaces():
    """英文并列的间隙含空格（' and '）也要判成'只有连接词'。"""
    assert nl._only_connectors(" and ")
    assert nl._only_connectors("或")
    assert not nl._only_connectors("")      # 相邻类型词不成并列
    assert not nl._only_connectors(" md ")  # 纯空白/纯类型词不算


