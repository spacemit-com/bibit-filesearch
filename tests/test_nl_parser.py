"""NL 解析层测试（D1 时间全模式 / D2 dateField / D3 关键词净化）。

覆盖：
- _parse_time_expr：时间表达式全模式（含中文数字、以来/内/里/近/这、N月/周/年前单日）
- _rule_extract：类型词边界匹配（"py" 不误命中 "python"）、dateField 动作词
- is_structured（D-7 2026-08-26）：route 三分类废弃后，布尔标记"是否提取到结构化条件"
- _purify_keywords：示例 query → GPU（D3）
- parse_nl_query：确定性解析（生产路径，不调 LLM）；parse_nl_query_llm：LLM 路径（调试端点）

测试日期固定为 2026-08-25（周二，本周一 = 2026-08-24）。
"""
import sys
from pathlib import Path

import pytest


import datetime as dt  # noqa: E402

from bibit_filesearch import nl_parser as nl  # noqa: E402

FIXED_TODAY = dt.date(2026, 8, 25)  # 周二；本周一 2026-08-24


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(nl, "_today", lambda: FIXED_TODAY)


# ---------- D1：时间表达式全模式 ----------

@pytest.mark.parametrize("expr,expected", [
    ("今天", {"start": "2026-08-25", "end": "2026-08-25"}),
    ("昨天", {"start": "2026-08-24", "end": "2026-08-24"}),
    ("前天", {"start": "2026-08-23", "end": "2026-08-23"}),
    ("本周", {"start": "2026-08-24", "end": "2026-08-25"}),
    ("上周", {"start": "2026-08-17", "end": "2026-08-23"}),
    ("本月", {"start": "2026-08-01", "end": "2026-08-25"}),
    ("上月", {"start": "2026-07-01", "end": "2026-07-31"}),
    ("今年", {"start": "2026-01-01", "end": "2026-08-25"}),
    ("去年", {"start": "2025-01-01", "end": "2025-12-31"}),
    ("最近一周", {"start": "2026-08-19", "end": "2026-08-25"}),
    ("最近3天", {"start": "2026-08-23", "end": "2026-08-25"}),
    ("3天前", {"start": "2026-08-22", "end": "2026-08-22"}),
    ("3周前", {"start": "2026-08-03", "end": "2026-08-09"}),
    # D1 新增
    ("一个月以来", {"start": "2026-07-25", "end": "2026-08-25"}),
    ("近一个月", {"start": "2026-07-25", "end": "2026-08-25"}),
    ("最近三个月", {"start": "2026-05-25", "end": "2026-08-25"}),
    ("三个月内", {"start": "2026-05-25", "end": "2026-08-25"}),
    ("两周里", {"start": "2026-08-12", "end": "2026-08-25"}),
    ("一年以来", {"start": "2025-08-25", "end": "2026-08-25"}),
    ("一个月前", {"start": "2026-07-25", "end": "2026-07-25"}),
    ("两个月前", {"start": "2026-06-25", "end": "2026-06-25"}),
    ("一年前", {"start": "2025-08-25", "end": "2025-08-25"}),
    ("半年来", {"start": "2026-02-24", "end": "2026-08-25"}),  # 182 天
    ("十天内", {"start": "2026-08-16", "end": "2026-08-25"}),
])
def test_parse_time_expr(expr, expected):
    assert nl._parse_time_expr(expr) == expected


def test_parse_time_expr_month_end_safety():
    """月末边界：3-31 减 1 月 → 2-28（2026 非闰年）；闰年 2028-02-29 减 1 年 → 2027-02-28。"""
    assert nl._add_months(dt.date(2026, 3, 31), -1) == dt.date(2026, 2, 28)
    assert nl._add_months(dt.date(2028, 2, 29), -12) == dt.date(2027, 2, 28)
    assert nl._add_months(dt.date(2026, 8, 31), 1) == dt.date(2026, 9, 30)


def test_parse_time_expr_unknown():
    assert nl._parse_time_expr("不存在的时间") is None
    assert nl._parse_time_expr("") is None


# ---------- D2：dateField 动作词 ----------

@pytest.mark.parametrize("query,expected", [
    ("创建于上周的文档", "created"),
    ("新建的表格", "created"),
    ("访问过的pdf", "accessed"),
    ("昨天打开过的文档", "accessed"),
    ("修改过的文档", "modified"),
    ("上周更新过的报告", "modified"),
])
def test_date_field_of(query, expected):
    assert nl._date_field_of(query) == expected


# ---------- 类型词边界 ----------

def test_extract_types_boundary():
    """短扩展名词边界：'py' 不误命中 'python'；'md' 不误命中 'markdown' 的子串位置。"""
    assert nl._extract_types("关于GPU的markdown文档") == ["md"]
    # 2026-10-08 中心语判据：'python' 后跟内容词（多线程编程实践）→ 是修饰语，
    # 留在 keywords 里当内容词搜，不进 fileTypes。
    assert nl._extract_types("python 多线程编程实践") == []
    # 'python' 后跟"文件"（限定词）→ 中心语；'pdf' 后跟"文件" → 中心语
    assert nl._extract_types("python 的 pdf 文件") == ["pdf"]
    assert nl._extract_types("python文件") == ["py"]
    assert nl._extract_types("GPU纹理压缩相关的文档") == []
    assert nl._extract_types("pdf文档") == ["pdf"]
    assert nl._extract_types("word文档") == ["docx"]


def test_extract_types_head_vs_modifier():
    """2026-10-08 中心语判据：类型词是中心语 → 进 fileTypes；是修饰语 → 留在 keywords。

    修的是"同一个句式对不同类型待遇不一致"——原实现只认 _TYPE_EN_WORDS 白名单，
    于是 "python源码剖析"（python 在表）/ "java源码"（java 不在表）走向两个极端。
    """
    # 中心语：类型词独立 / 句末 / 后跟文件性名词
    for q, want in [("jpg", ["jpg"]), ("发票的jpg", ["jpg"]), ("有哪些pdf", ["pdf"]),
                    ("python文件", ["py"]), ("python脚本", ["py"]), ("表格类", ["xlsx"]),
                    ("markdown文档", ["md"]), ("excel表格", ["xlsx"])]:
        assert nl._extract_types(q) == want, f"{q}: {nl._extract_types(q)} != {want}"
    # 修饰语：类型词之后还有内容词 → 不当筛选（语言名与格式名一视同仁）
    for q in ("python源码剖析", "java源码", "cpp教程", "go语法", "excel公式",
              "jpg压缩", "markdown语法", "pdf解析库", "rust入门", "sql优化"):
        assert nl._extract_types(q) == [], f"{q}: 应留在 keywords，实际 {nl._extract_types(q)}"


# ---------- D3：关键词净化 ----------

def test_purify_example_query():
    """用户示例 query：一个月以来我们修改过的关于GPU的markdown文档 → GPU。"""
    rules = {"fileTypes": ["md"], "_time_expr_raw": "一个月以来", "dateField": "modified"}
    assert nl._purify_keywords("一个月以来我们修改过的关于GPU的markdown文档", "", rules) == "gpu"


def test_purify_empty_returns_empty_string():
    """净化后为空 → 返回空串（2026-08-26 修正：不再回退原始 query；函数名同步 2026-09-04）。

    全约束词查询（"给我找出上周修改过的markdown" → fileTypes+dateRange 已进 filters）
    关键词应为空串 = 纯筛选检索；回退整句会让祈使词污染 embed。
    """
    rules = {"fileTypes": ["pdf"], "_time_expr_raw": "上周"}
    assert nl._purify_keywords("上周修改过的pdf文档", "", rules) == ""
    # 有内容词的场景仍正常净化
    rules2 = {"fileTypes": ["md"], "_time_expr_raw": "一个月以来"}
    assert nl._purify_keywords("一个月以来我们修改过的关于GPU的markdown文档", "", rules2) == "gpu"


def test_purify_keeps_dir_word():
    """目录词（视频/图片/音乐）不剔除：双义词也是内容词，剔除会丢语义。"""
    rules = {"dirs": ["/root/Videos"], "_time_expr_raw": "昨天"}
    r = nl._purify_keywords("昨天看过的GPU视频", "", rules)
    assert "视频" in r
    assert "GPU" in r.lower() or "gpu" in r


def test_purify_english_keywords_kept_intact():
    """英文关键词净化（2026-09-04 英文支持）：停用词只整词删除，不拦腰切断英文内容词。

    回归：原 substring replace 会把 theme→me、proof→pro（the/of 子串被删），
    英文关键词查询被静默腐蚀。词边界匹配后 theme/proof/today 等完整保留。
    """
    assert nl._purify_keywords("GTK theme engine", "", {}) == "gtk theme engine"
    assert nl._purify_keywords("proof of concept", "", {}) == "proof concept"
    assert nl._purify_keywords("the wayland surface protocol", "", {}) == "wayland surface protocol"
    assert nl._purify_keywords("GPU texture compression", "", {}) == "gpu texture compression"
    assert nl._purify_keywords("Kata containers", "", {}) == "kata containers"


# ---------- 结构化判定（route 简化 2026-08-26：route 三分类 → is_structured 布尔） ----------

@pytest.mark.parametrize("query,expected", [
    ("一个月以来我们修改过的关于GPU的markdown文档", True),   # 时间命中 → 结构化
    ("上周桌面上修改过的pdf文档", True),                      # 时间+目录+类型
    ("这周修改的文档", True),                                 # "这周"（D1 补齐）
    ("关于GPU的markdown文档", True),                          # 仅类型 → 结构化
    # 2026-10-08 中心语判据：'python' 后跟内容词 → 当内容词（不再产生 fileTypes）
    ("python 多线程编程实践", False),
    ("python源码剖析", False),
    ("java源码", False),
    ("python文件", True),     # 后跟"文件" → 类型筛选
    # 2026-10-08：裸 python 是语言名 → 当内容词（与 java/c/cpp 一致），不再产生 fileTypes
    ("python", False),
    ("GPU纹理压缩相关的文档", False),                         # 无结构化条件 → 纯语义（旧 plain）
    ("周报", False),
    ("pdf", True),      # 2026-08-25 审查修正：类型词不设长度门槛
    ("md", True),
    ("ppt", True),
])
def test_is_structured(query, expected):
    assert nl.parse_nl_query(query).is_structured is expected


@pytest.mark.parametrize("query,index_dirs,expected", [
    # 2026-08-25 审查修正：parse 同 index_dirs 口径
    ("桌面上的文档", ["/root/Documents"], True),   # 目录词全未索引 → unsupported 非空 → 结构化
    ("桌面上的文档", ["/root/Desktop"], True),     # 目录已索引 → 结构化
    ("桌面", ["/root/Desktop"], True),             # 超短目录查询也结构化（dirs 命中 filters 非空；旧 route 的 plain 是长度<4 特判，已废弃）
    ("桌面和下载的文档", ["/root/Desktop"], True),  # 部分未索引 → 结构化
])
def test_is_structured_with_index_dirs(query, index_dirs, expected):
    assert nl.parse_nl_query(query, index_dirs=index_dirs).is_structured is expected


def test_is_structured_property_definition():
    """is_structured = filters 非空 或 unsupported 非空（替代 route 三分类）。"""
    # filters 空 + unsupported 空 → False（纯语义）
    p = nl.ParseResult(query="GPU纹理压缩相关", filters={}, unsupported_dirs=[])
    assert p.is_structured is False
    # filters 有类型 → True
    p2 = nl.ParseResult(query="pdf", filters={"fileTypes": ["pdf"]}, unsupported_dirs=[])
    assert p2.is_structured is True
    # unsupported 非空 → True（未索引目录提示路径）
    p3 = nl.ParseResult(query="下载里的视频", filters={}, unsupported_dirs=["/root/Downloads"])
    assert p3.is_structured is True


# ---------- parse_nl_query 集成（规则层兜底，不调 LLM） ----------

def test_parse_nl_query_example():
    """示例 query 完整解析：dateRange + dateField + fileTypes + 净化关键词。"""
    r = nl.parse_nl_query("一个月以来我们修改过的关于GPU的markdown文档",
                          index_dirs=["/root/Desktop", "/root/Documents"])
    assert r.query == "gpu"
    assert r.filters["fileTypes"] == ["md"]
    assert r.filters["dateRange"]["start"] == "2026-07-25"
    assert r.filters["dateRange"]["end"] == "2026-08-25"
    assert r.filters["dateRange"]["dateField"] == "modified"
    assert r.unsupported_dirs == []


def test_parse_nl_query_created_field():
    """D2：'创建' → dateField=created。"""
    r = nl.parse_nl_query("上周创建的pdf文档")
    assert r.filters["dateRange"]["dateField"] == "created"
    assert r.filters["dateRange"]["start"] == "2026-08-17"


def test_parse_nl_query_unsupported_dirs():
    """目录全未索引 → 快速返回暂未提示。"""
    r = nl.parse_nl_query("上周桌面上的文档", index_dirs=["/root/Documents"])
    assert r.unsupported_dirs == ["/root/Desktop"]
    assert r.filters == {}


def test_parse_nl_query_partial_unsupported():
    """部分目录未索引 → 不短路：已索引目录进 filters，未索引目录进 unsupported。

    回归：旧实现 `if unsupported_dirs` 一刀切短路，部分未索引时丢弃时间/类型等有效条件。
    确定性化：不调 LLM，部分未索引仍保留有效条件。
    """
    r = nl.parse_nl_query("桌面和下载里的文档",
                          index_dirs=["/root/Desktop"])  # 桌面已索引，下载未索引
    assert r.filters["directories"] == ["/root/Desktop"]
    assert r.unsupported_dirs == ["/root/Downloads"]
