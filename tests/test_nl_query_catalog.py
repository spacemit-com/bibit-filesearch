"""NL 查询输入清单 → 可执行测试基准（2026-08-26，D-7 route 简化后）。

对应早期 `backend/learn/05-NL查询输入清单.md` 的 10 大类（**该文档未随本仓迁移**，
过程稿；本文件即其可执行化成果，编号只作历史锚点），每条断言：
- is_structured：filters 非空 或 unsupported 非空（替代旧 route 三分类）
- query：purify 后关键词（期望精确值，空串 = 纯筛选）
- filters：结构化条件（fileTypes / directories / dateRange 键存在性 + 关键值）

日期固定 2026-08-26（周三，本周一 = 2026-08-24）。
INDEX_DIRS 与生产一致（~/Desktop 等，动态 home——2026-09-01 nl_parser
DIR_MAP 改为 Path.home() 动态生成后，测试同步用 Path.home()，不硬编码 bianbu）。
"""
import sys
from pathlib import Path

import pytest


import datetime as dt  # noqa: E402

from bibit_filesearch import nl_parser as nl  # noqa: E402

FIXED_TODAY = dt.date(2026, 8, 26)  # 周三
_HOME = Path.home()
INDEX_DIRS = [
    str(_HOME / "Desktop"),
    str(_HOME / "Documents"),
    str(_HOME / "Downloads"),
    str(_HOME / "Pictures"),
    str(_HOME / "Videos"),
    str(_HOME / "Music"),
]
DOCS = str(_HOME / "Documents")


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(nl, "_today", lambda: FIXED_TODAY)


def _check(query, expected_is_structured, expected_kw=None, filters_has=None):
    """断言 parse_nl_query 输出（filters_has: {key: 期望值或子串}）。"""
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert p.is_structured is expected_is_structured, f"{query}: is_structured={p.is_structured}"
    if expected_kw is not None:
        assert p.query == expected_kw, f"{query}: kw={p.query!r} != {expected_kw!r}"
    if filters_has:
        for k, v in filters_has.items():
            assert k in p.filters, f"{query}: filters 缺 {k}（实际 {p.filters}）"
            if isinstance(v, dict):
                # dateRange 等：部分匹配（实际含 dateField 额外键）
                for sk, sv in v.items():
                    assert p.filters[k].get(sk) == sv, \
                        f"{query}: {k}.{sk}={p.filters[k].get(sk)} != {sv}（实际 {p.filters[k]}）"
            elif isinstance(v, list):
                for item in v:
                    assert item in p.filters[k], f"{query}: {k} 缺 {item}（实际 {p.filters[k]}）"
            elif v is not None:
                assert p.filters[k] == v, f"{query}: {k}={p.filters[k]} != {v}"


# ---------- 1. 纯内容语义（is_structured=False，filters 空） ----------

@pytest.mark.parametrize("query", [
    "GPU纹理压缩相关的文档",
    "网络配置笔记",
    "K3板卡调试记录",
    "项目周报模板",
    "关于内存泄漏的分析",
    "GTK的mapped状态切换机制",
    "wayland 合成器帧提交流程",
])
def test_plain_semantic(query):
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert p.is_structured is False, f"{query}: 纯语义不应结构化（filters={p.filters}）"
    assert p.filters == {}, f"{query}: 纯语义 filters 应为空"
    assert p.query, f"{query}: 纯语义 keywords 应非空"


# ---------- 2. 时间维度 ----------

@pytest.mark.parametrize("query,expected_start,expected_end", [
    ("今天", "2026-08-26", "2026-08-26"),
    ("昨天的文档", "2026-08-25", "2026-08-25"),
    ("前天的文档", "2026-08-24", "2026-08-24"),
    ("本周的文档", "2026-08-24", "2026-08-26"),
    ("这周的文档", "2026-08-24", "2026-08-26"),
    ("上周的文档", "2026-08-17", "2026-08-23"),
    ("本月的文档", "2026-08-01", "2026-08-26"),
    ("上月的文档", "2026-07-01", "2026-07-31"),
    ("今年的文档", "2026-01-01", "2026-08-26"),
    ("去年的文档", "2025-01-01", "2025-12-31"),
    ("最近一周的文档", "2026-08-20", "2026-08-26"),
    ("最近3天的文档", "2026-08-24", "2026-08-26"),
    ("3天前的文档", "2026-08-23", "2026-08-23"),
    ("两周前的文档", "2026-08-10", "2026-08-16"),  # N周前 = 那一整周（周一起始，非单日）
    ("一个月以来的文档", "2026-07-26", "2026-08-26"),
    # 2026-08-26 漏洞1/2 修复：具体日期/月份/年份
    ("3月的文档", "2026-03-01", "2026-03-31"),          # 具体月份（≠3个月前）
    ("3个月的文档", "2026-05-26", "2026-08-26"),        # 相对月数
    ("2025年的报告", "2025-01-01", "2025-12-31"),       # 年份
    ("2025年3月的报告", "2025-03-01", "2025-03-31"),    # 具体年月
    ("8月15号的笔记", "2026-08-15", "2026-08-15"),      # 具体日期
    ("1月1日", "2026-01-01", "2026-01-01"),             # 具体日期（日）
    ("3个月前", "2026-05-26", "2026-05-26"),            # 相对单日（前）
    ("3月前", "2026-05-26", "2026-05-26"),              # 相对单日（前，无"个"）
])
def test_time_dimension(query, expected_start, expected_end):
    _check(query, True, filters_has={
        "dateRange": {"start": expected_start, "end": expected_end},
    })


@pytest.mark.parametrize("query,expected_start,expected_end", [
    # P0-1（2026-08-26）：口语时间词已支持
    ("前两天修改的文档", "2026-08-24", "2026-08-25"),   # 前天+昨天
    ("前几天整理的东西", "2026-08-22", "2026-08-25"),   # 近四天（不含今天）
    ("这几天整理的东西", "2026-08-23", "2026-08-26"),   # 近三天（含今天）
    ("上上个月的项目计划", "2026-06-01", "2026-06-30"),  # 两个月前整月
    ("上上周的周报", "2026-08-10", "2026-08-16"),       # 两周前自然周
    ("去年底的资料", "2025-12-01", "2025-12-31"),       # 去年 12 月
    ("今年上半年的规划", "2026-01-01", "2026-06-30"),   # 1~6 月
    ("最近的文件", "2026-08-20", "2026-08-26"),         # 近七天
])
def test_time_gap(query, expected_start, expected_end):
    """P0-1 口语时间词（2026-08-26 前为缺口，现支持）。"""
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    dr = p.filters.get("dateRange")
    assert dr, f"{query}: 应产生 dateRange（实际 {p.filters}）"
    assert dr["start"] == expected_start and dr["end"] == expected_end, \
        f"{query}: dateRange={dr} != 期望 {expected_start}~{expected_end}"


# ---------- 3. 目录维度 ----------

@pytest.mark.parametrize("query,expected_dir", [
    ("桌面上的文档", "/root/Desktop"),
    ("桌面上", "/root/Desktop"),
    ("下载里的文件", "/root/Downloads"),
    ("图片里的照片", "/root/Pictures"),
    ("视频里的录像", "/root/Videos"),
    ("音乐里的歌", "/root/Music"),
    ("文档目录里的笔记", DOCS),
    ("在文档里找", DOCS),
    ("桌面和下载里的", "/root/Desktop"),
])
def test_dir_dimension(query, expected_dir):
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert p.is_structured is True, f"{query}: 目录应结构化"
    assert "directories" in p.filters, f"{query}: 应产生 directories"
    assert expected_dir in p.filters["directories"], f"{query}: 应含 {expected_dir}（实际 {p.filters['directories']}）"


def test_dir_location_priority():
    """漏洞4 修复：'X里的Y' 结构 → 目录只取前置 X（后置 Y 是内容词不是目录）。"""
    p = nl.parse_nl_query("下载里的视频", index_dirs=INDEX_DIRS)
    assert p.filters["directories"] == ["/root/Downloads"], \
        f"下载里的视频 → 目录应只含 Downloads（实际 {p.filters['directories']}）"
    p2 = nl.parse_nl_query("桌面上的图片", index_dirs=INDEX_DIRS)
    assert p2.filters["directories"] == ["/root/Desktop"], \
        f"桌面上的图片 → 目录应只含 Desktop（实际 {p2.filters['directories']}）"


@pytest.mark.parametrize("query,expected_kw", [
    ("项目里的配置文件", ""),   # 泛化目录词"项目"不映射路径；"配置文件"是类型词 → purify 剔除 → kw 空
    ("工作目录下的周报", "周报"),
    ("代码目录里的脚本", "脚本"),
    ("学习资料", ""),                   # 整体剔除（泛化名词）
    ("照片", ""),                       # 仅类型（图片），非目录
    ("截图", ""),                       # 类型映射（图片扩展名）
])
def test_dir_gap(query, expected_kw):
    """P0-2 目录泛化（2026-08-26）：泛化目录词（项目/工作/代码/资料）不映射路径
    （用户未配置对应目录 → 全库检索降级），但方位粘连剔除保证 kw 干净。"""
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert "directories" not in p.filters or not p.filters["directories"], \
        f"{query}: 泛化目录词不应产生 directories（实际 {p.filters}）"
    assert p.query == expected_kw, f"{query}: kw 应={expected_kw!r}（实际 {p.query!r}）"


# ---------- 4. 类型维度 ----------

@pytest.mark.parametrize("query,expected_type", [
    ("pdf文档", "pdf"),
    ("pdf文件", "pdf"),
    ("word文档", "docx"),
    ("word文件", "docx"),
    ("md文件", "md"),
    ("markdown文档", "md"),
    ("markdown文件", "md"),
    ("文本文件", "txt"),
    ("python脚本", "py"),
    ("py文件", "py"),
])
def test_type_dimension(query, expected_type):
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert "fileTypes" in p.filters, f"{query}: 应产生 fileTypes（实际 {p.filters}）"
    if expected_type:
        assert expected_type in p.filters["fileTypes"], \
            f"{query}: 应含 {expected_type}（实际 {p.filters['fileTypes']}）"


def test_type_generic_code_not_mapped():
    """中文"代码"是泛指词（无具体扩展名映射）→ 不产生 fileTypes，fallback 纯语义。"""
    p = nl.parse_nl_query("java代码", index_dirs=INDEX_DIRS)
    assert "fileTypes" not in p.filters, f"中文代码不应映射具体类型（实际 {p.filters}）"


def test_type_image_video_priority():
    """漏洞3 修复：图片/视频/照片 裸词 → 类型优先（fileTypes 多扩展名），目录让位。"""
    p = nl.parse_nl_query("图片", index_dirs=INDEX_DIRS)
    assert "png" in p.filters["fileTypes"] and "jpg" in p.filters["fileTypes"], \
        f"图片 → 应命中图片扩展名（实际 {p.filters.get('fileTypes')}）"
    assert "directories" not in p.filters, f"图片裸词 → 目录应让位（实际 {p.filters}）"
    p2 = nl.parse_nl_query("视频", index_dirs=INDEX_DIRS)
    assert "mp4" in p2.filters["fileTypes"], f"视频 → 应命中视频扩展名（实际 {p2.filters.get('fileTypes')}）"
    assert "directories" not in p2.filters, f"视频裸词 → 目录应让位（实际 {p2.filters}）"


def test_type_location_not_type():
    """漏洞3b：'X里的Y'结构中 X 是位置限定词（图片里的视频 → 图片=位置，不是类型）。"""
    p = nl.parse_nl_query("图片里的视频", index_dirs=INDEX_DIRS)
    assert p.filters["directories"] == ["/root/Pictures"], \
        f"图片里的视频 → 目录应含 Pictures（实际 {p.filters.get('directories')}）"
    assert "mp4" in p.filters["fileTypes"] and "png" not in p.filters["fileTypes"], \
        f"图片里的视频 → 类型应只含视频（实际 {p.filters.get('fileTypes')}）"


def test_unsupported_full_shortcut_kw_empty():
    """用户实测（2026-08-26）：全未索引短路时 kw 应为空（原始整句污染 debug 展示）。

    任务角度：未索引目录的查询 kw 无意义（调用方直接提示"暂未索引"不检索），
    前端弹窗应显示 kw 空 + 未索引提示。
    """
    # index_dirs 不含 Downloads
    idx = [d for d in INDEX_DIRS if "Downloads" not in d]
    p = nl.parse_nl_query("下载目录中的文件", index_dirs=idx)
    assert p.query == "", f"全未索引短路 kw 应为空（实际 {p.query!r}）"
    assert p.unsupported_dirs == ["/root/Downloads"], \
        f"应收集 Downloads 为未索引（实际 {p.unsupported_dirs}）"
    assert p.filters == {}, "全未索引短路 filters 应为空"
    # 部分未索引：保留内容词
    p2 = nl.parse_nl_query("下载里的视频", index_dirs=idx)
    assert p2.query == "视频", f"部分未索引应保留内容词（实际 {p2.query!r}）"
    assert p2.unsupported_dirs == ["/root/Downloads"]


@pytest.mark.parametrize("query,expected_type", [
    ("表格类的数据", "xlsx"),     # P1-1：表格 → xlsx
    ("表格", "xlsx"),
    ("幻灯片", "pptx"),           # P1-1：幻灯片 → pptx
    ("演示文稿", "pptx"),
    ("截图", "png"),              # P1-1：截图 → 图片扩展名
    ("配置文件", "json"),         # P1-1：配置 → json
])
def test_type_gap(query, expected_type):
    """P1-1 类型泛化（2026-08-26 前为缺口，现映射到 KNOWN_TYPES 内类型）。"""
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert "fileTypes" in p.filters, f"{query}: 应产生 fileTypes（实际 {p.filters}）"
    assert expected_type in p.filters["fileTypes"], \
        f"{query}: fileTypes 应含 {expected_type}（实际 {p.filters['fileTypes']}）"


# ---------- 5. 动作 / dateField ----------

@pytest.mark.parametrize("query,expected_field", [
    ("上周修改过的文档", "modified"),
    ("昨天创建的文档", "created"),
    ("今天新建的文件", "created"),
    ("上周更新过的文档", "modified"),
])
def test_action_datefield(query, expected_field):
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert p.is_structured is True, f"{query}: 动作+时间应结构化"
    assert p.filters["dateRange"].get("dateField") == expected_field, \
        f"{query}: dateField={p.filters.get('dateRange', {}).get('dateField')} != {expected_field}"


@pytest.mark.parametrize("query,expected_field", [
    # P1-2（2026-08-26）：模糊近义词（最近/上次/刚刚/刚才）+ accessed 动作词 → dateRange + dateField
    ("最近打开过的文档", "accessed"),
    ("上次浏览过的页面", "accessed"),
    ("最近看过的视频", "accessed"),
    ("刚刚创建的文档", "created"),
    ("上次修改过的文档", "modified"),
])
def test_action_datefield_gap(query, expected_field):
    """P1-2 无数字时间词（2026-08-26 前为缺口）：模糊近义词触发 dateRange（近7天）+ dateField。"""
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    dr = p.filters.get("dateRange")
    assert dr, f"{query}: 应产生 dateRange（实际 {p.filters}）"
    assert dr.get("dateField") == expected_field, \
        f"{query}: dateField={dr.get('dateField')} != {expected_field}"


# ---------- 6. 祈使 / 语气 ----------

@pytest.mark.parametrize("query,expected_kw", [
    ("给我找出上周修改过的markdown", ""),        # 纯筛选（kw 空）
    ("帮我找上周的周报", "周报"),
    ("请帮我找到上周修改过的md文件", ""),          # 纯筛选
    ("帮我找一下昨天创建的word文档", ""),          # 纯筛选
    ("有哪些pdf", ""),                             # "有哪些"净化（含"有"补词）
])
def test_imperative(query, expected_kw):
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert p.is_structured is True, f"{query}: 祈使句应结构化"
    assert p.query == expected_kw, f"{query}: kw={p.query!r} != {expected_kw!r}"


# ---------- 7. 组合查询 ----------

@pytest.mark.parametrize("query,expected_kw,filters_has", [
    ("一个月以来我们修改过的关于GPU的markdown文档", "gpu", {"fileTypes": ["md"], "dateRange": {"start": "2026-07-26", "end": "2026-08-26"}}),
    ("上周桌面上修改过的pdf文档", "", {"fileTypes": ["pdf"], "directories": ["/root/Desktop"], "dateRange": {"start": "2026-08-17", "end": "2026-08-23"}}),  # 漏洞4/5修复：方位残留剔除，kw 干净
    ("上周修改过的桌面的pdf文档", "", {"fileTypes": ["pdf"], "directories": ["/root/Desktop"], "dateRange": {"start": "2026-08-17", "end": "2026-08-23"}}),  # 用户前端实测样例（2026-08-26）：纯目录词"桌面"剔除
    ("下载里最近一周的周报", "周报", {"directories": ["/root/Downloads"], "dateRange": {"start": "2026-08-20", "end": "2026-08-26"}}),
    ("今天创建的照片", "", {"dateRange": {"start": "2026-08-26", "end": "2026-08-26"}}),  # 照片=类型词，purify 剔除 → kw 空（纯筛选）
    ("最近三个月GPU性能优化的文档", "gpu性能优化", {"dateRange": {"start": "2026-05-26", "end": "2026-08-26"}}),
])
def test_combined(query, expected_kw, filters_has):
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert p.is_structured is True, f"{query}: 组合查询应结构化"
    assert p.query == expected_kw, f"{query}: kw={p.query!r} != {expected_kw!r}"
    for k, v in filters_has.items():
        assert k in p.filters, f"{query}: filters 缺 {k}（实际 {p.filters}）"
        if isinstance(v, dict):
            for sk, sv in v.items():
                assert p.filters[k].get(sk) == sv, \
                    f"{query}: {k}.{sk}={p.filters[k].get(sk)} != {sv}（实际 {p.filters[k]}）"
        elif isinstance(v, list) and v:
            for item in v:
                assert item in p.filters[k], f"{query}: {k} 缺 {item}（实际 {p.filters[k]}）"
        elif v is not None:
            assert p.filters[k] == v, f"{query}: {k}={p.filters[k]} != {v}"


# ---------- 8. 短查询 ----------

@pytest.mark.parametrize("query,expected_structured,expected_kw", [
    ("pdf", True, ""),
    ("md", True, ""),
    ("ppt", True, ""),
    ("周报", False, "周报"),
    ("桌面", True, ""),   # 纯目录词剔除（2026-08-26：kw 残留污染 embed，filters.directories 已限定）
])
def test_short_query(query, expected_structured, expected_kw):
    _check(query, expected_structured, expected_kw)


# ---------- 9. 属性维度（parser 不处理 → 纯语义） ----------

@pytest.mark.parametrize("query", [
    "超过10MB的文件",
    "大文件",
    "张三写的文档",
    "最大的几个文件",
])
def test_attribute_dimension(query):
    """属性维度：parser 不处理（检索链路不支持大小/作者）→ 纯语义，不炸。"""
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)
    assert p.filters == {}, f"{query}: 属性维度不应产生 filters（实际 {p.filters}）"


# ---------- 10. 边界 / 异常输入（不炸） ----------

@pytest.mark.parametrize("query", [
    "",
    "？？？",
    "!!!",
    "   ",
    "a" * 500,
    "find pdf files",
    "GPU related md 文档",
    "有没有关于GPU的？",
])
def test_edge_inputs(query):
    """边界输入：解析不抛异常（route/plain/结构化均可），核心是「不炸」。"""
    p = nl.parse_nl_query(query, index_dirs=INDEX_DIRS)  # 不抛即通过
    assert isinstance(p.is_structured, bool)
    assert isinstance(p.filters, dict)
    assert isinstance(p.query, str)
