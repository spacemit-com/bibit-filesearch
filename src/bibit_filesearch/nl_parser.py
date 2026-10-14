"""NL 意图解析层：自然语言查询 → 结构化检索条件。

链路：query → 规则层优先（时间/目录/类型确定性转换）→ LLM仅抽取 keywords
      → 关键词净化（purify）→ 拼装成与 /api/search 相同的 filters 结构。

关键设计：
- 结构化条件以规则层为准（确定性，避免 LLM 幻觉污染检索）
- **D-7（2026-08-26）：route 三分类（nl/rule/plain）已废弃**——/api/search 统一走
  parse_nl_query（3ms 确定性），is_structured 布尔标记"是否提取到结构化条件"；
  plain = 零结构化条件的结构化（同一函数同一条路径，filters={} 时退化为纯语义 embed+BM25）
- 目录限制：解析出的目录仅支持用户配置的 index_dirs；未索引目录 → unsupported_dirs
  （调用方回复"暂未索引"）；**全未索引时 kw 净化后返回**（无需为必然
  "暂未索引"的结果付 LLM 成本）
- 降级：LLM 失败/超时/输出非法 → 规则层结果兜底，最坏等于纯关键词语义检索
- 时间表达式由规则层计算（datetime 精确算，不交给 LLM）；"文档"是弱目录词，
  仅位置语境（文档目录/文档里/在文档等）才算目录
- **D1 时间全模式**：N个月/周/年 的 以来/内/里/近/这 全模式 + 中文数字
  （一/两/三/…/半），如"一个月以来"、"最近三个月"、"两周里"、"这周"、"半年来"
- **D2 dateField**：动作词 → 时间戳列——修改过/更新过 → modified（默认）、
  创建/新建 → created、访问/打开过 → accessed（filters.dateRange.dateField）
- **D3 关键词净化**：prompt 收紧 + 规则层 scrub（剔除已结构化提取的
  类型/时间/目录/动作词与语气词），如"一个月以来我们修改过的关于GPU的markdown文档" → keywords=GPU
- **P0/P1（2026-08-26）**：时间口语（前两天/上上个月/去年底/今年上半年/最近）+ 目录泛化
  （项目/工作不映射路径，方位剔除保 kw 干净）+ 类型泛化（表格/幻灯片/截图/配置文件）
  + accessed 无时间词（最近打开过 → 近7天）
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import logging
import re
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("bibit.nl")


# NL 解析缓存（2026-08-17 优化：相同整句查询重复调 LLM ~2s，浪费端侧推理）。
# 简单 LRU：key = (query, index_dirs tuple)；value = (解析结果, 日期)。
# 相对时间表达式（昨天/本周/最近N天等）结果与日期相关 → 按天失效（缓存日期 ≠ 今天则重解析）。
_NL_CACHE: dict = {}
_NL_CACHE_MAX = 128
_NL_CACHE_LOCK = threading.Lock()

PARSE_SYSTEM = (
    "你是文件搜索查询解析器。把用户的自然语言文件搜索查询解析为结构化 JSON，"
    "只输出 JSON，不要输出其他文字。JSON 字段："
    '{"keywords": 核心检索关键词字符串（只保留用于语义检索的内容词，中文输出；'
    '去掉时间表达式、文件类型、目录、动作词（修改/创建/访问等）和语气词（我们/帮我/请/相关/关于等）；'
    '不要翻译成英文；没有内容词则空串）, '
    '"fileTypes": 文件类型数组（可选值: txt md pdf docx py js java cpp c cs go rs ts sql sh json png jpg jpeg gif webp bmp svg xlsx pptx，没有则[]）, '
    '"timeExpr": 时间表达式字符串（今天/昨天/前天/本周/上周/本月/上月/最近N天/N天前/最近一周/一个月以来等，没有则null）, '
    '"dirs": 目录关键词数组（中文：桌面/文档/下载/图片/视频/音乐 等，没有则[]）}'
)

def _home_dirs() -> dict[str, str]:
    """中文目录词 → 当前用户 home 下的实际路径（动态，不硬编码用户名）。

    2026-09-01：原硬编码 /home/<user>/... 在其他用户名（root 等）下解析
    "桌面/下载/图片"会指向不存在目录——改为 Path.home() 动态生成。
    """
    home = Path.home()
    return {
        "桌面": str(home / "Desktop"),
        "桌面上": str(home / "Desktop"),
        "下载": str(home / "Downloads"),
        "图片": str(home / "Pictures"),
        "视频": str(home / "Videos"),
        "音乐": str(home / "Music"),
    }


DIR_MAP = _home_dirs()

# "文档"是弱目录词：仅在位置语境才当目录（"pdf文档"里的"文档"是类型说法，不是目录）
_WEAK_DIR_WORDS = {"文档": str(Path.home() / "Documents")}
_WEAK_DIR_PAT = re.compile(r"(文档目录|文档文件夹|文档里|文档中|文档内|在文档|Documents)")

# ---------- D1：中文数字 / 日期计算 ----------

_CN_NUM = {"一": 1, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
           "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}
_NUM_RE = r"[0-9]{1,3}|一|两|二|三|四|五|六|七|八|九|十|半"


def _num_val(s: str) -> float | None:
    """中文/阿拉伯数字 → 数值（'半'=0.5）。"""
    if s.isdigit():
        return int(s)
    return _CN_NUM.get(s)


def _add_months(d: dt.date, n: int) -> dt.date:
    """d 加 n 个月（月末安全：3-31 减 1 月 → 2-28/29，闰年 2-29 正确）。"""
    total = d.year * 12 + (d.month - 1) + n
    y, m = divmod(total, 12)
    m += 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


# ---------- D2：动作词 → 时间戳字段 ----------

# 动作词 → dateRange.dateField；匹配顺序优先 created/accessed，其余默认 modified
_ACTION_FIELDS: list[tuple[tuple[str, ...], str]] = [
    (("创建", "新建", "生成", "建立", "产生"), "created"),
    (("访问", "打开过", "打开", "浏览", "看过", "阅读过", "使用过", "用过"), "accessed"),
]
# 动作词全集（purify 时从 keywords 剔除；modified 系动词也是约束词不是内容词）
_ACTION_WORDS = ("创建", "新建", "生成", "建立", "产生", "访问", "打开", "浏览", "看过",
                 "阅读", "使用", "修改", "更新", "编辑", "改动", "变更", "保存", "写过", "改过")


def _date_field_of(query: str) -> str:
    for kws, field in _ACTION_FIELDS:
        if any(k in query for k in kws):
            return field
    return "modified"


def _dir_word_hits(query: str, index_dirs: list[str] | None = None) -> list[str]:
    """从 query 提取目录路径。强词直接命中；弱词（文档）要求位置语境。

    仅保留**用户已配置的索引目录**（index_dirs）：解析出的目录若不在索引范围内，
    不进返回（由调用方收集为 unsupported_dirs，向前端提示"暂未索引"）。
    index_dirs 为 None 时保持旧行为（不过滤）。

    2026-08-26 漏洞4 消歧（方案 B）：
    - "X里的Y"结构（下载里的视频/桌面上传的图片）：前置 X 是目录限定词，
      后置 Y（视频/图片）是内容词**不是目录**——只取前置目录，避免双目录命中。
      （如"下载里的视频" → 目录=[Downloads]，不把 Videos 也加进来）
    - 裸目录词（"图片"/"视频"单独）→ 目录命中（与类型词并存，由调用方/检索层处理）
    """
    # "X里的Y" / "X上的Y" / "X中Y"：前置目录词 + 后置内容词
    # 例：下载里的视频 → 前置"下载"命中 Downloads，后置"视频"不命中目录
    for pat in (r"([\u4e00-\u9fa5]{2,3})里(?:的)?([\u4e00-\u9fa5]{2,4})",
                r"([\u4e00-\u9fa5]{2,3})上(?:的)?([\u4e00-\u9fa5]{2,4})",
                r"([\u4e00-\u9fa5]{2,3})中(?:的)?([\u4e00-\u9fa5]{2,4})"):
        m = re.search(pat, query)
        if m and m.group(1) in DIR_MAP:
            # 前置是目录词 → 只取前置，后置不命中目录
            hits = [DIR_MAP[m.group(1)]]
            # 前置"下载"→Downloads；后置"视频"是内容词不命中目录
            return [p for p in hits if p in index_dirs] if index_dirs is not None else hits
    hits = [p for k, p in DIR_MAP.items() if k in query]
    if _WEAK_DIR_PAT.search(query) and _WEAK_DIR_WORDS["文档"] not in hits:
        hits.append(_WEAK_DIR_WORDS["文档"])
    hits = list(dict.fromkeys(hits))
    if index_dirs is None:
        return hits
    return [p for p in hits if p in index_dirs]


def _today() -> dt.date:
    return dt.date.today()


# ---------- D1：时间表达式（全模式） ----------

# 精确词表（含 2026-08-25 补齐的"这"系列：这周/这月/这年/这个星期）
# 2026-08-26 P0-1：口语时间词（前两天/这几天/上上个月/上上周/去年底/今年上半年/最近）
_EXACT_TIME_EXPRS: dict[str, str] = {
    "今天": "今天",
    "昨天": "昨天",
    "前天": "前天",
    "前两天": "前两天",
    "前几天": "前几天",
    "这几天": "这几天",
    "最近": "最近",
    "本周": "本周",
    "这周": "本周",
    "这个星期": "本周",
    "这星期": "本周",
    "上周": "上周",
    "上上周": "上上周",
    "本月": "本月",
    "这个月": "本月",
    "这月": "本月",
    "上月": "上月",
    "上个月": "上月",
    "上上个月": "上上个月",
    "今年": "今年",
    "本年": "今年",
    "这年": "今年",
    "去年": "去年",
    "上一年": "去年",
    "去年底": "去年底",
    "今年上半年": "今年上半年",
    "上半年": "今年上半年",
}

# 相对区间/单日模式：(近|最近)?N(个)?(天|月|星期|周|年)(以来|以内|内|里|来|前|以前)?
# 2026-08-26 修复：加前/后向边界——"8月15号"的"8月"（后跟"15号"）不误识别为"8个月前"；
# "2025年"的"年"不被误匹配为相对年份（数字 2025 会进 num，period=年 → 0001 年 bug）。
# 负向前瞻：(?<![0-9号日])——num 前不能是数字/号/日（排除 "8月15号" 里 15 前面的 8？不，是排除
# "2025年" 的 5 前面有数字 → 2025 整体不匹配）；(?![号日])——period 后不能跟 号/日。
# 2026-08-26 漏洞1 修复：具体月份"3月"（后无"个"且词尾）≠ 相对月数"3个月"。
# 规则：period=月/年 时，若**无"个"字且后无 以来/内/里/前/以前** → 视为具体月份/年份（如"3月"），
# 不匹配相对模式（具体年月由 _rule_extract 的 _parse_specific_date 处理）。
_REL_TIME_RE = re.compile(
    rf"(?<![0-9号日])(?P<pre>近|最近)?(?P<num>{_NUM_RE})(?P<unit>个)?(?P<period>天|月|星期|周|年)"
    rf"(?P<suf>以来|以内|内|里|来|前|以前)?(?![0-9号日])"
)


# 具体日期模式（2026-08-26 漏洞2 修复）：2025年3月 / 8月15号 / 8月15日 / 2025年
# 2026-08-26 边界：具体月份"3月"后不能跟 前/以来/内/里/来（那是相对时间"3个月前"），
# 用 (?!前|以来|以内|内|里|来|以前) 排除。
_SPECIFIC_DATE_RE = re.compile(
    r"(?P<year>\d{4})年(?:(?P<month>\d{1,2})月)?(?:(?P<day>\d{1,2})[号日])?|"
    r"(?P<month2>\d{1,2})月(?:(?P<day2>\d{1,2})[号日])?(?!前|以来|以内|内|里|来|以前)"
)


def _parse_specific_date(query: str) -> dict | None:
    """具体年月日 → {start, end}（2025年3月 / 8月15号 / 2025年）。解析失败返回 None。"""
    m = _SPECIFIC_DATE_RE.search(query)
    if not m:
        return None
    t = _today()
    g = m.groupdict()
    year = int(g["year"]) if g.get("year") else t.year
    month = int(g["month"] or g["month2"] or 0)
    day = int(g["day"] or g["day2"] or 0)

    def iso(d):
        return d.strftime("%Y-%m-%d")

    # 只有年：2025年 → 整年
    if not month:
        if not (1900 <= year <= 2100):
            return None
        return {"start": iso(dt.date(year, 1, 1)), "end": iso(dt.date(year, 12, 31))}
    # 年月：2025年3月 / 3月 → 整月
    if not day:
        if not (1 <= month <= 12):
            return None
        import calendar as _cal

        last = _cal.monthrange(year, month)[1]
        return {"start": iso(dt.date(year, month, 1)), "end": iso(dt.date(year, month, last))}
    # 年月日：8月15号 → 单日
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    try:
        return {"start": iso(dt.date(year, month, day)), "end": iso(dt.date(year, month, day))}
    except ValueError:
        return None


def _parse_time_expr(expr: str) -> dict | None:
    """时间表达式 → {start: YYYY-MM-DD, end: YYYY-MM-DD}（闭区间）。解析失败返回 None。"""
    if not expr:
        return None
    e = expr.strip().lower()
    t = _today()
    monday = t - dt.timedelta(days=t.weekday())  # 本周一

    def iso(d):
        return d.strftime("%Y-%m-%d")

    if e == "今天":
        return {"start": iso(t), "end": iso(t)}
    if e == "昨天":
        d = t - dt.timedelta(days=1)
        return {"start": iso(d), "end": iso(d)}
    if e == "前天":
        d = t - dt.timedelta(days=2)
        return {"start": iso(d), "end": iso(d)}
    if e == "前两天":  # P0-1：前天+昨天
        return {"start": iso(t - dt.timedelta(days=2)), "end": iso(t - dt.timedelta(days=1))}
    if e == "前几天":  # P0-1：近四天（不含今天）
        return {"start": iso(t - dt.timedelta(days=4)), "end": iso(t - dt.timedelta(days=1))}
    if e == "这几天":  # P0-1：近三天（含今天）
        return {"start": iso(t - dt.timedelta(days=3)), "end": iso(t)}
    if e == "最近":  # P0-1：近七天（"最近的文件"）
        return {"start": iso(t - dt.timedelta(days=6)), "end": iso(t)}
    if e == "本周":
        return {"start": iso(monday), "end": iso(t)}
    if e == "上周":
        return {"start": iso(monday - dt.timedelta(days=7)), "end": iso(monday - dt.timedelta(days=1))}
    if e == "上上周":  # P0-1：两周前的自然周
        return {"start": iso(monday - dt.timedelta(days=14)), "end": iso(monday - dt.timedelta(days=8))}
    if e == "本月":
        return {"start": iso(t.replace(day=1)), "end": iso(t)}
    if e == "上月":
        first_this = t.replace(day=1)
        last_last = first_this - dt.timedelta(days=1)
        return {"start": iso(last_last.replace(day=1)), "end": iso(last_last)}
    if e == "上上个月":  # P0-1：两个月前的自然月
        first_this = t.replace(day=1)
        two_before = first_this - dt.timedelta(days=1)  # 上月最后一天
        two_before = two_before.replace(day=1) - dt.timedelta(days=1)  # 上上月最后一天
        return {"start": iso(two_before.replace(day=1)), "end": iso(two_before)}
    if e == "今年":
        return {"start": iso(t.replace(month=1, day=1)), "end": iso(t)}
    if e == "去年":
        last_year = t.replace(year=t.year - 1)
        return {"start": iso(last_year.replace(month=1, day=1)), "end": iso(last_year.replace(month=12, day=31))}
    if e == "去年底":  # P0-1：去年 12 月
        last_year = t.replace(year=t.year - 1)
        return {"start": iso(last_year.replace(month=12, day=1)), "end": iso(last_year.replace(month=12, day=31))}
    if e == "今年上半年":  # P0-1：1 月 1 日 ~ 6 月 30 日
        return {"start": iso(t.replace(month=1, day=1)), "end": iso(t.replace(month=6, day=30))}
    if e == "最近一周":
        return {"start": iso(t - dt.timedelta(days=6)), "end": iso(t)}

    # D1 扩展：N(月|星期|周|年) 的 以来/内/里（区间），N天前 保留单日
    m = re.match(rf"(?:近|最近)?({_NUM_RE})(?:个)?(月|星期|周|年|天)(?:以来|以内|内|里|来)?$", e)
    if m:
        n = _num_val(m.group(1))
        period = m.group(2)
        if n is None or n <= 0:
            return None
        if n < 1:  # "半"：半个月/周/年 → 近似天数
            half_days = {"月": 15, "星期": 4, "周": 4, "年": 182}
            start = t - dt.timedelta(days=half_days.get(period, 15))
            return {"start": iso(start), "end": iso(t)}
        ni = int(n)
        if period == "月":
            start = _add_months(t, -ni)
        elif period == "年":
            start = _add_months(t, -12 * ni)
        elif period == "星期":
            start = t - dt.timedelta(days=7 * ni - 1)
        elif period == "周":
            start = t - dt.timedelta(days=7 * ni - 1)
        else:  # 天
            start = t - dt.timedelta(days=ni - 1)
        return {"start": iso(start), "end": iso(t)}

    # N天前（单日，保留原语义）
    m = re.match(rf"({_NUM_RE})天前$", e)
    if m:
        d = t - dt.timedelta(days=int(_num_val(m.group(1))))
        return {"start": iso(d), "end": iso(d)}
    # N周前（返回那一周的区间，保留原语义）
    m = re.match(rf"({_NUM_RE})周前$", e)
    if m:
        n = int(_num_val(m.group(1)))
        return {"start": iso(monday - dt.timedelta(weeks=n)), "end": iso(monday - dt.timedelta(weeks=n - 1, days=1))}
    # D1：N个月前 / N年前（单日）
    m = re.match(rf"({_NUM_RE})(?:个)?月前$", e)
    if m:
        d = _add_months(t, -int(_num_val(m.group(1))))
        return {"start": iso(d), "end": iso(d)}
    m = re.match(rf"({_NUM_RE})年前$", e)
    if m:
        d = _add_months(t, -12 * int(_num_val(m.group(1))))
        return {"start": iso(d), "end": iso(d)}
    return None


# ---------- 类型词（D3 边界匹配：短扩展名不误命中长英文词，如 "py" ∉ "python"） ----------

# 中文类型词：substring 匹配（"pdf文档" 等）
# 2026-08-26 漏洞3 修复：图片/视频 裸词 → 类型优先（fileTypes 多扩展名），
# 而非目录（用户说"图片"意图是找图片文件；Pictures 目录通常未索引会落空）
_TYPE_CN_MAP: list[tuple[str, str | list[str] | None]] = [
    ("pdf文档", "pdf"), ("pdf文件", "pdf"),
    ("word文档", "docx"), ("word文件", "docx"),
    ("md文件", "md"), ("markdown文档", "md"),
    ("markdown文件", "md"), ("文本", "txt"),
    ("代码", None),
    ("图片", ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"]),
    ("视频", ["mp4", "mkv", "avi", "mov", "webm", "flv"]),
    ("照片", ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"]),
    # P1-1 类型泛化（2026-08-26）：中文口语常用词 → 已知类型（KNOWN_TYPES 内，避免 1002）
    ("表格", "xlsx"), ("表格类", "xlsx"),
    ("幻灯片", "pptx"), ("演示文稿", "pptx"), ("演示", "pptx"),
    ("截图", ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"]),
    ("配置文件", "json"),  # 仅"配置文件"整体（"配置"单独会误伤"网络配置"等内容词）
]
# 英文/扩展名：词边界匹配（前后非 a-z0-9）；_EN_EXT 为别名 → 扩展名映射（_extract_types 与 purify 共用）
_EN_EXT = {
    "word": "docx", "excel": "xlsx", "ppt": "pptx", "python": "py", "markdown": "md",
    "javascript": "js", "typescript": "ts", "golang": "go", "rust": "rs",
    "c++": "cpp", "c#": "cs", "shell": "sh", "bash": "sh",
}

# 2026-10-08 两类英文类型词（消歧的核心，见 _TYPE_HEAD_LIMITERS 说明）：
#   格式词（_FORMAT_EN_WORDS）——说"文件长什么样"（扩展名/软件名）。
#       独立出现即当类型（"jpg"/"pdf"/"markdown" 就是找这类文件；"发票的jpg" 实测保留）。
#   语言名（_CODE_LANG_WORDS）——说"内容在讲什么"（编程语言/数据格式名）。
#       独立出现当**内容词**（"python"→搜内容；"java"/"c"/"cpp"/"rust" 同理，行为统一）；
#       后跟限定词（文件/脚本）才当类型（"python文件"→py）。
#   两者**后面还有别的内容词**时，一律当内容词（"excel公式"/"python源码剖析"/"jpg压缩"）。
_FORMAT_EN_WORDS = ("pdf", "docx", "word", "xlsx", "excel", "pptx", "ppt",
                    "md", "markdown", "txt", "png", "jpg", "jpeg")
_CODE_LANG_WORDS = ("python", "py", "javascript", "js", "typescript", "ts",
                    "java", "cpp", "c++", "c", "cs", "c#", "go", "golang",
                    "rust", "rs", "sql", "shell", "sh", "bash", "json")
# 检测白名单 = 两类之和（长词在前，避免 "c" 抢先匹配 "c++"/"cs"）
_TYPE_EN_WORDS = _FORMAT_EN_WORDS + _CODE_LANG_WORDS

# 纯目录词（2026-08-26 用户前端实测暴露）：DIR_MAP 中**不是**类型词（双义词）的目录词。
# 纯目录词已进 filters.directories，purify 时剔除（kw 残留污染 embed）；
# 双义词（图片/视频）保留（可能是内容词）。须定义在 _TYPE_CN_MAP 之后。
_PURE_DIR_WORDS = frozenset(
    w for w in DIR_MAP
    if w not in dict(_TYPE_CN_MAP) and w not in _WEAK_DIR_WORDS
)
_TYPE_EN_PAT = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(w) for w in sorted(_TYPE_EN_WORDS, key=len, reverse=True))
    + r")(?![a-z0-9])")


def _type_aliases(ext: str) -> list[str]:
    """扩展名 → 所有可剔除写法（扩展名本身 + 英文别名 + 中文词），purify 用。"""
    aliases = [ext]
    aliases += [w for w, e in _EN_EXT.items() if e == ext]
    aliases += [kw for kw, e in _TYPE_CN_MAP if (e == ext or (isinstance(e, list) and ext in e))]
    return aliases


# ---------- 类型词消歧：中心语判据（2026-10-08） ----------
# 背景（用户实测）：同一个句式对 20 个类型的待遇不一致——"X语法"里 X 在 _TYPE_EN_WORDS
# 就当类型筛、不在就当内容词，于是 "python源码剖析" / "excel公式" 这类查询大量返回 0 条
# （实测 20 条"类型词+内容词"型查询里 14 条为 0，而整句当内容词 20/20 都有结果）。
#
# 判据：类型词在句中可能承担两种角色，按中文语序（修饰语在前、中心语在后）区分——
#   中心语 head      "发票的jpg" / "python文件"  → 用户要的是**这类文件** → 进 filters.fileTypes
#   修饰语 modifier  "python源码剖析" / "excel公式" → 用户要的是**关于它的内容** → 留在 keywords
#
# 三条规则（自上而下，先命中先定）：
#   ① 类型词后紧跟文件性名词（_TYPE_HEAD_LIMITERS）→ 中心语
#   ② 类型词之后还有其它内容词或其它类型词      → 修饰语
#   ③ 其余（独立成词 / 位于句末）               → 中心语
#
# 这样判定与"词是否在英文白名单里"解耦：语言名（python/cpp/java…）与格式名（jpg/pdf…）
# 走同一套语序判据，不再有"写死的特例集合"。
_TYPE_HEAD_LIMITERS = (
    # 长词在前（"源码文件"含"文件"，短词先匹配会截断）
    "源码文件", "源代码文件", "文件", "文档", "脚本", "表格",
    "幻灯片", "演示文稿", "演示", "图片", "视频", "照片", "截图",
)
# 英文文件性名词（2026-10-09 修 1）：同一判据的英文侧。
#   原表只有中文，于是中文并列「pdf、md、txt 文档」收全三种，而同一句式的
#   英文「pdf, md, txt documents」只收 txt —— 判据不分语种，表却分。
#   ASCII 词用词边界匹配（"file" 不得命中 "filetype" / "profile"）。
_TYPE_HEAD_LIMITERS_EN = (
    "source files", "source file", "code files", "code file",
    "spreadsheets", "spreadsheet", "presentations", "presentation",
    "screenshots", "screenshot", "documents", "document",
    "scripts", "script", "slides", "slide", "images", "image",
    "videos", "video", "photos", "photo", "files", "file", "docs", "doc",
)
_TYPE_HEAD_LIMITERS_EN_PAT = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(w) for w in sorted(_TYPE_HEAD_LIMITERS_EN, key=len, reverse=True))
    + r")(?![a-z0-9])")
# 中文类型词表内部排序：长词优先 + docx 项置于 md 项之前。
# 2026-10-08：原表 "word文档"(docx) 排在 "markdown文档"(md) 之前只是巧合；
#   排序修正后 "文档markdown" 这类句尾类型词不会被 docx 项的长中文词抢占。
_TYPE_CN_ORDERED = sorted(
    _TYPE_CN_MAP,
    key=lambda kv: (0 if kv[1] == "docx" else 1, -len(kv[0])),
)

# 并列连接词（2026-10-09 补）。中文的限定词后缀只挂在**最后一个**并列项上：
#   「pdf、md、txt 文档」里只有 txt 后面跟着"文档"，但用户要的是三种文件。
#   故一个类型词若跨过"只有连接词"的间隙紧邻右边的中心语，它也是中心语。
# 长词在前（"以及"必须先于"及"被消费，否则剩"以"卡住）。
_TYPE_CONNECTORS = ("以及", "或者", "和", "或", "与", "及", "、", "，",
                    ",", "/", "\\", "or", "and")
_TYPE_CONNECTORS_SORTED = tuple(sorted(_TYPE_CONNECTORS, key=len, reverse=True))
# 连接词里的**字词**（不含标点）——净化收尾时按整词丢弃（修 4）。
#   不能像普通停用词那样做 substring 替换："及"会把「普及率」拆成「普 率」、
#   「涉及内存」拆成「涉 内存」；按整词丢弃则这两例都不受影响（它们不是独立词）。
_TYPE_CONNECTOR_WORDS = frozenset(
    w for w in _TYPE_CONNECTORS if re.fullmatch(r"[a-z\u4e00-\u9fa5]+", w)
)


def _only_connectors(gap: str) -> bool:
    """两个类型词之间是否**只有**连接词（无内容）。

    空白先整体去掉再判：英文连接词两侧有空格（"pdf and md files" 的间隙是
    " and "），只在串首消费连接词的话，开头的空格会让整个间隙判不出。

    须非空：相邻两个类型词（"pythong文件"之外的 "pdfmd"）不算并列——
    间隙为空说明它们本就是粘在一起的一个词，不该当并列传播；
    纯空白间隙（"pdf md"）同样不算。
    """
    s = re.sub(r"\s+", "", gap)
    if not s:
        return False
    changed = True
    while s and changed:
        changed = False
        for c in _TYPE_CONNECTORS_SORTED:
            if s.startswith(c):
                s = s[len(c):]
                changed = True
                break
    return not s


def _followed_by_limiter(q: str, start: int, end: int) -> bool:
    """类型词是否"后跟文件性名词"。三种形态都要认：

      1) 跨度之后紧跟（允许中间有空白）："txt 文档"（尾缀隔了空格）
      2) 跨度**自己就含**限定词：中文表里的 "word文件" / "pdf文档" 这类条目，
         跨度是 6 个字（word+文件），限定词落在跨度内部——
         只查"跨度之后"会漏，导致该词判不出中心语、并列传播也无从触发。
      3) 英文文件名词（2026-10-09 修 1）："md documents" / "word file"。
         原表只有中文 → 同一句式的英文并列收不全。ASCII 走词边界匹配。
    """
    if any(q[start:end].endswith(lim) for lim in _TYPE_HEAD_LIMITERS):
        return True
    j = end
    while j < len(q) and q[j].isspace():
        j += 1
    if any(q.startswith(lim, j) for lim in _TYPE_HEAD_LIMITERS):
        return True
    return _TYPE_HEAD_LIMITERS_EN_PAT.match(q, j) is not None


def _type_matches(query: str) -> list[tuple[int, int, tuple[str, ...], bool]]:
    """全部类型词命中的 (start, end, exts, is_code_lang)，按起点排序。

    exts 为**元组**：中文词可能一个词映射多个扩展名（"图片"→png/jpg/jpeg/…），
    不能按单扩展名聚合。
    is_code_lang=True 表示命中的是**编程语言/数据格式名**（`_CODE_LANG_WORDS`：
    python/js/c/cpp/java/rust/sql…）——这类词独立出现时当内容词（见 `_extract_types`）。
    中文类型词（配置文件/表格/幻灯片…）一律 False（它们是格式词）。

    中文 substring；英文/扩展名保持词边界（`c` ∉ `cells`）。同一起点上若命中多个
    长度不同的词（如"文档markdown"里 `markdown` 与长中文词），取最长者。
    """
    q = query.lower()
    hits: list[tuple[int, int, tuple[str, ...], bool]] = []
    for kw, ext in _TYPE_CN_ORDERED:
        if not ext:
            continue
        exts = tuple(ext) if isinstance(ext, list) else (ext,)
        pos = q.find(kw)
        while pos >= 0:
            hits.append((pos, pos + len(kw), exts, False))
            pos = q.find(kw, pos + 1)
    for m in _TYPE_EN_PAT.finditer(q):
        word = m.group(1)
        # 词边界用 lookbehind/lookahead（不占字符）→ m.start(1)/m.end(1) 就是词自身的
        # 跨度。（旧写法把边界放进捕获组，m.end() 会多含一个边界字符，"类型词之后
        # 是否还有内容"永远判真；用 group(2) 能绕开，但并列连接词的 gap 切分仍会错。）
        hits.append((m.start(1), m.end(1), (_EN_EXT.get(word, word),),
                     word in _CODE_LANG_WORDS))
    # 同一 (start,end) 去重 → 同起点只留最长的那个词
    dedup: dict[tuple[int, int], tuple[tuple[str, ...], bool]] = {}
    for s, e, exts, is_lang in hits:
        dedup.setdefault((s, e), (exts, is_lang))
    best: dict[int, tuple[int, int, tuple[str, ...], bool]] = {}
    for (s, e), (exts, is_lang) in dedup.items():
        cur = best.get(s)
        if cur is None or e - s > cur[1] - cur[0]:
            best[s] = (s, e, exts, is_lang)
    return sorted(best.values(), key=lambda h: h[0])


def _extract_types(query: str) -> list[str]:
    """类型词 → 扩展名列表（仅收**中心语**型；修饰语型留在 keywords）。

    2026-08-26 漏洞3：_TYPE_CN_MAP 的 ext 支持 list（图片→多扩展名），展开后去重。
    2026-08-26 漏洞3b："X里的Y"结构中 X 是位置限定词（图片里的视频 → 图片=位置），
    不作为类型词——剔除位置限定词后再做类型匹配。
    2026-10-08：加中心语判据（见 _TYPE_HEAD_LIMITERS 上方说明）。
    """
    # 剔除"X里的Y"/"X上的Y"结构中的 X（位置限定词，不参与类型匹配）——在**原串**上
    # 替换后再小写，保证命中位置与 residual 判定用同一坐标系。
    for pat in (r"([\u4e00-\u9fa5]{2,3})里(?:的)?", r"([\u4e00-\u9fa5]{2,3})上(?:的)?"):
        m = re.search(pat, query)
        if m and m.group(1) in DIR_MAP:
            query = query.replace(m.group(1), " ", 1)
            break
    q = query.lower()
    hits = _type_matches(q)
    if not hits:
        return []
    # 残留内容词（把全部候选类型词一并剔除后的剩余文本）——用于判"类型词之后还有内容吗"
    all_exts: list[str] = []
    for _s, _e, exts, _lang in hits:
        for ext in exts:
            if ext not in all_exts:
                all_exts.append(ext)
    residual_positions = []
    for tok in _purify_keywords(query, "", {"fileTypes": all_exts}).split():
        pos = q.find(tok)
        if pos >= 0:
            residual_positions.append(pos)

    types: list[str] = []
    is_head: list[bool] = []
    right_lims: list[int] = []   # 全部"紧跟限定词"的类型词下标（可能多个）
    for i, (start, end, exts, is_lang) in enumerate(hits):
        # ① 后跟文件性名词（"python文件"/"excel表格"/"txt 文档"）→ 一定是中心语
        #    附加："类" 后缀（"表格类/图片类的数据"）同样表类型中心语
        #    注意限定词可能落在**跨度内部**（中文表条目 "word文件"），见 _followed_by_limiter
        head = (_followed_by_limiter(q, start, end)
                or q.startswith("类", end) or q[start:end].endswith("类"))
        is_head.append(head)
        if head:
            right_lims.append(i)

    # ①b 并列传播：限定词后缀只挂在每个并列段的最后一个类型词上——
    #     「pdf、md、txt 文档」里只有 txt 后跟"文档"，但用户要的是三种；
    #     「pdf、md、txt 文档和表格」由 right_lims 逐个右端出发（txt→pdf、表格→…），
    #     保证中转类型词也在一侧被传播到。
    #     逐段向左：若两个类型词之间**只有连接词**（且非空）则同段，否则断段。
    #     "excel和word文件" 收 docx+xlsx（无内容词阻隔）；
    #     "Python源码剖析和另一个word文件" 只收 docx（间隙含内容词）。
    for ridx in right_lims:
        for i in range(ridx - 1, -1, -1):
            if not _only_connectors(q[hits[i][1]:hits[i + 1][0]]):
                break
            is_head[i] = True

    for i, (start, end, exts, is_lang) in enumerate(hits):
        if is_head[i]:
            pass
        elif is_lang:
            # ② 语言名/数据格式名（python/c/cpp/java/rust/js/sql…）：
            #    默认当**内容词**（"python" 找讲 python 的资料）——与格式词分流，
            #    这是"分不清 keyword 还是 filter"的正解：语言名说的是内容主题。
            pass
        else:
            # ③ 格式词（配置文件/jpg/pdf/docx/excel/markdown…）：
            #    后面还有别的词（内容词或类型词）→ 当修饰语；
            #    否则（独立成词/句末）→ 中心语（"发票的jpg" 只要 jpg 文件）。
            after_type = any(s2 > start for s2, _e2, _x2, _l2 in hits)
            after_content = any(pos >= end for pos in residual_positions)
            is_head[i] = not (after_type or after_content)
        if is_head[i]:
            for ext in exts:
                if ext not in types:
                    types.append(ext)
    return types


# ---------- D3：关键词净化 ----------

_PURIFY_STOPWORDS = (
    # 语气/疑问/连接词
    "我们", "帮我", "请", "一下", "哪些", "什么", "所有", "一些", "相关", "关于",
    "给我", "查找", "搜索", "看看", "查一下", "请问", "想要", "需要",
    # 动作/祈使词（2026-08-26：剔除"找出/找到/找"等——非内容词，如"给我找出上周修改过的markdown" → kw=md）
    "找出", "找到", "查查", "找找", "找", "整理",
    "的", "了", "过", "吗", "呢", "啊", "吧", "与", "和", "以及", "或者", "中",
    # 2026-08-26 catalog 暴露："有哪些pdf"净化后残留"有"（"有哪些"整体未在表）
    # 2026-09-04：顺序修复——"有没有"必须在"有"之前整体删除，
    #   否则先删"有"会把"有没有"拆成"没 没有"（实测"有没有GPU相关的笔记"→kw='没 gpu'）
    "有没有", "是不是", "是否", "有",
    # 泛化名词（不是内容词）
    "文档", "文件", "资料", "内容", "目录", "文件夹", "东西",
    # 2026-09-01：方位触发词"本地"（本地搜索/本地查找 → 触发词非内容词，
    # 残留进 query 会污染 embed 语义——"本地"维度拉高含"本地化/本地变量"的文件）
    "本地", "本机", "电脑",
    # 2026-09-04：口语残留（矩阵实测）——"我/在哪/多少/多少个"（"我的GPU调试笔记在哪"
    # →kw='我 gpu调试在哪'；"本地有多少个markdown文件"→kw='多少个'）。
    # 注意："我" substring 会拆"我们/我的"，但"我们"在上面已先删、无碍
    "我", "在哪", "多少个", "多少",
    # LLM 可能的英文翻译（最小集：高频残留词；prompt 已收紧"不译英文"）
    "document", "file", "related", "about", "the", "of", "for", "with",
    "modification", "modified", "update", "created", "accessed",
    "month", "week", "year", "day", "one", "two", "three", "ago", "recent",
    # 2026-09-04：英文 query 功能词/方位词/检索动词（整词边界删除，中文不受影响）——
    #   "search local files about GPU texture"→gpu texture；"find GTK documents on this
    #   machine"→gtk；复数文件名（files/docs/documents/pdfs/notes…）一并剔除
    "search", "find", "look", "browse", "local", "machine", "computer",
    "device", "this", "these", "those", "that", "my", "on", "in",
    "files", "docs", "documents", "pdfs", "notes", "videos", "images", "photos",
)


def _purify_keywords(query: str, raw_kw: str, rules: dict) -> str:
    """关键词净化（D3）：剔除已结构化提取的约束词 + 语气/停用词。

    raw_kw 为空/等于原始 query 时用 query 本身净化；LLM 返回英文翻译时也能清洗。
    净化后为空 → 回退原始 query（保证 embed 有内容）。
    """
    text = ((raw_kw or "").strip() or query).lower()
    t = text
    # 1. 已提取类型词：别名（扩展名/英文别名/中文词）统一词边界剔除，单循环
    #    2026-08-26：中文词用 replace（"GPU视频"里"视频"前是字母，词边界正则失效）；
    #    英文/扩展名保持词边界（防 "py" ∉ "python"）
    for ext in rules.get("fileTypes") or []:
        for alias in _type_aliases(ext):
            if re.fullmatch(r"[\u4e00-\u9fa5]+", alias):
                t = t.replace(alias, " ")
            else:
                t = re.sub(rf"(^|[^a-z0-9]){re.escape(alias)}($|[^a-z0-9])", r"\1\2", t)
    # 2. 时间表达式原文
    raw = rules.get("_time_expr_raw")
    if raw:
        t = t.replace(raw, " ")
    # 3. 目录词：**不剔除双义词**（"视频/图片" 等既是目录词也是内容词，剔除会丢语义；
    #    检索时 filters.directories 已限定目录，keywords 残留无害）
    #    但 2026-08-26 漏洞5：目录词+方位词粘连（"下载里/桌面上/图片里"）是纯方位残留
    #    （不是内容词），需剔除——如"下载里最近一周的周报" → 去掉"下载里"保留"周报"
    for dir_word in ("下载里", "桌面上", "图片里", "视频里", "音乐里", "文档里", "文件夹里", "图片中", "视频中"):
        t = re.sub(rf"(^|[^a-z0-9]){re.escape(dir_word)}($|[^a-z0-9])", r"\1\2", t)
    # 2026-08-26 漏洞5b：方位词单独残留（"图片里的视频"剔除"图片里"后剩"里"）——剔除孤立方位词
    for pos_word in ("里面", "里", "中", "上"):
        t = re.sub(rf"(^|[^a-z0-9]){re.escape(pos_word)}($|[^a-z0-9])", r"\1\2", t)
    # 3c. 纯目录词剔除（2026-08-26 用户前端实测暴露）："桌面/下载/音乐"等非双义词
    #     （不在 _TYPE_CN_MAP 类型映射）已进 directories filter，kw 残留会污染 embed 语义；
    #     双义词（图片/视频）保留（可能是内容词）。
    for dir_word in _PURE_DIR_WORDS:
        t = re.sub(rf"(^|[^a-z0-9]){re.escape(dir_word)}($|[^a-z0-9])", r"\1\2", t)
    # 3d. 泛化目录词方位剔除（P0-2，2026-08-26）："项目里/工作里/代码里/资料里"等
    #     ——泛化目录词不映射具体路径（用户未配置对应目录，全库检索降级），
    #     但方位粘连需剔除（kw 残留污染 embed）。"学习资料/工作文档"整体剔除。
    for gen_word in ("项目里", "项目中", "项目目录", "工作里", "工作中", "工作目录",
                     "代码里", "代码中", "代码目录", "资料里", "资料中", "资料目录",
                     "学习资料", "工作文档", "项目文档", "笔记里", "笔记中",
                     # 泛化目录词裸词（剔方位后残留）：项目/工作/笔记 是位置词非内容词
                     "项目", "工作", "笔记", "资料"):
        t = re.sub(rf"(^|[^a-z0-9]){re.escape(gen_word)}($|[^a-z0-9])", r"\1\2", t)
    # 3e. 剔除后残留的方位/语气碎片（P0-1/P1-1 实测暴露）：
    #     "上次"的"上"被方位剔除后剩"次"；"工作目录下"的"下"；"表格类"的"类"；
    #     "刚刚/刚才"语气词；"刚看过的"的"刚"单字
    #     2026-09-01：碎片后紧跟英文（"一下GTK"→"下gtk"）时词边界正则失效，
    #     补一版中文边界（左侧非中文字符或串首 + 碎片 + 任意）——"下"直接粘连英文也剔除
    #     2026-09-01 修：先把"一下"整体剔除（否则碎片循环会把"一"当碎片删掉，
    #     留下"下gtk"——"一下"整体在停用词表也匹配不到了）。
    #     注意：不能用词边界正则（"一下gtk"里"下"后是字母 g，$|[^a-z0-9] 不匹配），
    #     用 replace 直接删。
    t = t.replace("一下", " ")
    for frag in ("次", "下", "类", "刚刚", "刚才", "刚", "一"):
        t = re.sub(rf"(^|[^a-z0-9]){re.escape(frag)}($|[^a-z0-9])", r"\1\2", t)
        # 串首 + 碎片 + 英文（"下gtk"）；或 非中文 + 碎片 + 英文（"一下GTK"→"下gtk"）
        t = re.sub(rf"^{re.escape(frag)}(?=[a-z0-9])", " ", t)
        t = re.sub(rf"([^一-龥]){re.escape(frag)}(?=[a-z0-9])", r"\1 ", t)
    # 4. 动作词 + 语气/停用词（"给我找出上周修改过的markdown" → 删除"给我/找出"后剩"上周修改  markdown"，
    #    其中"上周修改"是动作+时间残留——时间已结构化到 filters，动作词"修改"也在 _ACTION_WORDS，
    #    但"上周修改"作为一个整体未被单独命中 → 需在停用词替换后再做一轮词边界/整体清理）
    for w in _ACTION_WORDS:
        t = t.replace(w, " ")
    # 4. 停用词：中文词 substring 替换（无词边界概念，保持原行为）；
    #    英文/ASCII 词整体词边界剔除（2026-09-04 英文支持：原统一 substring replace
    #    会把英文内容词拦腰切断——theme→me、proof→pro、today→to、weather/before/
    #    theory 等含 the/of/for 子串的词全部被静默腐蚀，英文关键词查询命中率受损。
    #    英文停用词本意只删整词，改词边界匹配后仅在独立成词时删除，中文不受影响）
    for w in _PURIFY_STOPWORDS:
        if re.fullmatch(r"[a-z0-9_]+", w):
            t = re.sub(rf"(^|[^a-z0-9]){re.escape(w)}($|[^a-z0-9])", r"\1\2", t)
        else:
            t = t.replace(w, " ")
    # 4b. 时间表达式（raw 已在步骤 2 处理；此处兜底"上周修改/昨天创建"等动作词+时间粘连残留）
    #     规则：把"时间词+动作词"粘连整体替换为空格（如"上周修改"→" "）
    for tw in ("今天", "昨天", "前天", "上周", "本周", "这周", "本月", "上月", "今年", "去年", "最近"):
        for aw in ("修改", "创建", "更新", "编辑", "改动", "变更", "保存"):
            t = t.replace(tw + aw, " ")
    # 5. 清理标点/空白
    t = re.sub(r"[,，、;；:：/\\|（）()]+", " ", t)
    words = [w for w in t.split()
             # 4c. 并列连接词按**整词**丢弃（修 4，2026-10-09）：类型词净化后留下的
             #     "或/及/or/and/和…"会作为残余关键词把纯筛选查询拖回语义检索
             #     （"pdf或md文件" kw='或' → 召回从 3 掉到 1）。这里只丢独立成词的
             #     连接词，不动 substring（"普及率"、"涉及内存" 不受影响）。
             if w and not re.fullmatch(r"[\W_]+", w) and w not in _TYPE_CONNECTOR_WORDS]
    if not words:
        # 2026-08-26：净化后为空 → 返回空串而非原始 query。
        # 原因：结构化条件已进 filters（如"给我找出上周修改过的markdown"→ fileTypes=[md]+dateRange），
        # 关键词空 = 纯筛选检索，语义正确；回退整句会让"找出/给我"等祈使词污染 embed 检索。
        # 检索层对空 query 安全：embed 空串 + 筛选仍能命中（embed_one("") 有向量但无意义词，
        # 实际靠 filters 约束；BM25 空 query 返回空）。若需纯筛选路径可后续加显式分支。
        return ""
    return " ".join(words)


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中容错提取 JSON 对象（容忍 markdown 围栏/前后缀文字）。"""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None


def _norm_types(types_raw) -> list[str]:
    """规范化 fileTypes（小写、去空白、过滤非法）。"""
    if not isinstance(types_raw, list):
        return []
    known = {"txt", "md", "pdf", "docx", "py", "js", "java", "cpp", "c", "cs", "go",
             "rs", "ts", "sql", "sh", "json", "png", "jpg", "jpeg", "gif", "webp",
             "bmp", "svg", "xlsx", "pptx"}
    out = []
    for t in types_raw:
        t = str(t).strip().lower()
        if t in known and t not in out:
            out.append(t)
    return out


def _rule_extract(query: str, index_dirs: list[str] | None = None,
                  _dirs: list[str] | None = None) -> dict:
    """规则层提取：时间表达式 / 目录词 / 类型词 / dateField（确定性，单一事实源）。

    返回 {fileTypes, dateRange, dirs, dateField?, _time_expr_raw?}（可能为空）。
    _time_expr_raw 为内部字段（purify 用），不进 filters。_dirs 为调用方已算好的
    目录命中（parse 路径复用，避免同一 query 重复提取）。
    """
    out: dict = {}
    # 时间表达式：精确词表（含"这"系列）→ 相对模式（N月/周/年 以来/内/里/前 + 中文数字）
    # 2026-08-26 漏洞1/2 修复：先试具体年月日（2025年3月 / 8月15号 / 3月），
    # 再试精确词表（今天/上周…），最后相对模式（3个月/最近3天…）
    found = None
    q_low = query.lower()
    # 1. 具体年月日（"3月"=具体月份，"3个月"=相对；"2025年3月"=具体年月）
    specific = _parse_specific_date(query)
    if specific:
        out["dateRange"] = specific
        out["_time_expr_raw"] = _SPECIFIC_DATE_RE.search(query).group(0)
        out["dateField"] = _date_field_of(query)
        found = "specific"
    # 2. 精确词表（今天/昨天/本周/上月…）——仅当非具体日期时（避免"上个月"被具体日期吞）
    #    P0-1：长词优先（"上上个月"含"上个月"、"前两天"含"前天"、"今年上半年"含"今年"）
    #    "最近"不在此循环：单独"最近"是近7天，但"最近3天"走相对模式（_REL_TIME_RE），
    #    若"最近"在精确词表会先匹配导致"最近3天"被截成"最近"→ 需在相对模式后兜底
    #    2026-09-04：_time_expr_raw 必须存**原文**（命中的 exact 词），不能存规范化映射值
    #    （"上个月"→映射"上月"，purify 用 raw 删原文时删不净，残留"个月"碎片）
    if found is None:
        raw_text = None
        for exact in ("这个月", "上上个月", "上个月", "上上周", "前两天", "前几天", "这几天",
                      "今年上半年", "上半年", "去年底",
                      "今天", "昨天", "前天", "本周", "这周",
                      "这个星期", "这星期", "上周", "本月", "这月", "上月", "今年", "本年",
                      "这年", "去年", "上一年"):
            if exact in query:
                found = _EXACT_TIME_EXPRS[exact]
                raw_text = exact
                break
    # 3. 相对模式（"3个月/最近3天/两周前"等）——仅当非具体日期且非精确词表时
    #    漏洞1：排除"3月"（具体月份，无"个"且词尾）——_parse_specific_date 已处理；
    #    这里 _REL_TIME_RE 仍会匹配"3月"（num=3, period=月, 无unit），需在循环内跳过
    if found is None:
        for m in _REL_TIME_RE.finditer(q_low):
            num, period, suf = m.group("num"), m.group("period"), (m.group("suf") or "")
            unit = m.group("unit") or ""
            # 漏洞1：period=月/年 且无"个"且无后缀（以来/前/内…）→ 具体月份/年份（已由 specific 处理），跳过
            if period in ("月", "年") and not unit and not suf:
                continue
            found = f"{m.group('pre') or ''}{num}{unit}{period}{suf}"
            raw_text = found
            break
    # P0-1/P1-2：模糊近义词兜底（相对模式未匹配到"最近N"才生效）
    #   "最近的文件"→近7天；"上次浏览过的文档"→近7天+accessed（dateField 由动作词定）；
    #   "刚刚创建的"→近7天+created。注意不匹配"最近3天"（已被 _REL_TIME_RE 处理）。
    #   2026-09-04：raw 存原文（"上次/刚刚"若统一存"最近"，purify 删不掉原文词残留）
    if found is None:
        for vague_word in ("最近", "上次", "刚刚", "刚才"):
            if vague_word in query:
                found = "最近"  # 统一用近7天语义
                raw_text = vague_word
                break
    if found and found != "specific":
        dr = _parse_time_expr(found)
        if dr:
            out["dateRange"] = dr
            out["_time_expr_raw"] = raw_text or found
            out["dateField"] = _date_field_of(query)
    # 目录词（强词直配；"文档"要求位置语境；仅保留 index_dirs 内的）
    dirs = _dir_word_hits(query, index_dirs=index_dirs) if _dirs is None else _dirs
    if dirs:
        out["dirs"] = dirs
    # 类型词（中文词 substring + 英文/扩展名词边界）
    types = _extract_types(query)
    if types:
        out["fileTypes"] = types
    # 2026-08-26 漏洞3：类型优先——"图片/视频"裸词是找文件（类型），
    # 目录词（Pictures/Videos）让位（用户意图是图片文件，不是逛目录）。
    # 但"X里的Y"前置目录（下载里的视频 → dirs=Downloads）保留（那是位置限定），
    # 此时 Y 是类型词但"图片"是位置限定词，不触发类型优先移除。
    if out.get("dirs") and out.get("fileTypes"):
        # 判断是否有"X里的Y"结构（前置目录 + 后置内容词）——有则不触发类型优先
        has_loc_pattern = any(
            re.search(rf"({re.escape(k)})里(?:的)?|({re.escape(k)})上(?:的)?", query)
            for k in DIR_MAP
        )
        if not has_loc_pattern:
            # 裸词场景：移除与类型词同名的目录（图片→Pictures，视频→Videos）
            type_words = [kw for kw, e in _TYPE_CN_MAP if e and kw in query]
            remove_dirs = set()
            for tw in type_words:
                if tw in DIR_MAP:
                    remove_dirs.add(DIR_MAP[tw])
            if remove_dirs:
                out["dirs"] = [d for d in out["dirs"] if d not in remove_dirs]
                if not out["dirs"]:
                    del out["dirs"]
    return out


def _collect_unsupported_dirs(query: str, index_dirs: list[str] | None) -> tuple[list[str], list[str]]:
    """目录命中收集 → (unsupported_dirs, dirs_in)。

    unsupported_dirs：query 中出现的目录词但不在 index_dirs（调用方提示"暂未索引"）；
    dirs_in：在 index_dirs 内的目录（进 filters）。index_dirs=None 时两者皆空。

    2026-08-26 漏洞3c：类型优先消歧——query 中同时是类型词+目录词的（图片/视频/照片），
    类型优先（用户意图是找文件），不收集为 unsupported 目录（避免"图片"被 Pictures
    未索引短路，类型过滤落空）。
    """
    if index_dirs is None:
        return [], []
    # 类型词命中 → 目录让位：图片/视频/照片 等双义词不作为目录收集
    type_words = [kw for kw, e in _TYPE_CN_MAP if e and kw in query]
    type_dir_words = {tw for tw in type_words if tw in DIR_MAP}  # 图片/视频（在 DIR_MAP 里的类型词）
    raw_dirs = _dir_word_hits(query, index_dirs=None)
    raw_dirs = [d for d in raw_dirs if not any(DIR_MAP.get(tw) == d for tw in type_dir_words)]
    unsupported = [p for p in raw_dirs if p not in index_dirs]
    dirs_in = _dir_word_hits(query, index_dirs=index_dirs)
    dirs_in = [d for d in dirs_in if not any(DIR_MAP.get(tw) == d for tw in type_dir_words)]
    return unsupported, dirs_in


def _build_filters(rules: dict) -> dict:
    """规则层结果 → filters（fileTypes / directories / dateRange[+dateField]）。"""
    filters: dict = {}
    if rules.get("fileTypes"):
        filters["fileTypes"] = rules["fileTypes"]
    if rules.get("dirs"):
        filters["directories"] = rules["dirs"]
    if rules.get("dateRange"):
        dr = dict(rules["dateRange"])
        if rules.get("dateField"):
            dr["dateField"] = rules["dateField"]
        filters["dateRange"] = dr
    return filters


@dataclass
class ParseResult:
    """确定性 NL 解析结果（生产路径）。

    is_structured：filters 非空 或 unsupported_dirs 非空 → 查询含结构化意图
    （时间/目录/类型/未索引目录）；False = 纯语义查询（等价旧 route='plain'）。
    """

    query: str
    filters: dict
    unsupported_dirs: list[str]

    @property
    def is_structured(self) -> bool:
        return bool(self.filters) or bool(self.unsupported_dirs)


def parse_nl_query(query: str, index_dirs: list[str] | None = None) -> ParseResult:
    """确定性解析（生产入口，/api/search 的 nl/rule 路径；3ms 级）。

    架构：结构化条件以规则层为准（确定性，避免 LLM 幻觉——实测 LLM 会漏时间条件、
    英文翻译污染 keywords）；keywords 由 purify 确定性提取。全未索引目录短路返回
    （无需关键词，调用方直接提示"暂未索引"）；部分未索引保留有效条件 + unsupported 附提示。
    只保留规则确定性解析。
    """
    unsupported, dirs_in = _collect_unsupported_dirs(query, index_dirs)
    # 快速路径：目录词**全部未索引** → 无需关键词（必然"暂未索引"提示）
    # 2026-08-26 用户实测：query 字段显示净化后 kw（原始整句污染 debug 展示——
    # 任务角度 kw 应为空，未索引提示由调用方/前端负责）
    if not dirs_in and unsupported:
        return ParseResult(_purify_keywords(query, "", {}), {}, unsupported)
    rules = _rule_extract(query, index_dirs=index_dirs, _dirs=dirs_in or None)
    return ParseResult(
        query=_purify_keywords(query, "", rules),
        filters=_build_filters(rules),
        unsupported_dirs=unsupported,
    )

