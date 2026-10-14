"""运行时配置层（2026-08-28 改为 config.yaml 单一来源）。

设计（独立 deb 包 + per-user + 主程序 HTTP 读写 + 测试直接改配置文件）：
- 配置唯一来源是 config.yaml（出厂默认复制自 /opt/bibit-filesearch → 用户 ~/.config/bibit-filesearch/config.yaml）
- 服务启动读 yaml → 运行时内存；PUT /api/settings 写回 yaml（加锁）→ 热生效
- 无 SQLite settings 表（单一配置源，无双写/优先级陷阱）
- 独立测试：直接编辑 config.yaml → 重启服务生效（永远生效，无 DB 覆盖）

配置 key（与前端设置页对齐）：
- index_dirs        知识库路径列表（默认动态 ~/Documents）
- auto_index        自动索引开关（watcher 增量监听；false = 关闭增量）
- indexable_types   可索引格式列表（默认全 20 类，含 png/jpg OCR）
- exclude_patterns  排除规则（glob 模式列表；默认 [.git,node_modules,__pycache__]）
- max_depth         索引深度（1-based：1=仅 base 本身，N=base+下 N-1 层子目录；
-                    默认 3；0=无限深，不允许负数）
- top_k             检索返回条数（默认 5）
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("bibit_filesearch.settings")

# 可索引格式全量集合（与 parser 对齐；供设置页勾选）
# 2026-09-07：png/jpg 加入（OCR 支持，图片文字可检索；rev 5 起引擎为 PP-OCR）
ALL_INDEXABLE_TYPES = [
    "txt", "md", "docx", "pdf",
    "py", "js", "java", "cpp", "c", "cs", "go", "rs", "ts", "sql", "sh", "json",
    "xlsx", "pptx",
    "png", "jpg",
]

# 出厂默认勾选集合（2026-09-09：png/jpg 纳入默认——图片文字开箱可搜）
DEFAULT_INDEXABLE_TYPES = list(ALL_INDEXABLE_TYPES)

# 出厂默认（代码即权威）。2026-09-09 起用户配置文件只保存「用户显式改过的
# 键」，其余一律回落到这里——出厂默认改进才能触达老用户。
# max_depth 语义（2026-09-04 统一，1-based）：0 = 无限深（显式 0 合法，全链路保留）；
# 正整数 N = base 目录 + 下 N-1 层子目录（1 = 仅 base 本身，不进入任何子目录；
# 2 = base+1 层子目录）；默认 3；负数不允许。
DEFAULTS: dict[str, Any] = {
    "index_dirs": [],
    "auto_index": True,
    "indexable_types": list(DEFAULT_INDEXABLE_TYPES),
    "exclude_patterns": [".git", "node_modules", "__pycache__"],
    "max_depth": 3,
    "top_k": 5,
}

# 配置版本（用户配置文件里的 config_version）。
# 语义：用户文件只存「与出厂默认不同的键」；本版本号用于一次性迁移旧文件
# （旧版首启会把整份出厂模板复制成用户配置，把当时的默认值冻结成用户选择）。
# v2 → v3：`models.embed.dim` 是唯一漏掉的老化石键——0.1.1~0.1.3 的出厂模板把它
# 冻成 512，而服务早已是 1024 维，导致老用户升级后每次检索都 1004 且界面无提示。
# 见 docs/troubleshooting.md 现象 G。
# v3 → v4：embed 后端由「直连推理进程 :8082」改为「经统一端点 :18790」——
# url 与 model 两个键的出厂默认同时变了，老文件里的旧值会盖住新默认、把请求继续
# 打到已退休的 :8082。见 _CHANGED_DEFAULTS 的两条新登记。
# v5 → v6：`ocr.max_side_len` 出厂默认 2000 → 1400（2026-09-28，降 OCR worker 峰值
# 39%）。老文件里冻着的 2000 会盖住新默认、把内存又拉回去 —— 登记为化石删掉。
# ★ 与 v2→v3 的 dim 同理：删配置**不会**让存量图片的文本跟着变，必须配
#   parser.PARSER_REVISION 6 的重排（reconcile_parser_revision → requeue_ocr_text）。
# v6 → v7：`top_k` 出厂默认 10 → 5（2026-10-14）。见 _CHANGED_DEFAULTS 的新登记。
CONFIG_VERSION = 7

# 默认值变更史：键路径 → 变更前的出厂默认值。
# 迁移时若用户文件里某键的值 == 这里的旧默认值，说明它只是首启复制进来的
# 化石（用户从未改过），删除让新默认生效；不等于旧值 = 用户真改过，保留。
_CHANGED_DEFAULTS: dict[str, Any] = {
    # v1 → v2：mean pooling 下指令前缀会淹没用短查询（haha.md 检索 0% 实测）
    "models.embed.query_instruction": "为这个句子生成表示以用于检索相关文章",
    # v2 → v3：dim 的老出厂默认值（bge-small-zh-v1.5 512 维）。Qwen3 换 1024 维后
    # 未登记，故老用户配置里的 512 一直存活并盖住新默认。
    # 注意：删掉配置键**不会**让存量索引跟着变，必须配 reconcile_embed_dim 一起用
    # （它按服务维度判定并把不可复用的旧向量清空重建）——只删键会让老用户从
    # 「静默搜不到」掉进「启动即报维度不符」。
    "models.embed.dim": 512,
    # v3 → v4：embed 后端改走统一端点。老出厂模板/用户文件里冻着直连地址与旧模型名，
    # 若不清理，升级后 filesearch 会继续请求**已不再监听的** :8082
    # → 表现为「服务在线但索引全失败」。
    # 只登记旧默认值：用户若自己指向过别处（自建服务 / 别的端口）不在列表内，
    # 会原样保留，不会被这次迁移误删。
    "models.embed.url": "http://127.0.0.1:8082",
    "models.embed.model": "qwen3-embedding-0.6b",
    # v4 → v5：embed 推理接口从早期草案的 /v1/embed/local/embeddings
    # 定为 /v1/embed/embeddings（上游 0.1.15 只有后者，前者返 404）。v4 那版出厂
    # 默认就是 local/embeddings，用户保存过设置就会把它冻在文件里 → 升级后一直 404，
    # 表现为「向量化服务未启动」。只登记这个旧默认值：用户自己填过的路径原样保留。
    "models.embed.path": "/v1/embed/local/embeddings",
    # 2026-09-09：png/jpg 纳入出厂默认。老版出厂模板（0.0.1–0.1.5）冻结的
    # 18 类列表 == 旧默认值 → 识别为化石删除，让新默认（20 类含图片）生效；
    # 用户自行增删过格式的列表不匹配此值，原样保留。
    "indexable_types": [
        "txt", "md", "docx", "pdf",
        "py", "js", "java", "cpp", "c", "cs", "go", "rs", "ts", "sql", "sh", "json",
        "xlsx", "pptx",
    ],
    # v5 → v6：OCR 输入长边上限 2000 → 1400（2026-09-28）。只登记旧默认值：用户自己
    # 调过别的数（内存紧的机器可能设过 1200）不匹配，原样保留。
    "ocr.max_side_len": 2000,
    # v6 → v7：`top_k` 出厂默认 10 → 5（2026-10-14）。老用户文件里冻着的 10 会盖住
    # 新默认，登记为化石删掉（改成别的数 = 用户真调过，保留）。
    # 与 index_dirs 不同：index_dirs 的动态默认只在文件里为空/缺失时才生效，本键不是。
    "top_k": 10,
}


def _factory_defaults() -> dict:
    """出厂默认树（代码权威）：config.DEFAULTS ⊕ settings.DEFAULTS。

    2026-09-09：出厂默认曾有三份（config.py DEFAULTS / settings.py DEFAULTS /
    packaging/config.yaml），彼此漂移——例如 cors_origins 只在模板里有值，
    DEFAULTS 是空列表，用户删配置后 GUI 直接跨域挂。现在代码是唯一权威，
    /opt 模板降级为「人类可读的参考」，并由 test_config_defaults_parity 守护一致。
    """
    from bibit_filesearch.config import DEFAULTS as CFG_DEFAULTS
    from bibit_filesearch.config import _deep_merge

    return _deep_merge(_deep_merge({}, CFG_DEFAULTS), DEFAULTS)


def _strip_fossils(data: dict, defaults: dict) -> tuple[dict, list[str]]:
    """删除「值 == 出厂默认」的键，返回 (瘦身后的树, 被删键路径)。

    旧版首启把整份出厂模板复制成用户配置，把当时的默认值冻结成用户选择——
    这类键与「用户显式改成了恰好等于默认的值」在文件里长得一模一样。
    删除前者是无损的（值仍由 DEFAULTS 提供，行为不变），且让后续默认值改进
    能触达该用户；代价是后者会被误判（用户若想钉住某值需另行约定，当前无此需求）。
    """
    out: dict = {}
    removed: list[str] = []
    for k, v in (data or {}).items():
        if isinstance(v, dict):
            sub_def = defaults.get(k) if isinstance(defaults.get(k), dict) else {}
            sub, sub_removed = _strip_fossils(v, sub_def)
            removed.extend(f"{k}.{r}" for r in sub_removed)
            if sub:
                out[k] = sub
        elif k in defaults and v == defaults[k]:
            removed.append(k)
        else:
            out[k] = v
    return out, removed


def reconcile_embed_dim(db, configured_dim: int, probed_dim: int | None) -> tuple[int, int]:
    """定出生效向量维度，并在存量索引不可复用时清空向量。返回 (生效维度, 重置文件数)。

    维度**以 embed 服务实际返回为准**（`probed_dim`），配置值只是提示：它可能是老版
    出厂模板冻下来的化石（512），也可能是用户为自己的 embed 服务显式声明的维度——
    两种情况都靠「探一次服务」区分，不靠猜配置。探不到（服务未起）时退回配置值。

    存量向量维度 ≠ 生效维度时旧向量一律不可复用，且**不能只删配置键**：那样会从
    「检索全部失败」变成「启动即报维度不符」。索引是派生数据，清空 + 重排队即可。
    """
    effective = int(probed_dim or configured_dim)
    stored = db.stored_vector_dim()
    purged = 0
    if stored and stored != effective:
        purged = db.purge_vectors()
        logger.warning(
            "存量索引维度 %d ≠ 当前 embed 维度 %d → 已清空向量并排队重索引（%d 个文件）。"
            "索引为派生数据，将从知识库目录重建。",
            stored, effective, purged,
        )
    return effective, purged


def reconcile_parser_revision(db, state, indexable_types, image_types=()) -> int:
    """解析能力升级后，把按旧能力误判的文件重新排一次队。返回重排的文件数。

    只在解析器「能读到以前读不到的内容」时才有事可做（见 `parser.PARSER_REVISION`）。
    版本未变 → 直接返回 0，不碰数据库，所以每次启动零代价。跑过一次就把版本记进
    `scan_state`，不会再重复（否则真·空文件每轮启动都要白解析一遍）。

    为什么要清 content_hash、为什么只收窄到索引格式内的类型：见
    `db.requeue_unsupported` 的说明。

    **三个影响面都要盖**（2026-09-29 补第三条）：
      · `requeue_unsupported` —— 当时解析不出内容、落了 unsupported 的文件；
      · `requeue_ocr_text`    —— 当时解析出了**旧版 OCR 文本**、落了 completed 的文件；
      · `requeue_all_pdfs`    —— rev 7 起 PDF 解析策略改变，全部 PDF 的文本都可能不同。
    只做前者的话，rev 2（OCR 方向转正）对已入库的图片毫无作用：它们不是 unsupported，
    而是「有文本但文本是乱码」，content_hash 又看不出 OCR 结果 → 永远 skipped。
    `image_types` 只对后者有意义（哪些类型属于 OCR 产物），默认空 = 跳过那一步。
    """
    from bibit_filesearch.parser import PARSER_REVISION

    seen = int(state.get("parser_revision", 0) or 0)
    if seen >= PARSER_REVISION:
        return 0
    # ★ 顺序：先跑「按 status 的 completed/queued/failed 全体」那几条，最后跑
    #   `requeue_unsupported`。两者按 status 天然互斥（unsupported 是唯一被排除的那个
    #   状态），但**每条都会把命中的行改写成 queued** —— 若 unsupported 那条先跑，它
    #   改出的 queued 会被后面的 requeue_all_* 再数一遍，计数虚高（实测：一个 unsupported
    #   的 docx 被数成 2）。所以 unsupported 必须放最后，才能让计数与「实际重排了几个
    #   文件」一致（这个数会打进启动日志）。
    requeued = 0
    if image_types:
        requeued += db.requeue_ocr_text(image_types)
    # rev 7（2026-09-29）：PDF 解析策略变了（两遍 + 逐页薄页判据 + 预算早停），
    # 影响面是**所有** PDF 而不只是图片型 —— 混合型 PDF 有文本层、pdf_images 为空，
    # 上面那条 requeue_ocr_text 一个都命中不到。见 db.requeue_all_pdfs。
    if 7 <= PARSER_REVISION:
        requeued += db.requeue_all_pdfs()
    # rev 9（2026-10-09）：docx 加了内嵌图 OCR，影响面是**所有** docx —— 纯图 docx
    # 从 unsupported 变有文本，正文 + 插图的 docx 文本变多。与 rev 7 对 PDF 同理，
    # 判据直写「所有 docx」，不只那些落了 unsupported 的。见 db.requeue_all_docx。
    if 9 <= PARSER_REVISION:
        requeued += db.requeue_all_docx()
    # 最后：当时解析不出内容、落了 unsupported 的文件（按 indexable_types 收窄）。
    requeued += db.requeue_unsupported(indexable_types)
    state.set("parser_revision", PARSER_REVISION)
    return requeued


def reconcile_chunk_revision(db, state, legacy_cap: int = 0, short_min_chars: int = 0) -> int:
    """分块规则变化后，把受影响的文件重新排一次队。返回重排的文件数。

    与 `reconcile_parser_revision` 同构，但管的是另一件事：解析**之后的**分块。
    典型触发是抬高 `scan.max_chunks_per_file`——被截断的文件因为 content_hash
    不变，重扫一律 `skipped`，配置改了却不生效（详见 `db.requeue_truncated`）。

    `legacy_cap` 服务于**老库**（`files.total_chunks` 全为 0、拿不到真实总块数），
    由调用方传 `indexer.CHUNK_REVISION_LEGACY_CAP`；语义与近似性见
    `db.requeue_truncated`。**同一个坑的第二层**：老库若只走精确判据，一行都不会被
    重排，而水位线会在本次就被置成已见——这次机会过了就再也没有了。

    `short_min_chars` 服务于 rev 2（极短块不向量化）：把「已有向量的极短块」所在
    文件重排一次，否则它们的旧向量永远活着（见 `db.requeue_short_chunks`）。

    rev 3（图片型 PDF 的 OCR 块被双写）不需要参数：判据是块文本的 indexer 侧独有前缀
    `[图N页]`，天然只命中重复块（见 `db.requeue_image_ocr_dupes`）。

    版本未变 → 直接返回 0、不碰数据库，每次启动零代价。跑过一次记进 `scan_state`，
    之后即使文件**仍然**处于截断态（真实块数 > 上限）也不会反复重灌——这是水位线
    存在的意义，不是遗漏。
    """
    from bibit_filesearch.indexer import CHUNK_REVISION

    seen = int(state.get("chunk_revision", 0) or 0)
    if seen >= CHUNK_REVISION:
        return 0
    requeued = db.requeue_truncated(legacy_cap=legacy_cap)
    if short_min_chars > 0:
        requeued += db.requeue_short_chunks(min_chars=short_min_chars)
    # rev 3：清掉 parser/indexer 双写留下的重复 OCR 块
    requeued += db.requeue_image_ocr_dupes()
    state.set("chunk_revision", CHUNK_REVISION)
    return requeued


# 设置 schema（update 类型校验，防前端传错类型）
_SETTING_TYPES = {
    "index_dirs": (list,),
    "auto_index": (bool,),
    "indexable_types": (list,),
    "exclude_patterns": (list,),
    "max_depth": (int,),
    "top_k": (int,),
}

_DEFAULT_CONFIG_PATHS = (
    Path("/opt/bibit-filesearch/config.yaml"),          # deb 出厂模板（人类可读参考）
    Path(__file__).resolve().parents[1] / "config.yaml",  # 源码开发默认（仓库内）
)


def default_config_path() -> Path:
    """出厂模板路径（人类可读参考；**不再是配置复制源**）。

    2026-09-09：模板不再被复制成用户配置，权威默认值在代码里（`_factory_defaults`）。
    本函数仅供文档/排障定位模板文件用。
    """
    for p in _DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p
    return _DEFAULT_CONFIG_PATHS[0]


def _prune_empty(data: dict) -> dict:
    """递归删除空 dict 值（化石清理后可能留下空父层，如 models.embed={}）。"""
    out: dict = {}
    for k, v in (data or {}).items():
        if isinstance(v, dict):
            sub = _prune_empty(v)
            if sub:
                out[k] = sub
        else:
            out[k] = v
    return out


def dedup_paths(dirs) -> list:
    """知识库路径去重（保留首次出现的顺序）。

    2026-09-20（用户报告）：index_dirs 允许出现相同路径（前端 textarea 按行收集、
    后端原先不去重），重复项会让 scanner 的计数与遍历按份数各算一遍/各走一遍——
    真实 2 个文件时 [A] 报 2、[A,A] 报 4、[A,A,A] 报 6，线性放大，并白白多扫磁盘。
    同一目录写多遍 = 同一个意图，故静默去重。

    判据是**完全相同**的路径字符串（规范化即 `~` 展开之后比较）。嵌套重叠
    （[/kb, /kb/sub]）是另一类判据（前缀包含），不在本函数处理范围内。
    """
    seen: set[str] = set()
    out: list = []
    for d in dirs or []:
        s = str(d)
        if s in seen:
            continue
        seen.add(s)
        out.append(d)
    return out


def _get_path(data: dict, path: str):
    """按点号路径取值，返回 (值, 是否存在)。"""
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def _del_path(data: dict, path: str) -> None:
    """按点号路径删除键（父层级不存在时静默）。"""
    parts = path.split(".")
    cur: Any = data
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _norm_indexable_types(v) -> list | None:
    """indexable_types 规范化：按 ALL_INDEXABLE_TYPES 固定顺序输出（集合语义）。

    2026-09-04：前端用 Set 收集勾选，Array.from 的顺序 = 用户点击/插入顺序——
    取消再勾选同一格式会让元素挪到末尾，payload 顺序漂移。此前 update/幂等判定
    按列表顺序比较，同内容不同序被误判为"变化"（时好时坏：txt 恰在末尾则不变）。
    配置语义是集合，比较与落盘都忽略顺序，统一按规范顺序写 yaml（diff 稳定）。
    """
    if v is None:
        return None
    s = set(v)
    known = [t for t in ALL_INDEXABLE_TYPES if t in s]
    unknown = sorted(s - set(ALL_INDEXABLE_TYPES))
    return known + unknown


class SettingsService:
    """config.yaml 单一来源配置读写（加锁，线程安全）。

    路径：~/.config/bibit-filesearch/config.yaml

    2026-09-09 语义变更：
    - 用户文件只保存「用户显式改过的键」，其余一律回落到出厂默认
      （代码 DEFAULTS）。首启**不再复制整份出厂模板**——复制会把当时的
      默认值冻结成用户选择，此后出厂默认改进永远触达不了该用户
      （query_instruction 0% 那个 bug 就是这样漏掉老用户的）。
    - 首次运行不写文件；首次保存设置时才落盘（含 config_version）。
    - 老文件在启动时一次性迁移：删除「值 == 出厂默认」的化石键 + 补版本号。
    """

    def __init__(self, config_path: Path | str | None = None, data_root: Path | None = None):
        # ★ 2026-10-08（审查 R09）：**可重入**锁，且覆盖「读 → 改 → 写」整个事务。
        #   原先 update() 在锁外 _load()、只在 _write_all() 里加锁，于是两个并发
        #   保存会各自读到同一份旧文件、各写各的，后写的把先写的字段覆盖掉（实测：
        #   两个线程分别改 top_k 与 max_depth，落盘只剩其一）。换成 RLock 是为了让
        #   update/reset 能在外层持锁的同时继续调 _write_all（普通 Lock 会自锁）。
        self._lock = threading.RLock()
        # config_path：create_app 传入的配置（测试/自定义）；None → 用户默认路径
        if config_path:
            self._config_path = Path(config_path)
        else:
            self._config_path = self._user_config_path()
        self._data_root = data_root
        self._migrate()
        self._cache: dict[str, Any] | None = None  # 内存缓存（读时刷新）

    # ---------- 路径 ----------
    def _user_config_path(self) -> Path:
        env = os.environ.get("BIBIT_FILESEARCH_CONFIG")
        if env:
            return Path(env)
        return Path.home() / ".config/bibit-filesearch/config.yaml"

    def config_path(self) -> Path:
        return self._config_path

    def _migrate(self) -> None:
        """老用户配置一次性迁移：删化石键 + 补版本号（不存在则什么都不做）。

        两类要删的键：
        ① 值 == 当前出厂默认（`_strip_fossils`）——首启复制的化石，删掉无损；
        ② 值 == 某历史版本的出厂默认（`_CHANGED_DEFAULTS`）——默认值已改进，
           用户文件还留着旧值会盖住新默认，必须删。
        用户真正改过的键（值既非当前默认也非历史默认）原样保留。
        """
        if not self._config_path.exists():
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("配置迁移读取失败 %s: %s", self._config_path, e)
            return
        if not isinstance(data, dict):
            return

        ver = data.get("config_version")
        if isinstance(ver, int) and ver >= CONFIG_VERSION:
            return  # 已是当前版本，无需迁移

        defaults = _factory_defaults()
        slim, removed = _strip_fossils(data, defaults)
        # 历史默认值残留（默认值本身已变，值不等于当前默认 → 上面漏删）
        for path, old_default in _CHANGED_DEFAULTS.items():
            val, present = _get_path(slim, path)
            if present and val == old_default:
                _del_path(slim, path)
                removed.append(path)
        # 空父层清理（例如 models.embed 只剩空 dict）
        slim = _prune_empty(slim)
        slim.pop("config_version", None)
        slim["config_version"] = CONFIG_VERSION

        try:
            self._write_all(slim)
        except Exception as e:  # noqa: BLE001
            logger.warning("配置迁移写入失败 %s: %s", self._config_path, e)
            return
        logger.info(
            "配置已迁移到 v%d: 删除 %d 个出厂默认键%s",
            CONFIG_VERSION, len(removed),
            ("（" + ", ".join(sorted(removed)[:8]) + "…）") if len(removed) > 8 else
            ("（" + ", ".join(sorted(removed)) + "）") if removed else "",
        )

    # ---------- 读写（yaml，加锁）----------
    def _load(self) -> dict:
        try:
            with open(self._config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("配置读取失败 %s: %s，使用默认", self._config_path, e)
            data = {}
        return data

    def _write_all(self, data: dict) -> None:
        with self._lock:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            # 原子写：先写临时文件再替换（防崩溃留半文件）
            tmp = self._config_path.with_suffix(".yaml.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp, self._config_path)

    # ---------- 读取 ----------
    def _default(self, key: str) -> Any:
        if key == "index_dirs":
            # 动态默认：当前用户 ~/Documents（不硬编码用户名，目录存在才给）。
            # 2026-09-09：曾追加 ~/Desktop（桌面是用户随手放文件的第二个高频位置，
            # 顺序 Desktop 在前）；2026-10-14 收敛回只索引 ~/Documents。
            # 仅存在于磁盘的目录才出现。
            home = Path.home()
            candidates = (home / "Documents",)
            return [str(p) for p in candidates if p.is_dir()]
        return DEFAULTS.get(key)

    @staticmethod
    def _dedup_if_dirs(key: str, val):
        """index_dirs 的读取侧去重（手改 config.yaml 写重复项同样不得放大计数）。"""
        return dedup_paths(val) if key == "index_dirs" else val

    def get(self, key: str) -> Any:
        """返回生效值：yaml 配置优先，缺 key 用默认。

        2026-09-04：indexable_types 的空列表是用户显式清空（"不支持任何格式"），
        必须保留 []——此前 val in (None,) 的兜底（DEFAULTS 18 类）会把文件里
        写死的 [] 当成"缺省"吞掉，读侧失真（get_all 幂等比较也受影响）。
        仅 index_dirs 的空列表走动态默认（首装用户填 ~/Documents，行为保留）。
        2026-09-20：index_dirs 去重（重复项 = 同一意图）——用户手改文件或旧版本
        落盘的重复项在此一并归一，运行时（scanner/watcher 经 main.create_app 取
        get_all）不再按份数放大计数与遍历。
        """
        data = self._load()
        if key in data:
            val = data[key]
            if key == "index_dirs" and val in (None, []):
                return self._dedup_if_dirs(key, self._default(key))
            if key == "indexable_types" and val is None:
                return self._default(key)
            return self._dedup_if_dirs(key, val)
        return self._dedup_if_dirs(key, self._default(key))

    def get_all(self) -> dict:
        """返回全部设置（yaml 值 + 默认合并；index_dirs 去重）。"""
        data = self._load()
        out = {}
        for k in DEFAULTS:
            if k == "index_dirs" and data.get(k) in (None, []):
                out[k] = self._dedup_if_dirs(k, self._default(k))
            elif k == "indexable_types" and data.get(k) is None:
                out[k] = self._default(k)
            else:
                out[k] = self._dedup_if_dirs(k, data.get(k, self._default(k)))
        return out

    # ---------- 写入 ----------
    def update(self, partial: dict) -> dict:
        """更新部分设置（写回 yaml，稀疏存储）。返回更新后的全部设置。

        2026-08-26：加类型校验（_SETTING_TYPES）——类型不符直接拒绝。
        2026-08-28：写回 config.yaml（单一来源，无 DB 覆盖陷阱）。
        2026-09-04：indexable_types 规范化（_norm_indexable_types）后再比较/落盘——
        集合语义忽略顺序，避免同内容不同序反复触发 changed。
        2026-09-09：稀疏存储——值等于出厂默认的键不落盘（删掉），只写用户真正
        改过的；同时写入 config_version。这样出厂默认的改进能自动触达老用户。
        2026-10-08（审查 R09）：整个「读 → 合并 → 写」在同一把锁内完成。此前
        `_load()` 在锁外，两个并发请求会各读一份旧文件、后写的覆盖先写的字段。
        """
        with self._lock:
            data = self._load()
            defaults = _factory_defaults()
            changed = False
            for k, v in partial.items():
                if k not in DEFAULTS:
                    logger.warning("未知设置 key 忽略: %s", k)
                    continue
                exp = _SETTING_TYPES.get(k)
                if exp and not isinstance(v, exp):
                    logger.warning("设置 %s 类型不符（期望 %s，实际 %s）忽略: %r",
                                   k, [t.__name__ for t in exp], type(v).__name__, v)
                    continue
                # 2026-09-04：max_depth 不允许负数（0=无限深）——负数归一为 0，
                # 避免 -1 穿透到引擎（引擎 <=0 也按无限处理，语义本就该是 0）。
                if k == "max_depth" and v < 0:
                    logger.warning("设置 max_depth 不允许负数 %r，按 0（无限深）处理", v)
                    v = 0
                if k == "indexable_types":
                    v = _norm_indexable_types(v)
                    cur = _norm_indexable_types(data.get(k))
                else:
                    cur = data.get(k)
                if cur != v:
                    changed = True
                # 稀疏落盘：等于出厂默认 → 删除键（让默认值保持可演进）；否则写入。
                if v == defaults.get(k):
                    data.pop(k, None)
                else:
                    data[k] = v
            # 版本号始终维护（迁移过的老文件已有；新文件首次保存时补上）
            if data.get("config_version") != CONFIG_VERSION:
                data["config_version"] = CONFIG_VERSION
                changed = True
            if changed:
                self._write_all(_prune_empty(data))
                self._cache = None
            return self.get_all()

    # ---------- 恢复默认 ----------
    def reset(self) -> dict:
        """恢复出厂默认：删除用户文件里全部「用户改过的键」。

        2026-09-09（稀疏存储语义下）：用户文件只存「与默认不同的键」，
        所以「恢复默认」= 把文件清空到只剩 config_version。生效值随即回落
        代码默认（index_dirs 走动态默认 ~/Documents）。

        保留 config_version 而非删除整个文件：避免出现「文件不存在」的中间态
        （迁移与读取路径都以文件存在为前提分支），且下次启动不会重复迁移。

        返回恢复后的完整生效设置。

        2026-10-08（审查 R09）：持整把锁——「清空 + 读回生效值」若与并发 update
        交错，用户会看到「刚保存的键又冒出来」。锁改为 RLock 后可安全嵌套
        `_write_all`（普通 Lock 不可重入，会自锁）。
        """
        with self._lock:
            if self._config_path.exists():
                # 只保留版本号，其余用户键全部丢弃
                self._write_all({"config_version": CONFIG_VERSION})
                logger.info("配置已恢复出厂默认: %s（用户自定义键已清除）", self._config_path)
            else:
                logger.info("配置已恢复出厂默认: %s（文件本就不存在）", self._config_path)
            self._cache = None
            return self.get_all()
