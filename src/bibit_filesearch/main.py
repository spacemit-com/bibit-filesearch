"""bibit-filesearch 服务入口。

独立文件检索服务：
- watcher / indexer / scanner / retriever / nl_parser / settings
- 不提供：rerank / summary / mineru（PDF 只用 pymupdf）/ LLM 解析路径
- 向量化经可配置的外部 embed 端点（默认 :18790，1024 维）
- 配置默认：index_dirs=[~/Documents]、auto_index=True、全类型、
  exclude=[.git,node_modules,__pycache__]、max_depth=3、top_k=5
- 数据目录：BIBIT_FILESEARCH_DATA_ROOT 环境变量覆盖（systemd 用）
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import signal
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bibit_filesearch.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger("bibit_filesearch")


def _data_root() -> Path | None:
    """数据目录：环境变量 BIBIT_FILESEARCH_DATA_ROOT 或默认 ~/.local/share/bibit-filesearch。"""
    env = os.environ.get("BIBIT_FILESEARCH_DATA_ROOT")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    home = Path.home() / ".local/share/bibit-filesearch"
    home.mkdir(parents=True, exist_ok=True)
    return home


_SHUTDOWN_RELEASED = False


def _release_on_shutdown() -> None:
    """服务停止时**显式**放掉 OCR 的 NPU worker（幂等）。

    ★ 为什么必须有这一步（不是「反正进程要退了」）：`ep_worker` 是子进程，父进程退出
      后它读到管道 EOF 才自己退（`ep_worker.cpp` 主循环），中间有个窗口 —— 而且
      systemd 的 `KillMode=mixed` 是「SIGTERM 主进程 + 超时后 SIGKILL 其余」，
      窗口里那几个进程正占着 GB 级的 arena（实测 det 可到 4.6 GB）。
      显式 `release_idle()` 立刻 close 掉它们，内存当场归还，不留窗口。

    ★ 与空闲释放（TTL）的区别：那条是**运行期**省内存（`pool._settle` 里唯一那条，2026-10-10
      方案 A 收敛后运行期只有它），这条是**停机**时保证「不靠父死自清」。两条路径共用
      `ocr_engine.release_idle()`。

    调用点：① FastAPI lifespan 的 shutdown（uvicorn 收到 SIGTERM/SIGINT 的干净路径）
            ② `main()` 里注册的信号处理（兜底，见那里的说明）
    幂等：两者可能都触发（信号 → uvicorn → lifespan），第二次直接返回不重复记日志。
    """
    global _SHUTDOWN_RELEASED
    if _SHUTDOWN_RELEASED:
        return
    _SHUTDOWN_RELEASED = True
    try:
        from bibit_filesearch import ocr_engine

        n = ocr_engine.release_idle()
        if n:
            logger.info("服务停止：已显式释放 %d 个 OCR worker（内存当场归还）", n)
        else:
            logger.info("服务停止：无 OCR worker 需要释放")
    except Exception as e:                            # noqa: BLE001
        logger.warning("服务停止时释放 OCR worker 失败：%s", str(e)[:160])


def _lifespan(app: FastAPI):
    """应用生命周期。启动阶段留空（OCR 自检在 `main()` 里做，见其说明），停止阶段释放。

    用 lifespan 而不是 `atexit`：uvicorn 在**优雅关闭**流程里主动调用 lifespan.shutdown，
    时机早于解释器退出、也早于 systemd 的 `TimeoutStopSec` 兜底，能保证在
    「主进程被 SIGKILL」之前就跑完。
    """
    yield
    _release_on_shutdown()


def resolve_config_path(config_path: str | None = None) -> str | None:
    """把「配置从哪读」解析成一个路径，供**所有**读配置的组件共用。

    ★ 为什么必须统一（2026-10-08，审查 R06）：改前有**两套环境变量**各管一半：

        load_config()     读 BIBIT_CONFIG
        SettingsService   读 BIBIT_FILESEARCH_CONFIG

    于是只设 `BIBIT_CONFIG` 时，组件配置来自那个文件、而 `/api/settings`
    读写的却是另一个文件（`~/.config/...` 或另一个 env 指定的）。实测：

        组件层 top_k = 17（自定义文件）  设置层 top_k = 5（~/.config 文件）

    用户看到「保存成功」，运行行为却不跟；`/api/settings` 报的也不是生效值。
    两个名字都保留（老脚本可能在用），优先级与原先一致，**结果只解析一次**。

    优先级：显式参数 > BIBIT_CONFIG > BIBIT_FILESEARCH_CONFIG > None（各自默认）。
    返回 None 表示「谁也没指定，各组件按自己的默认路径走」（通常都是
    `~/.config/bibit-filesearch/config.yaml`，仍然同一处）。
    """
    if config_path:
        return str(config_path)
    return (os.environ.get("BIBIT_CONFIG")
            or os.environ.get("BIBIT_FILESEARCH_CONFIG")
            or None)


def create_app(config_path: str | None = None) -> FastAPI:
    """应用工厂：加载配置，组装服务组件，注册路由。

    装配顺序（2026-09-09 更新）：SettingsService 先于 load_config。
    - SettingsService.__init__ 会做一次性配置迁移（老用户文件里「值 == 出厂默认」
      的化石键删除 + 补 config_version），首启**不再复制** /opt 出厂模板。
    - 权威默认值在代码（config.DEFAULTS ⊕ settings.DEFAULTS），load_config 的
      分层合并直接给出完整配置，不再依赖用户文件里有那些键。
    - 默认值不经过「从模板复制」这条路径：复制一旦缺失就会落空 DEFAULTS
      （cors 空 + dim 旧值）→ GUI 跨域挂 + embed 维度不符。
    """
    data_root = _data_root()
    from bibit_filesearch.settings import SettingsService

    # ★ 2026-10-08（审查 R06）：解析成一个路径，**两个组件共用**——此前
    #   SettingsService 认 BIBIT_FILESEARCH_CONFIG、load_config 认 BIBIT_CONFIG，
    #   只设其一时界面读写与实际运行分属两个文件。见 resolve_config_path。
    config_path = resolve_config_path(config_path)
    settings_svc = SettingsService(config_path=config_path, data_root=data_root)
    config = load_config(config_path)
    app = FastAPI(title="Bibit Filesearch", version="0.1.0", lifespan=_lifespan)

    app.state.config = config
    app.state.settings = settings_svc

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- 运行时配置层（config.yaml 单一来源）：先于组件构造，scanner/watcher 才能拿到生效配置
    runtime = settings_svc.get_all()
    if runtime.get("index_dirs"):
        config.set("index_dirs", [str(d) for d in runtime["index_dirs"]])
    if runtime.get("exclude_patterns"):
        config.set("exclude_patterns", list(runtime["exclude_patterns"]))
    scan_cfg = config.get("scan", {})
    if hasattr(scan_cfg, "to_dict"):
        scan_cfg = scan_cfg.to_dict()
    else:
        scan_cfg = dict(scan_cfg or {})
    # 2026-09-04：max_depth 语义（1-based）0=无限深；1=仅 base 目录本身；
    # N=base+下 N-1 层子目录。显式 0 必须保留（不能 `or 3` 吞 0）；
    # 仅缺省/None 用默认 3；负数（手改 yaml）归一为 0（无限）。
    md = runtime.get("max_depth")
    scan_cfg["max_depth"] = 3 if md is None else max(int(md), 0)
    config.set("scan", scan_cfg)

    # ---- 服务组件 ----
    from bibit_filesearch.model_client import EmbedClient
    from bibit_filesearch.db import Database
    from bibit_filesearch.indexer import Indexer
    from bibit_filesearch.retriever import Retriever

    # 数据目录：env 覆盖 > config.storage.db_path
    db_path = config.storage.db_path
    if data_root is not None:
        db_path = str(data_root / "filesearch.db")

    db = Database(db_path)
    # embed 模型语义全部来自配置（2026-09-03 解耦：model/dim/query_instruction 切换不改代码）
    embed_cfg = config.models.embed
    embed = EmbedClient(
        embed_cfg.url,
        dim=embed_cfg.dim,
        model=embed_cfg.get("model", "bge"),
        query_instruction=embed_cfg.get("query_instruction", ""),
        # 2026-09-18：单请求输入上限（防单文件整包一发灌爆 embed 服务）
        batch_size=int(embed_cfg.get("batch_size", 32) or 32),
        # 2026-09-18：真实探活超时（见 ping）；比业务请求的 timeout 短得多——探活要快速失败
        probe_timeout=float(embed_cfg.get("probe_timeout", 5) or 5),
        # 2026-09-20：业务请求超时（read）与连接超时分离，均可配置。
        # read 是「一批 32 条算完的预算」（非流式 embeddings），默认 60s；
        # 超时后客户端会把该批队列二分（32→16→8→4）再试，所以 60s 只需覆盖
        # 「一个能过的批」，不必覆盖最坏内容。
        timeout=float(embed_cfg.get("timeout", 60) or 60),
        connect_timeout=float(embed_cfg.get("connect_timeout", 5) or 5),
        # 2026-09-20：后端接入方式（路径 / 附加字段 / 模型加载接口）全部配置化，
        # 见 config.DEFAULTS["models"]["embed"] 的说明。
        path=embed_cfg.get("path", "/v1/embeddings"),
        encoding_format=embed_cfg.get("encoding_format", ""),
        load_path=embed_cfg.get("load_path", ""),
        load_timeout=float(embed_cfg.get("load_timeout", 150) or 150),
    )

    # 2026-09-15：向量维度以 embed 服务为准，配置值只当提示（可能是老版化石）。
    # 存量索引维度与生效维度不符时在此清空重建——否则老用户升级后检索全部 1004。
    from bibit_filesearch.settings import reconcile_embed_dim

    probed_dim = embed.probe_dim()
    if probed_dim and probed_dim != embed_cfg.dim:
        logger.warning(
            "config models.embed.dim=%s 与 embed 服务实际 %s 维不符，按服务为准"
            "（配置值疑似旧版出厂化石，见 docs/troubleshooting.md 现象 G）",
            embed_cfg.dim, probed_dim,
        )
    effective_dim, _purged = reconcile_embed_dim(db, embed_cfg.dim, probed_dim)
    embed.dim = effective_dim  # 让 EmbedClient 的维度防线与检索器同源

    # parser_mod 的 PDF_MAX_OCR_PAGES 已改由 ocr_engine.apply_settings 统一赋值
    # （见下文 OCR 旋钮那段），此处不再单独引用；IMAGE_TYPES 那段自带 import。
    indexer = Indexer(db, embed, chunk_size=config.chunk.size, chunk_overlap=config.chunk.overlap,
                      # 2026-09-18：单文件块数上限（超限只索引前 N 块，防超大文件灌爆 embed）
                      # 2026-09-21：默认 32 → 64，与 config.DEFAULTS 同步（旧上限的内存依据
                      # 已作废，见 config.py 该键的注释）。
                      max_chunks_per_file=int(config.get("scan", {}).get("max_chunks_per_file", 64) or 64),
                      # 2026-09-21（A）：极短块不向量化（见 indexer.index_file 的说明）。
                      min_embed_chars=int(config.get("scan", {}).get("min_embed_chars", 3) or 0))

    # 运行时 indexable_types 必须在 scanner.start() 之前生效（2026-09-09 修复）。
    # 此前赋值放在 scanner.start() 之后：遍历线程已开跑，用 Indexer 的构造默认值
    # （DEFAULT_INDEXABLE_TYPES）判定类型，落在窗口内的文件被
    # 误标 unsupported（用户实测：配置勾了 png，11 个文件只有 6 个 completed，
    # ocr-test.png 被误判）。竞态窗口约 180ms——recovery.start() 的首次同步探测
    # 期间遍历线程已经在跑。scanner 持有同一 indexer 引用，此处赋值立即对它可见。
    #
    # 2026-10-08（审查 R02）：判据必须是 `is not None`，不能用真值判断——**空列表
    # 是用户的明确意图**（「取消全部格式」），而 `if types:` 对 [] 为假，会跳过赋值，
    # Indexer 于是保留 20 类构造默认，文件照旧被索引（实测：设置页全部取消格式 →
    # 重启后仍索引 txt 并落 completed）。PUT /api/settings 路径早就用
    # `new_types is not None` 保留空集合，启动路径漏了同一处，本项即补齐。
    # 「未配置」与「配置为空」的区别：settings.get_all() 对缺省键会回落
    # DEFAULTS['indexable_types']（非空列表），所以这里拿到 None 只可能是配置层
    # 真的没给值——那时保持构造默认才是对的。
    types = runtime.get("indexable_types")
    if types is not None:
        indexer.indexable_types = set(types)

    # OCR 旋钮（rev 5 / 2026-09-28 起只剩设备与内存档）：全部走
    # `ocr_engine.apply_settings` —— 启动路径（main() 的自检）与工厂路径必须共用
    # 同一份赋值逻辑，否则自检先构造引擎、工厂后改全局量，配置根本不生效
    # （2026-10-08，审查 R03）。旋钮清单与逐项理由见该函数。
    from bibit_filesearch import ocr_engine as ocr_engine_mod

    ocr_engine_mod.apply_settings(config.get("ocr", {}) or {})


    # ★ 2026-10-08（审查 R07）：候选池的判据必须是「缺省才用默认」，不能 `or 1000`
    #   —— `0 or 1000` 会把**显式 0** 变成 1000，而 0 是文档写明的「不截断」
    #   （`_pool_size()` 的约定与 `min_similarity<=0`、`max_depth=0` 同款）。
    #   实测：配置 0 → Retriever 1000；配置 -1 → -1（负数反而正常通过）。
    #   注意 Config.get 已把显式 null 当缺省（返回 default），所以这里只需判 None。
    _pool_cfg = config.search.get("candidate_pool", 1000)

    retriever = Retriever(
        db, embed,
        dim=effective_dim,  # 2026-09-15：探测后的生效维度（服务为准），不再直接取 config
        # 2026-09-15：候选池与「最多返回条数」**彻底解耦**——此前（0.1.8）两者都取
        # 用户 top_k，导致块级窗口随显示条数收缩：Top-K 从 10 调到 5 时，某文件的最佳
        # 块排第 8 就被切掉、分数消失、排名掉出视野（用户实测：第 1 名从 PDF 变成无关
        # 文件，且同一文件同一查询的 score 由 0.6168 掉到 0.30）。
        # 现在：候选池 = config.search.candidate_pool（检索深度，默认 1000，调试旋钮，
        # 不进界面）；top_k = 用户设置（只管返回条数上限）。两者不得再接到同一个值上。
        candidate_pool=int(_pool_cfg if _pool_cfg is not None else 1000),
        top_k=int(runtime.get("top_k", 10) or 10),
        embed_weight=config.search.embed_weight,
        filename_weight=config.search.filename_weight,
        min_similarity=config.search.get("min_similarity", 0.35),
        bm25_weight=config.search.get("bm25_weight", 0.0),
        # 2026-09-29：BM25 绝对分映射的半分点 K（见 config.py 该键的说明与
        # docs/bm25-score-calibration.md）。缺省 2.8 = 本周语料判据法标定值。
        bm25_half_score=config.search.get("bm25_half_score", 2.8),
    )
    indexer.set_on_indexed(retriever.on_file_indexed)
    indexer.set_on_deleted(retriever.on_file_deleted)

    from bibit_filesearch.pool import IndexPool
    from bibit_filesearch.scanner import ScanService
    from bibit_filesearch.watcher import Watcher

    # 索引池（重构）：scanner 与 watcher **共用同一个池子**——这是本次重构的核心，
    # 两条路径不再是两套状态机。池子由 scanner 持有消费者（IndexWorker），
    # watcher 只往池里放/撤（设计文档 §3.2 的三条铁律）。
    pool = IndexPool(db, indexer,
                     max_attempts=int(config.scan.get("max_attempts", 3) or 3))
    scanner = ScanService(config, db, indexer, pool=pool)
    app.state.scanner = scanner
    app.state.pool = pool
    watcher = Watcher(config, db, indexer, pool=pool)
    # 2026-08-28：补扫等 scanner 首次扫描完成（避免抢 embed 导致超时 failed）
    watcher.backfill_delay = max(10, int(config.scan.get("backfill_delay", 1800)))
    watcher._scanner_ref = scanner
    # 2026-09-09：auto_index=false 时不得启动 watcher。此前无条件 start()，用户关掉
    # 自动索引并重启服务后 journal 仍打「inotify 监听启动」——开关只在 PUT settings
    # 时生效，重启即被忽略。仍注册 app.state.watcher，供后续 PUT 打开时启用。
    if bool(runtime.get("auto_index", True)):
        watcher.start()
    else:
        logger.info("auto_index=false：跳过 watcher 启动（增量监听已关闭）")
    app.state.watcher = watcher

    # 解析能力升级（parser.PARSER_REVISION）：把受旧解析能力影响的文件强制重排一次。
    # 两个影响面：① 「类型在索引格式内、却因当时解析器读不到文本而落成 unsupported」的；
    # ② 「解析出了**旧版 OCR 文本**（图片 / 图片型 PDF）而落成 completed」的。
    # 不修的话这两类文件永远停在旧结果上——content_hash 是文件字节的 hash，OCR 结果
    # 变了它看不出来，重扫一律 skipped（详见 db.requeue_unsupported / requeue_ocr_text）。
    # 必须在 scanner.start() 之前做，否则遍历线程可能与重排并发。
    from bibit_filesearch import parser as parser_mod
    from bibit_filesearch.parser import PARSER_REVISION
    from bibit_filesearch.settings import reconcile_parser_revision

    requeued = reconcile_parser_revision(
        db, scanner.state, indexer.indexable_types,
        image_types=sorted(parser_mod.IMAGE_TYPES & set(indexer.indexable_types)))
    if requeued:
        logger.warning(
            "解析能力升级（rev %s）→ %d 个文件重新排队"
            "（此前判为「无文本」的按新能力重解析；图片/图片型 PDF 重新 OCR）",
            PARSER_REVISION, requeued,
        )

    # 分块规则升级（indexer.CHUNK_REVISION）：抬高单文件块数上限后，被截断的文件必须
    # 按新上限重新索引一次。不修的话 content_hash 幂等会让它们永远停在旧份数——
    # 改了上限却不生效（详见 db.requeue_truncated）。
    # legacy_cap：老库没有真实总块数（该列刚补、一律 0），只能按「是否顶在旧上限」
    # 近似判断——否则老用户一个文件都不会被重排，而水位线本次就被置成已见。
    # 同样必须在 scanner.start() 之前。
    from bibit_filesearch.indexer import (CHUNK_REVISION, CHUNK_REVISION_LEGACY_CAP,
                                          SHORT_CHUNK_REVISION_MIN_CHARS)
    from bibit_filesearch.settings import reconcile_chunk_revision

    requeued_chunks = reconcile_chunk_revision(
        db, scanner.state, legacy_cap=CHUNK_REVISION_LEGACY_CAP,
        # rev 2：把「已有向量的极短块」所在文件重排一次（A 对存量生效的唯一途径）
        short_min_chars=SHORT_CHUNK_REVISION_MIN_CHARS)
    if requeued_chunks:
        logger.warning(
            "分块规则升级（rev %s）→ %d 个文件重新排队（被块数上限截断的按新上限重索引；"
            "含极短块的摘掉旧向量；图片型 PDF 的重复 OCR 块只留一份）",
            CHUNK_REVISION, requeued_chunks,
        )

    scanner.start()
    retriever.warmup()

    # embed 恢复看门狗（2026-09-09）：embed 掉线 → 恢复后自动重跑 failed/queued 文件。
    # 否则没有任何触发点（watcher 只听文件事件、PUT settings 幂等早退），
    # 用户不打开 GUI 就永远停在 failed。
    from bibit_filesearch.recovery import EmbedRecovery

    recovery = EmbedRecovery(
        embed, db, scanner,
        # 2026-09-09：默认 30s → 5s（见 recovery.py 注释）。用户实测 start embed 后
        # 界面归零要等 25s，主因是白等下一次探测节拍。配置项仍可覆盖。
        interval=float(config.get("scan", {}).get("embed_recovery_interval", 5) or 5),
        # 2026-09-18：只把还有重试预算的文件算作待重跑 + 自动重建加冷却，
        # 打断「embed 被灌爆 OOM → 重启 → 立刻再灌一轮」的正反馈。
        max_attempts=int(config.get("scan", {}).get("max_attempts", 3) or 3),
        cooldown=float(config.get("scan", {}).get("embed_recovery_cooldown", 300) or 300),
    )
    recovery.start()
    app.state.recovery = recovery

    app.state.top_k = int(runtime.get("top_k", 10) or 10)

    app.state.db = db
    app.state.indexer = indexer
    app.state.retriever = retriever
    app.state.embed = embed  # 2026-09-01：health/deep 需要 embed 健康探测

    # 启动即按当前配置回收一次（2026-09-11）。此前回收只在 PUT /api/settings 且
    # 配置「真的变了」时执行——于是两种情况会留下与配置矛盾的索引：
    #   ① 历史 bug 先于修复落库的越界文件（规则写入成功但回收漏判）；
    #   ② 用户直接编辑 config.yaml（旁路 PUT）后重启。
    # 症状统一是「排除规则/格式/深度明明设了，文件却仍能被检索到」，且再点一次
    # 「保存并索引」也没用（payload 与当前一致 → 幂等早退，不触发回收）。
    # 此处把不变式「库内容 ⊆ 当前配置允许范围」在启动时重建一次。成本是遍历
    # files 表（O(库内文件数)），仅对越界文件调 delete_file。
    # 守卫：index_dirs 或 indexable_types 为空时跳过——reconcile 的语义是「不在
    # 允许范围内即删除」，空集合意味着「允许范围为空」→ 会把整个索引清空。PUT
    # 路径上「清空格式」是用户明确意图（已在 PUT 当场执行），启动时缺字段则更可能
    # 是配置损坏，宁可不动索引。
    startup_dirs = [str(d) for d in (runtime.get("index_dirs") or [])]
    startup_types = set(runtime.get("indexable_types") or [])
    if startup_dirs and startup_types:
        try:
            from bibit_filesearch.api.settings import _reconcile_index

            _reconcile_index(
                app, startup_dirs, list(runtime.get("exclude_patterns") or []),
                startup_types,
                3 if runtime.get("max_depth") is None else max(int(runtime["max_depth"]), 0),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("启动回收失败（不阻断启动）: %s", e)
    else:
        logger.warning("启动回收跳过：index_dirs 或 indexable_types 为空（避免误清索引）")

    # ---- 路由 ----
    # 注：FastAPI 0.141 `_IncludedRouter` 惰性 include 在本环境（starlette 1.6）
    # 下部分路由（scan）请求时 404；改为手动逐个挂载，稳定生效。
    from bibit_filesearch.api import settings as settings_api
    from bibit_filesearch.api import search as search_api
    from bibit_filesearch.api import scan as scan_api
    from bibit_filesearch.api import files as files_api
    from bibit_filesearch.api import meta as meta_api

    def _mount(router, prefix: str) -> None:
        for route in router.routes:
            app.add_api_route(prefix + route.path, route.endpoint, methods=route.methods)

    _mount(search_api.router, "/api")
    _mount(settings_api.router, "/api")
    _mount(scan_api.router, "/api")
    _mount(files_api.router, "/api")
    _mount(meta_api.router, "/api")

    @app.get("/health")
    def health():
        """轻量健康检查（统一 {code,message,data} 包装）。"""
        return {"code": 0, "message": "success", "data": {"status": "ok"}}

    @app.get("/health/deep")
    def health_deep():
        """深度健康：db 可读 / embed 在线 / 索引计数。"""
        try:
            db = getattr(app.state, "db", None)
            embed = getattr(app.state, "embed", None)

            index_files = 0
            index_errors = 0
            db_ok = True
            if db is not None:
                try:
                    index_files = db.count_files()
                    q = db.conn.execute(
                        "SELECT COUNT(*) FROM files WHERE vectorize_status='failed'"
                    ).fetchone()
                    index_errors = q[0] if q else 0
                except Exception:  # noqa: BLE001
                    db_ok = False
            else:
                db_ok = False

            embed_ok = True
            if embed is not None:
                try:
                    # ping() 自身吞掉所有异常并返回 bool；返回值必须参与判断，
                    # 否则 embed 掉线也会报 ok（2026-09-09 实测 :8082 无监听却 ok）。
                    #
                    # 2026-09-21：先看**被动**信号（最近有没有真的成功产出过向量）。
                    # 探针要与批次抢同一个串行服务——扫描期间 batch 一个接一个
                    # （单批 25~46s），探针排在队尾必然超过 probe_timeout=5s，
                    # 把满负荷工作的后端误判成 unreachable，界面于是在「扫描中」
                    # 与「索引服务不可用」之间抖动。忙时自证，闲时才真探。
                    # getattr 兜底：测试桩可能只实现 ping()。
                    healthy = getattr(embed, "is_healthy", None)
                    embed_ok = (callable(healthy) and healthy()) or bool(embed.ping())
                except Exception:  # noqa: BLE001
                    embed_ok = False
            else:
                embed_ok = False

            status = "ok" if (db_ok and embed_ok) else "degraded"
            return {
                "code": 0,
                "message": "success",
                "data": {
                    "status": status,
                    "db": "ok" if db_ok else "error",
                    "embed": "ok" if embed_ok else "unreachable",
                    # 2026-09-09：前端降级提示需要地址定位是哪个服务（config 不对外暴露 models）
                    "embedUrl": getattr(embed, "url", "") if embed is not None else "",
                    "indexFiles": index_files,
                    "indexErrors": index_errors,
                },
            }
        except Exception as e:  # noqa: BLE001
            return {"code": 1004, "message": "深度健康检查失败", "data": {"status": "error", "message": str(e)}}

    logger.info("bibit-filesearch 启动: db=%s embed=%s", db.db_path, config.models.embed.url)
    return app


def ocr_selfcheck() -> None:
    """启动时证明 OCR 引擎真能用：构造一次 + **真跑一次推理**。

    ★ 为什么必须真跑、而不是判 import：实测 ep_worker 不在时 `import rapidocr` 照样
      成功，postinst 因此谎报过「模型就绪」——真正坏掉的是推理那一步。
    ★ 为什么放在这里、而不是 create_app 里（2026-09-28 订正）：create_app 是**应用工厂**，
      测试会调它几十次（每个用例一个临时 app）。放在里面等于每次用例都构造引擎 +
      起三个 ep_worker 子进程：实测单次 create_app 从 0.2s 涨到 8~19s、fd 累积到 121
      （进程上限 1024），整套测试跑不到尾就 EMFILE、批量报假错。生产路径只经 main()，
      所以自检挂在这里足够。
    ★ 失败只告警不阻断：图片索引会据实落 failed（见 indexer），文本类文件不受影响；
      「服务起不来」比「图片索引不可用」更糟。
    """
    _t0 = time.time()
    try:
        from PIL import Image
        import numpy as _np
        from bibit_filesearch import ocr_engine as _oe
        _probe = Image.new("RGB", (320, 80), "white")
        _txt, _conf = _oe.recognize_ppocr(_np.array(_probe))
        logger.info("OCR 自检通过：设备=%s，预热 %.1fs", _oe.describe().get("device"),
                    time.time() - _t0)
    except Exception as _e:                       # noqa: BLE001
        logger.error("OCR 自检失败：图片将索引为失败（文本类文件不受影响）：%s", str(_e)[:200])


def _port_for_uid(uid: int) -> int:
    """per-user 端口：独立段 18700（不冲突 runtime 9899 段）。

    uid<1000 → 18700；否则 18700 + (uid-1000)（per-user 唯一，对齐 runtime 的
    per-uid 派生思路，但独立端口段避免与 runtime/其他服务冲突）。
    """
    if uid < 1000:
        return 18700
    return 18700 + (uid - 1000)


def main() -> None:
    """CLI 入口（uvicorn 启动）。

    子命令：
      serve           启动服务（默认端口按 uid 派生：18700 段）
      serve --port N  指定端口
    """
    import argparse

    parser = argparse.ArgumentParser(description="bibit-filesearch 服务")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="启动服务")
    serve.add_argument("--config", default=None, help="config.yaml 路径")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=None,
                       help="端口（默认按 uid 派生，per-user）")
    args = parser.parse_args()

    if args.command != "serve":
        parser.print_help()
        return

    port = args.port
    if port is None:
        port = _port_for_uid(os.getuid())

    # ★ 2026-10-08（审查 R05）：`--config` 以前是**死旗标**——`uvicorn.run()` 只收到
    #   字符串工厂名 `bibit_filesearch.main:create_app`，factory 调用 create_app()
    #   不传参，args.config 从未被使用。`serve --config /tmp/x.yaml` 会静默落到真实
    #   用户的配置与数据目录（开发隔离实验踩过：以为在用临时配置，其实写脏了本机
    #   库）。这里先解析（`--config` 与其他 env 统一口径，见 resolve_config_path），
    #   再用闭包工厂把它带进 create_app。
    config_path = resolve_config_path(args.config)

    # ★ 2026-10-08（审查 R03）：OCR 旋钮必须在自检**之前**生效——自检会构造并缓存
    #   引擎，之后 create_app 再改全局量就晚了（引擎已是老参数），配置静默不生效。
    #   与 create_app 走同一个 apply_settings，保证两条路径同源。
    try:
        from bibit_filesearch.config import load_config
        from bibit_filesearch import ocr_engine as _oe

        _oe.apply_settings(load_config(config_path).get("ocr", {}) or {})
    except Exception as e:                            # noqa: BLE001
        logger.warning("启动前应用 OCR 配置失败（用默认值继续）：%s", str(e)[:160])

    ocr_selfcheck()          # 见其说明：只在这里做（create_app 会被反复调用）

    # 兜底：uvicorn 的干净关闭会走 lifespan（我们已在那里释放），但**非**优雅路径
    # （例如 uvicorn 自身启动失败、或将来换启动方式）不会。信号处理里再挂一道，
    # 幂等，重复触发无副作用。
    def _on_signal(signum, _frame):
        logger.info("收到信号 %s，释放 OCR worker 后退出", signum)
        _release_on_shutdown()
        raise SystemExit(0)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _on_signal)
        except (ValueError, OSError):                 # noqa: PERF203 —— 非主线程/受限环境
            logger.debug("信号 %s 注册失败（非主线程？），仅依赖 lifespan 释放", _sig)

    # 闭包工厂：字符串形式（factory=True）无法携带参数，那正是 `--config` 失效的
    # 直接原因。host/port/log_level 与原先的 uvicorn.run 调用保持一致。
    def _app_factory() -> FastAPI:
        return create_app(config_path)

    uvicorn.Server(uvicorn.Config(
        _app_factory, host=args.host, port=port, log_level="info")).run()


if __name__ == "__main__":
    main()
