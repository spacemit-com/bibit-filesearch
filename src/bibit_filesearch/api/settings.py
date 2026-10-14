"""设置 API：GET/PUT /api/settings、POST /api/settings/reset。

- GET 返回当前生效配置（用户文件稀疏覆盖 + 代码默认合并）
- PUT 更新用户设置 → 写 config.yaml → 立即生效：
    · index_dirs / exclude_patterns / max_depth / indexable_types → 更新 scanner/watcher
      并重启扫描 + reconcile 回收（不再属于索引范围的文件从向量库删除）
    · auto_index → watcher 启停
    · top_k → 写入内存（检索端点动态读取）
- POST /api/settings/reset 恢复出厂默认：清空用户文件里的自定义键（2026-09-09），
  生效值回落代码默认并无条件重扫。

契约：code/message/data 三层；data 为完整设置对象。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from bibit_filesearch.excludes import is_excluded
from bibit_filesearch.retriever import ERR_PARAM

logger = logging.getLogger("bibit_filesearch.settings_api")

router = APIRouter()


def _wrap(data, code: int = 0, message: str = "success") -> dict:
    """统一 {code, message, data} 包装。"""
    return {"code": code, "message": message, "data": data}


def _reconcile_index(app, index_dirs: list[str], excludes: list[str], types: set[str],
                     max_depth: int = 0) -> None:
    """回收不再属于索引范围的文件（走 delete_file 同步内存向量/BM25）。

    2026-08-28：配置变更（取消格式/移除目录/新增排除）后，已索引文件若 hash 未变
    会被扫描跳过、旧索引残留——此处全库核对删除。
    2026-09-04：max_depth 收窄（如 3→1）时，已入库的深层文件（超出新深度）同样
    残留——scanner._walk 只拦新扫描，hash 幂等跳过旧文件，reconcile 原不比较深度。
    此处加深度维度（1-based 口径，与 scanner._walk 一致）：文件所在目录相对其
    index_dir base 的子目录层数 d（文件在 base 本身=0，base/sub=1 …）>= max_depth
    即回收删除——max_depth=1 只索引 base 本身，故 d>=1 全回收。
    2026-09-11：排除判据改为 excludes.is_excluded（按路径分量逐级匹配）。此前只拿
    「完整路径 + 文件名」比对，规则写父目录名（如 `skipme`）时两者都对不上 →
    已在库文件永不回收（用户实测：加了排除规则保存后文件仍可搜到）。
    """
    db = getattr(app.state, "db", None)
    indexer = getattr(app.state, "indexer", None)
    if db is None or indexer is None:
        return
    # 2026-09-23（重构）：回收走索引池。此前直接 `indexer.delete_file`，
    # 会**删掉正在被消费者处理的那一行**——消费者算完还会把行写回来
    # （`index_file` 的末尾 upsert），于是被回收的文件重新出现在库中。
    # `pool.purge` 会区分「未消费（直接删）」与「正在算（打取消标记，算完丢弃）」。
    pool = getattr(app.state, "pool", None)

    def _drop(file_id: str, path: str):
        if pool is not None:
            pool.purge(path)
        else:
            indexer.delete_file(file_id)

    dirs = [str(d).rstrip("/") for d in index_dirs]
    removed = 0
    for f in db.all_files():
        path = f.get("path") or ""
        ftype = f.get("file_type") or ""
        # 文件可能落在多个 index_dir 下（嵌套配置）：按实际路径前缀匹配的最长 base
        # 计算深度（与"最具体目录作为索引根"的扫描语义一致）。
        base = max(
            (d for d in dirs if path == d or path.startswith(d + "/")),
            key=len,
            default=None,
        )
        excluded = is_excluded(path, excludes)
        too_deep = False
        depth = -1
        if max_depth > 0 and base is not None:
            # 与 scanner._walk 同口径（1-based）：文件所在目录相对 base 的子目录层数
            # （base 本身=第 0 层，base/sub=1 …），>= max_depth 即超出新深度——
            # max_depth=1 只索引 base 本身，base 直接子目录里的文件也要回收。
            parent = path.rsplit("/", 1)[0]
            depth = parent.count("/") - base.count("/")
            too_deep = depth >= max_depth
        # 2026-08-31：图片类型也受"支持的格式"约束（与 indexer 一致）。
        # 只回收有索引状态的文件（completed/queued/vectorizing）——unsupported
        # 本来就没索引，保留（重扫时 _enqueue_file 会按当前 types 重新评估，
        # 避免"回收→重扫→再入库→再回收"循环）
        if base is None or excluded or ftype not in types or too_deep:
            if f.get("vectorize_status") != "unsupported":
                if too_deep:
                    logger.info(
                        "索引回收(深度): %s 超出 max_depth=%d（所在目录深度 %d）",
                        path, max_depth, depth,
                    )
                else:
                    logger.info("索引回收: %s", path)
                _drop(f["file_id"], path)
                removed += 1
    if removed:
        logger.info("索引回收完成: 移除 %d 个文件", removed)


def _validate_index_dirs(index_dirs: list[str]) -> list[str]:
    """校验知识库路径（保存前拦截），返回错误信息列表（空=全部合法）。

    2026-09-07：此前 PUT 不校验路径，用户填错（不存在/不可访问/是文件）时
    保存"成功"但扫描静默跳过/索引被清空，无任何提示。此处单点拦截：
      - 悬空 symlink / 不存在     → "不存在"
      - 存在但是文件             → "不是目录"
      - 存在但服务用户读不了      → "无访问权限"
    以 Path.exists()（跟随 symlink）判存在；可读性用 os.access(R_OK|X_OK)，
    但以 root 跑时 os.access 恒 True（root 无视权限）——服务实际以桌面用户
    （bianbu）跑 user systemd，此处按运行用户判（root 场景跳过可读校验，
    由运行环境保证；生产 deb 服务以非 root 跑，校验有效）。
    """
    if not index_dirs:
        return []
    errors: list[str] = []
    for d in index_dirs:
        p = Path(d)
        if not p.exists():
            errors.append(f"{d}（不存在）")
            continue
        if not p.is_dir():
            errors.append(f"{d}（不是目录）")
            continue
        # 可读性：仅非 root 时校验（root 无视权限，判了恒过；服务以 root 跑
        # 属 dev 场景，由用户自行保证权限）
        if os.geteuid() != 0 and not os.access(d, os.R_OK | os.X_OK):
            errors.append(f"{d}（无访问权限）")
    return errors


def _normalize_index_dirs(raw_dirs: list) -> tuple[list[str], list[str]]:
    """规范化知识库路径：strip → 拒绝空项 → ~ 展开 → 拒绝相对路径 → 去重。

    返回 (规范化后的绝对路径列表, 错误列表)。错误非空时调用方应整单拒绝。
    2026-09-07：从 PUT handler 抽出——规范化必须发生在幂等比较**之前**，
    否则用户填 ~/x 保存（落盘绝对路径）后表单仍显 ~/x，再保存 raw != 绝对
    → 误判 changed → 每次空转全量 reconcile+重扫（D1）。
    2026-09-20：末尾去重（保留首次出现顺序）——重复项会放大扫描计数并白扫磁盘；
    且去重必须发生在幂等比较之前，否则 [A] vs [A,A] 恒判 changed → 反复重扫。
    判据是完全相同的路径（`~` 已展开）；嵌套重叠属另一类判据，不在此处理。
    """
    from bibit_filesearch.settings import dedup_paths

    norm: list[str] = []
    errors: list[str] = []
    for d in raw_dirs:
        d = str(d).strip()
        if not d:
            errors.append("存在空路径项")
            continue
        if d.startswith("~"):
            d = str(Path.home() / d.lstrip("~/").lstrip("/"))
        if not d.startswith("/"):
            errors.append(f"{d}（请填写绝对路径）")
            continue
        norm.append(d)
    return dedup_paths(norm), errors


def _apply_settings(request: Request, settings: dict) -> None:
    """把设置应用到运行中的 scanner/watcher/retriever/indexer（无需重启服务）。"""
    app = request.app
    scanner = getattr(app.state, "scanner", None)
    watcher = getattr(app.state, "watcher", None)
    cfg = getattr(app.state, "config", None)

    index_dirs = [str(d) for d in (settings.get("index_dirs") or [])]
    excludes = list(settings.get("exclude_patterns") or [])
    # 2026-09-04：max_depth 语义（1-based）0=无限深；1=仅 base 目录本身；
    # N=base+下 N-1 层子目录。显式 0 必须保留（不能 `or 3` 吞 0）；
    # 仅缺省/None 用默认 3；负数归一为 0（update 已归一，此处兜底手改 yaml）。
    md = settings.get("max_depth")
    max_depth = 3 if md is None else max(int(md), 0)
    failures: list[str] = []

    if cfg is not None:
        try:
            cfg.set("index_dirs", index_dirs)
            cfg.set("exclude_patterns", excludes)
            scan_cfg = cfg.get("scan")
            if hasattr(scan_cfg, "to_dict"):
                scan_cfg = scan_cfg.to_dict()
            else:
                scan_cfg = dict(scan_cfg or {})
            scan_cfg["max_depth"] = max_depth
            cfg.set("scan", scan_cfg)
        except Exception as e:
            logger.warning("更新内存 config 失败: %s", e)
            failures.append("config")

    # 先应用 indexable_types 到 indexer（reconcile 要用新值判定）。
    # 2026-09-04：空列表也是有效意图（"取消全部格式"）——`if new_types` 对 [] 为
    # falsy 会跳过，导致清空格式后 indexer 仍保留旧类型、reconcile 不回收任何文件。
    try:
        indexer = getattr(app.state, "indexer", None)
        new_types = settings.get("indexable_types")
        if indexer is not None and new_types is not None:
            indexer.indexable_types = set(new_types)
    except Exception as e:
        logger.warning("应用 indexable_types 失败: %s", e)
        failures.append("indexable_types")

    # 回收：删除不再属于索引范围的文件（类型/目录/排除变化）+ 深度超出新 max_depth 的文件
    #
    # ★ 2026-09-23：顺序要紧 —— **先停 watcher，再回收**。
    #
    # 此前次序是「回收 → 改 watcher 配置 → watcher.stop()/start()」，而 `stop()`
    # 只置停止位、**不等正在跑的那一轮补扫结束**：如果此刻 `_backfill_dir` 正拿着
    # **旧的** index_dirs / exclude_patterns 走目录，它会在回收之后继续 offer，
    # 把刚被回收的文件原样加回库中。用户可见的后果是「加了排除规则保存，文件
    # 稍后又冒出来」——而且窗口随补扫规模变化，越大的知识库越容易中。
    #
    # 实测（未改动的基线代码同样复现，非本次重构引入）：
    #   同一份 `tests/test_t12_settings.py` 连跑 20 次 → 2 次失败，分别是
    #   · `app.log` 该回收却没回收（旧 excludes 下它不该被排除，补扫加了回来）
    #   · `top.txt` / `keep.txt` 不该回收却被删（旧 index_dirs 下重新入池后再被
    #     下一轮按新规则回收，或补扫重建后与断言时刻交错）
    #   单跑该文件 19/19 通过（单跑时补扫通常已结束，碰不到这个窗口）。
    #
    # 先 stop() 再回收，就把「旧轮补扫」这个竞态源头掐掉了：回收发生在补扫结束
    # 之后，此后新起的补扫用的是新配置。
    if watcher is not None:
        try:
            auto = bool(settings.get("auto_index", True))
            if auto:
                watcher.index_dirs = index_dirs
                watcher.excludes = excludes
                watcher.max_depth = max_depth
                watcher.stop()
            else:
                watcher.stop()
        except Exception as e:
            logger.warning("停止 watcher 失败: %s", e)
            failures.append("watcher")

    try:
        indexer = getattr(app.state, "indexer", None)
        types = set(indexer.indexable_types) if indexer is not None else set()
        _reconcile_index(app, index_dirs, excludes, types, max_depth)
    except Exception as e:
        logger.warning("索引回收失败: %s", e)
        failures.append("reconcile")

    if scanner is not None:
        try:
            scanner.index_dirs = index_dirs
            scanner.excludes = excludes
            scanner.max_depth = max_depth
            scanner.restart()
        except Exception as e:
            logger.warning("应用 scanner 设置失败: %s", e)
            failures.append("scanner")

    # 回收完成后再放开增量监听（用新配置起的补扫不会再命中已回收的文件）
    if watcher is not None:
        try:
            if bool(settings.get("auto_index", True)):
                watcher.start()
        except Exception as e:
            logger.warning("重启 watcher 失败: %s", e)
            failures.append("watcher")

    try:
        top_k = int(settings.get("top_k", 10) or 10)
        app.state.top_k = top_k
        retriever = getattr(app.state, "retriever", None)
        if retriever is not None:
            # 只同步「返回条数上限」。**不动候选池**——它由 config.search.candidate_pool
            # 决定（检索深度），与显示条数无关。2026-09-15 前这里把候选池也设成 top_k，
            # 是「改 Top-K 导致排序变化」的直接入口之一（另一个是 main.py 的构造参数）。
            retriever.top_k = top_k
    except Exception as e:
        logger.warning("应用 top_k 失败: %s", e)
        failures.append("top_k")

    if failures:
        logger.warning("设置部分未生效: %s", ", ".join(failures))


@router.get("/settings")
def get_settings(request: Request):
    """返回当前生效配置。"""
    svc = request.app.state.settings
    return {"code": 0, "message": "success", "data": svc.get_all()}


@router.put("/settings")
def put_settings(payload: dict[str, Any], request: Request):
    """更新用户设置并立即生效（幂等：payload 与当前一致时跳过重应用）。

    2026-09-01：幂等跳过时返回 data.changed=false，让上游明确区分
    "真的更新了" vs "配置未变化已跳过"（原先两种情况都返回 data 当前值，
    前端无法区分，误提示"保存成功"）。
    2026-09-07：index_dirs 先规范化（~展开/拒相对路径）再参与幂等比较——
    否则 ~/x 与落盘绝对路径比较失真导致每次保存都误判 changed（D1）；
    校验失败整单拒绝（code=ERR_PARAM），不落盘不重扫。
    """
    svc = request.app.state.settings
    updates = {k: v for k, v in payload.items() if v is not None}
    current = svc.get_all()

    # 规范化 + 校验 index_dirs（幂等比较之前）：坏路径整单拒绝
    if updates.get("index_dirs") is not None:
        norm_dirs, norm_errors = _normalize_index_dirs(updates["index_dirs"])
        if norm_errors:
            logger.warning("PUT /api/settings 拒绝: 知识库路径不合法 %s", norm_errors)
            return {"code": ERR_PARAM, "message": "知识库路径不合法: " + "; ".join(norm_errors),
                    "data": {"changed": False}}
        errs = _validate_index_dirs(norm_dirs)
        if errs:
            logger.warning("PUT /api/settings 拒绝: 知识库路径不可用 %s", errs)
            return {"code": ERR_PARAM, "message": "知识库路径不可用: " + "; ".join(errs),
                    "data": {"changed": False}}
        updates["index_dirs"] = norm_dirs

    # 2026-08-31：幂等——payload 与当前生效配置一致时跳过重应用。
    # 前端异常循环保存（同一配置反复 PUT）会反复触发 reconcile+重扫，
    # 且扫描把 unsupported 文件重新入队，造成"回收→重扫→再回收"循环
    # 2026-09-04：indexable_types 集合语义——忽略顺序比较（Set 勾选顺序漂移
    # 会误判 changed，见 settings._norm_indexable_types）
    def _same(k, a, b):
        if k == "indexable_types":
            return set(a or []) == set(b or [])
        return a == b

    changed = any(
        updates.get(k) is not None and not _same(k, updates[k], current.get(k))
        for k in updates
    )
    if not changed:
        return {"code": 0, "message": "success", "data": {**current, "changed": False}}

    updated = svc.update(updates)
    _apply_settings(request, updated)
    return {"code": 0, "message": "success", "data": {**updated, "changed": True}}


@router.post("/settings/reset")
def reset_settings(request: Request):
    """恢复出厂默认：清除用户文件里所有自定义键，并立即重新应用。

    2026-09-09（稀疏存储语义）：用户文件只存「与默认不同的键」，所以恢复默认
    = 清空这些键 → 生效值全部回落代码默认（index_dirs 走动态默认
    ~/Documents）。

    语义与 PUT 的区别：
    - PUT 是「改」，值等于默认时只是删键；reset 是「全部清空」，无论改没改过。
    - 无条件重应用（不比较 changed）——用户点这个按钮就是要让默认值重新生效
      并重扫（等价于「保存并索引」的强制入口），不能因「看起来没变化」而早退。

    返回 data 为恢复后的完整生效设置（不含 changed 字段——前端不需要）。
    """
    svc = request.app.state.settings
    restored = svc.reset()
    _apply_settings(request, restored)
    return {"code": 0, "message": "success", "data": restored}
