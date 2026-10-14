"""元数据提取。

字段：文件名、类型/扩展名、大小、创建/修改/访问时间、作者、目录路径。
创建时间回退链：birth time → ctime → 留空。
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

logger = logging.getLogger("bibit.metadata")

# 本地时区（2026-08-26 修复 UTC 跨日 bug：时间戳必须本地时区，否则"今天/昨天"筛选漏掉凌晨文件）
# 东八区等：动态取当前进程时区，避免硬编码
_LOCAL_TZ = dt.datetime.now().astimezone().tzinfo


def _iso(ts: float | None) -> str:
    """时间戳 → 本地时区 ISO 字符串（如 2026-08-27T00:30:00+08:00）。

    2026-08-26 修复：原用 UTC（...Z），导致本地凌晨创建的文件日期差一天——
    搜"今天的文档"漏掉凌晨文件。改为本地时区，与 NL 解析的本地日期一致。
    """
    if ts is None:
        return ""
    return dt.datetime.fromtimestamp(ts, tz=_LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S%z")


def _get_birthtime(p: Path) -> float | None:
    """尝试取文件系统 birth time（Linux statx stx_btime / os.stat st_birthtime）。

    2026-08-26：Python 3.14 的 os.stat 在支持 btime 的文件系统上返回 st_birthtime，
    无需 os.statx 冗余分支（原双路 try/except 吞异常且 statx 的 AT_SYMLINK_NOFOLLOW
    在部分环境行为不一致）。
    """
    try:
        st = p.stat()
        b = getattr(st, "st_birthtime", None)
        if b is not None and b > 0:
            return float(b)
    except Exception:
        pass
    return None


def extract_metadata(path: str | Path) -> dict:
    """提取文件元数据 → dict（字段与 /api/search 契约对齐）。"""
    p = Path(path)
    st = p.stat()
    ftype = _file_type(p)
    btime = _get_birthtime(p)

    return {
        "fileId": "",  # 入库时分配（T4 起由 db 层生成）
        "fileName": p.name,
        "filePath": str(p),
        "fileType": ftype,
        "fileSize": st.st_size,
        "createdAt": _iso(btime) if btime else _iso(st.st_ctime),
        "modifiedAt": _iso(st.st_mtime),
        "accessedAt": _iso(st.st_atime),  # 内部字段，前端不展示
        "author": "",  # 由解析器填充（docx/pdf）
        "directory": str(p.parent),
        "contentHash": "",  # 由索引层填充（T4）
    }


def _file_type(p: Path) -> str:
    from .parser import file_type_of

    return file_type_of(p)
