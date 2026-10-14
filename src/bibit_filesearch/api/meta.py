"""实例元信息 API（GET /api/meta）。

2026-09-08 新增（多用户支持）：服务端暴露当前运行实例的归属元信息——
uid / userName / homeDir / 派生端口。每 Linux 用户一套 filesearch（以
user systemd 运行），插件/宿主侧借此确认 :18700 段实例属于哪个用户，
避免归属错乱（拿错实例的 uid 去操作）。

端口语义：serve 子命令未显式指定 --port 时按 uid 派生
（main._port_for_uid：uid<1000 → 18700，否则 18700+(uid-1000)）；
config server.port 只是历史默认 8050，并非 serve 实际使用的来源——此处
必须与 _port_for_uid 同源，否则上报端口与实际监听不符。
"""
from __future__ import annotations

import logging
import os
import pwd
from pathlib import Path

from fastapi import APIRouter

from bibit_filesearch.retriever import ERR_INTERNAL

logger = logging.getLogger("bibit.meta_api")

router = APIRouter()


def _wrap(data: dict, code: int = 0, message: str = "success") -> dict:
    """统一 {code, message, data} 包装。"""
    return {"code": code, "message": message, "data": data}


def ocr_status() -> dict:
    """OCR 引擎自述（rev 5）。放在这里是为了「到底在跑什么引擎/设备」可查，
    而不是只能翻日志——旧版正是只能翻日志，才出现「报了就绪、实际跑不动」。"""
    try:
        from bibit_filesearch import ocr_engine as oe

        return oe.describe()
    except Exception as e:                            # noqa: BLE001
        return {"engine": "ppocr", "ready": False, "error": "探测失败：%s" % str(e)[:160]}


@router.get("/meta")
def get_meta():
    """当前实例归属：uid / userName / homeDir / uid 派生端口。"""
    try:
        # 函数内导入：避免 api 包与 main 的循环依赖，且运行期 main 必已加载
        from bibit_filesearch.main import _port_for_uid

        uid = os.getuid()
        pw = pwd.getpwuid(uid)
        return _wrap({
            "uid": uid,
            "userName": pw.pw_name,
            "homeDir": str(Path.home()),
            "port": _port_for_uid(uid),
            "ocr": ocr_status(),
        })
    except Exception as e:  # noqa: BLE001
        logger.warning("meta 查询失败: %s", e)
        return _wrap({}, code=ERR_INTERNAL, message=f"元信息查询失败: {e}")
