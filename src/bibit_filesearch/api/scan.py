"""扫描管理 API。

- POST /api/scan         启动扫描（幂等）
- GET  /api/scan/progress 两阶段进度
- POST /api/scan/pause   暂停（作用于整个任务）
- POST /api/scan/resume  继续

2026-09-01：统一响应为 {code, message, data} 三层包装（原裸对象返回
与 settings/search 不一致，上游需两套解析）。扫描状态机：
idle / scanning / paused / done / error。
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from bibit_filesearch.retriever import ERR_INTERNAL

router = APIRouter()


def _wrap(data: dict, code: int = 0, message: str = "success") -> dict:
    """统一 {code, message, data} 包装。"""
    return {"code": code, "message": message, "data": data}


@router.post("/scan")
def start_scan(request: Request):
    res = request.app.state.scanner.start()
    return _wrap(res)


@router.get("/scan/progress")
def scan_progress(request: Request):
    try:
        return _wrap(request.app.state.scanner.progress())
    except Exception as e:  # noqa: BLE001
        return _wrap({"status": "error", "message": str(e)}, code=ERR_INTERNAL,
                     message="扫描进度查询失败")


@router.post("/scan/pause")
def pause_scan(request: Request):
    return _wrap(request.app.state.scanner.pause())


@router.post("/scan/resume")
def resume_scan(request: Request):
    return _wrap(request.app.state.scanner.resume())
