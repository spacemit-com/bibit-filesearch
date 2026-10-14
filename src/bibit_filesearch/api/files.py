"""文件详情 API（GET /api/files/{fileId}）。

2026-09-01 新增：前端/上游点开文件详情时，查询单文件元数据 + 分块统计。

响应：{code, message, data:{fileId,fileName,filePath,fileType,fileSize,
      createdAt,modifiedAt,author,directory,chunkCount}}
      （summary/summarySource 已移除——摘要功能暂缓，生成层已删，
       字段永远为空属死数据，留空壳造成契约错位）
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from bibit_filesearch.retriever import ERR_INTERNAL

router = APIRouter()


def _wrap(data, code: int = 0, message: str = "success") -> dict:
    return {"code": code, "message": message, "data": data}


@router.get("/files/{file_id}")
def get_file_detail(file_id: str, request: Request):
    try:
        db = getattr(request.app.state, "db", None)
        if db is None:
            return _wrap(None, code=ERR_INTERNAL, message="服务未就绪")
        f = db.get_file(file_id)
        if f is None:
            return _wrap(None, code=1004, message="文件不存在")
        # 分块统计
        try:
            chunks = db.get_chunks(file_id)
            chunk_count = len(chunks)
        except Exception:  # noqa: BLE001
            chunk_count = 0
        return _wrap({
            "fileId": f.get("file_id"),
            "fileName": f.get("name"),
            "filePath": f.get("path"),
            "fileType": f.get("file_type"),
            "fileSize": f.get("size"),
            "createdAt": f.get("created_at"),
            "modifiedAt": f.get("modified_at"),
            "author": f.get("author"),
            "directory": f.get("directory"),
            "chunkCount": chunk_count,
        })
    except Exception as e:  # noqa: BLE001
        return _wrap(None, code=ERR_INTERNAL, message=f"查询文件详情失败: {e}")
