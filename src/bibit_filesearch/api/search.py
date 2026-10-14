"""搜索 API（POST /api/search）。

- 单一入口：前端/工具始终调 POST /api/search
- NL 解析由后端统一走 parse_nl_query（规则确定性，3ms）
- filters：{fileTypes, dateRange, directories, dateField}（透传 retriever）
- 响应：{code, message, data:{total, page, pageSize, totalPages, results[]}}
  results[]: {fileId, fileName, filePath, fileType, fileSize, createdAt,
              modifiedAt, author, confidence}
  已移除：summary / summarySource（摘要功能暂缓——2026-08-28 首次移除，2026-09-01
  契约统一时曾临时补回，随后再次移除（留空壳会误导上游）、rerank、LLM 解析
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from bibit_filesearch.retriever import ERR_PARAM

router = APIRouter()


def _config_ctx(request: Request) -> tuple[list[str] | None]:
    """从 app.state 读取 index_dirs（容错：缺失时 None）。"""
    cfg = getattr(request.app.state, "config", None)
    index_dirs = None
    if cfg is not None:
        try:
            index_dirs = [str(d) for d in (cfg.index_dirs or [])]
        except Exception:
            index_dirs = None
    return (index_dirs,)


def _do_nl_search(payload: dict[str, Any], request: Request) -> dict:
    """NL 解析检索（规则确定性路径，不调 LLM）。

    响应与 /api/search 完全一致；data.debug 回传实际生效的条件/关键词。
    """
    from bibit_filesearch.nl_parser import parse_nl_query

    query = str(payload.get("query") or "").strip()
    index_dirs, = _config_ctx(request)

    parsed = parse_nl_query(query, index_dirs=index_dirs)
    filters = parsed.filters or {}
    # 2026-09-04：不再 or query 回退整句——purify 净化为空 = 纯筛选意图
    #   （"本地搜索pdf文件" → kw='' + fileTypes:[pdf]，应返回全部 pdf；
    #   回退整句会把触发词当语义词 embed，纯列举场景 total=0）。
    #   retriever 已支持空 query + filters 纯筛选（_do_search 1.5 分支）；两者皆空由 retriever 报 1001。
    keywords = parsed.query

    body = {
        "query": keywords,
        "filters": filters,
        # 2026-09-09：原样透传（勿 int() 强转）——非数字/显式 0 交给 retriever._validate
        # 返回 1001 信封；此前 `int(...)` 在 handler 里抛 ValueError → HTTP 500。
        # 缺省仍按本层默认（page=1 / pageSize=10，与契约示例一致），用 is None 判断
        # 而非 `or`——`or 1` 会把显式 0 吞成 1 静默放行。
        "page": 1 if payload.get("page") is None else payload.get("page"),
        "pageSize": 10 if payload.get("pageSize") is None else payload.get("pageSize"),
    }
    retriever = request.app.state.retriever
    result = retriever.search(body)
    # 2026-09-01：debug 信息放入 data 内（原放顶层 code 旁，破坏 {code,message,data} 契约）
    # 2026-09-09：出错时 retriever 返回 data=None，setdefault 不会替换已存在的键
    # → `None["debug"]` 抛 TypeError 变成 HTTP 500（embed 掉线检索实测）。仅在
    # data 是 dict 时附加 debug。
    if isinstance(result.get("data"), dict):
        result["data"]["debug"] = {
            "route": "nl",
            "filters": filters,
            "keywords": keywords,
            "unsupported": parsed.unsupported_dirs,
        }
    return result


@router.post("/search")
def search(payload: dict[str, Any], request: Request):
    """语义检索：query + 可选 filters/page/pageSize。"""
    query = str(payload.get("query") or "").strip()
    if not query:
        return {"code": ERR_PARAM, "message": "query 不能为空", "data": {"total": 0, "results": []}}

    # 自然语言 → 结构化条件（规则确定性；NL 意图时走 _do_nl_search）
    index_dirs, = _config_ctx(request)
    from bibit_filesearch.nl_parser import parse_nl_query

    parsed = parse_nl_query(query, index_dirs=index_dirs)
    filters = payload.get("filters") or parsed.filters or {}
    # 2026-09-04：同 _do_nl_search——净化空串即纯筛选，不 or query 回退整句
    keywords = parsed.query

    retriever = request.app.state.retriever
    result = retriever.search({
        "query": keywords,
        "filters": filters,
        # 2026-09-09：同 _do_nl_search——原样透传，非法值由 retriever._validate 报 1001
        "page": 1 if payload.get("page") is None else payload.get("page"),
        "pageSize": 10 if payload.get("pageSize") is None else payload.get("pageSize"),
    })
    # 2026-09-01：debug 信息放入 data 内（原放顶层 code 旁，破坏 {code,message,data} 契约）
    # 2026-09-09：同 _do_nl_search——data=None（内部错误）时不可下标赋值，否则 500
    if isinstance(result.get("data"), dict):
        result["data"]["debug"] = {
            "route": "rule",
            "filters": filters,
            "keywords": keywords,
            "unsupported": parsed.unsupported_dirs,
        }
    return result
