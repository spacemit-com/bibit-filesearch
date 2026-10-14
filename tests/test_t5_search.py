"""检索服务 + /api/search 契约 + P95 性能基准测试。

覆盖：
- 契约字段（十字段 + code/message/data 三层）
- 排序（confidence 降序，同分 modifiedAt 降序）
- 筛选组合（fileTypes / dateRange / directories 递归前缀 + 大小写敏感）
- 分页（total / page / pageSize / totalPages）
- 错误码（1001 参数 / 1002 筛选非法 / 1004 内部）
- 文件级结果（Top-K = pageSize，非文本块）
- P95 ≤ 100ms 基准（1 万块规模模拟）
"""
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest


from fake_embed import FakeEmbed  # noqa: E402
from bibit_filesearch.db import Database, _pack_vec  # noqa: E402
from bibit_filesearch.retriever import Retriever  # noqa: E402
from bibit_filesearch.indexer import Indexer  # noqa: E402



@pytest.fixture(scope="module")
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_t5_"))
    db = Database(str(tmp / "test.db"))
    embed = FakeEmbed()
    idx = Indexer(db, embed)
    # min_similarity=0：FakeEmbed 是词袋尺度（cos 普遍 ~0.2），契约测试只验
    # 筛选/分页/排序结构，不验语义阈值（0.35 是真实模型实测值，与维度同属环境参数）
    # top_k=10：本文件是**契约**测试（形状/语义），必须让 _seed 的 8 个文件全部落在
    # 返回上限内才能断言「命中全部 8 个」。生产默认已改 5（2026-10-14），此处显式钉住
    # 一个大于 8 的值，免得默认值变动再次让契约测试误报。默认值本身由
    # test_config_migration / test_settings_reset 守护。
    retriever = Retriever(db, embed, candidate_pool=50, embed_weight=0.7, filename_weight=0.3,
                          min_similarity=0.0, top_k=10)
    idx.set_on_indexed(retriever.on_file_indexed)
    idx.set_on_deleted(retriever.on_file_deleted)
    yield db, embed, idx, retriever, tmp
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _seed(env, n_files=8):
    """造一批多样化的测试文件（用 os.utime 控制 mtime 以验证日期筛选与平局排序）。"""
    import os
    import datetime as dt

    db, embed, idx, retriever, tmp = env
    docs = [
        ("周报-2026-08-11.md", "# 项目周报\n\n本周完成需求评审，开发进度60%，下周计划联调。\n", "2026-08-11T10:00:00"),
        ("周报-2026-08-04.md", "# 项目周报\n\n上周完成原型，本周进入开发。\n", "2026-08-04T10:00:00"),
        ("技术方案-检索.md", "# 检索技术方案\n\n使用向量检索与重排序提升准确率。\n", "2026-08-10T10:00:00"),
        ("技术方案-数据库.md", "# 数据库方案\n\nSQLite 存储元数据与向量。\n", "2026-08-09T10:00:00"),
        ("会议纪要-评审.md", "# 评审会议\n\n讨论了需求变更与风险。\n", "2026-08-08T10:00:00"),
        ("产品需求文档.md", "# 产品需求\n\n文件操作与语义检索功能需求。\n", "2026-08-07T10:00:00"),
        ("代码-计算.py", 'def calculate_total_price(items):\n    """计算总价。"""\n    return sum(i["price"] for i in items)\n', "2026-08-06T10:00:00"),
        ("杂项-说明.txt", "一些不相关的文本内容。\n", "2026-08-05T10:00:00"),
    ]
    for name, content, mtime_str in docs:
        p = tmp / name
        p.write_text(content, encoding="utf-8")
        ts = dt.datetime.fromisoformat(mtime_str).timestamp()
        os.utime(p, (ts, ts))
        idx.index_file(str(p))
    return [d[0] for d in docs]


# ---------- 契约 ----------
def test_contract_response_shape(env):
    db, embed, idx, retriever, tmp = env
    _seed(env)
    r = retriever.search({"query": "项目周报", "page": 1, "pageSize": 5})
    assert r["code"] == 0
    assert r["message"] == "success"
    data = r["data"]
    assert set(data.keys()) == {"total", "page", "pageSize", "totalPages", "results"}
    assert len(data["results"]) <= 5
    if data["results"]:
        item = data["results"][0]
        fields = {"fileId", "fileName", "filePath", "fileType", "fileSize", "createdAt",
                  "modifiedAt", "author", "confidence"}
        assert fields <= set(item.keys()), f"字段缺失: {fields - set(item.keys())}"


def test_contract_relevant_first(env):
    db, embed, idx, retriever, tmp = env
    r = retriever.search({"query": "计算总价", "page": 1, "pageSize": 3})
    assert r["data"]["results"], "应有结果"
    top = r["data"]["results"][0]
    assert top["fileName"] == "代码-计算.py", f"搜'计算总价'应命中代码文件，实际 {top['fileName']}"


def test_contract_confidence_sorted(env):
    db, embed, idx, retriever, tmp = env
    r = retriever.search({"query": "项目周报", "page": 1, "pageSize": 20})
    confs = [x["confidence"] for x in r["data"]["results"]]
    assert confs == sorted(confs, reverse=True), "confidence 应降序"


def test_contract_filters_filetypes(env):
    db, embed, idx, retriever, tmp = env
    r = retriever.search({"query": "项目", "page": 1, "pageSize": 50, "filters": {"fileTypes": ["md"]}})
    types = {x["fileType"] for x in r["data"]["results"]}
    assert types == {"md"}, f"仅 md: {types}"


def test_contract_filters_daterange(env):
    db, embed, idx, retriever, tmp = env
    # 闭区间：start 到 end（end 当天含）
    r = retriever.search({
        "query": "项目", "page": 1, "pageSize": 50,
        "filters": {"dateRange": {"start": "2026-08-01", "end": "2026-08-05"}},
    })
    names = [x["fileName"] for x in r["data"]["results"]]
    assert "周报-2026-08-04.md" in names, "8-04 应在区间内"
    assert "周报-2026-08-11.md" not in names, "8-11 应在区间外"


def test_contract_filters_datefield(env):
    """D2（2026-08-25）：dateRange.dateField 选择时间戳列（created/accessed/modified）。

    检索层行为验证：直接改写 DB 的 created_at/accessed_at，确认 _filter_files
    按 dateField 选列过滤（防"解析层已做、检索层未接"）。
    """
    db, embed, idx, retriever, tmp = env
    _seed(env)  # 自包含：此前依赖同模块更早的测试替它播种，单独跑必失败
    # 所有文件 created_at 改为 2026-08-20（远离 modified_at 的 8 月上旬区间）
    db.conn.execute("UPDATE files SET created_at='2026-08-20T10:00:00Z'")
    db.conn.execute("UPDATE files SET accessed_at='2026-08-15T10:00:00Z'")
    db.conn.commit()
    # modified 窗口（默认）：8-01~8-05 → 命中周报-2026-08-04
    r = retriever.search({"query": "项目", "page": 1, "pageSize": 50,
                          "filters": {"dateRange": {"start": "2026-08-01", "end": "2026-08-05"}}})
    assert "周报-2026-08-04.md" in [x["fileName"] for x in r["data"]["results"]]
    # created 窗口：8-20 → 命中全部（created_at 已统一改到 8-20）；8-01~8-05 → 0
    r = retriever.search({"query": "项目", "page": 1, "pageSize": 50,
                          "filters": {"dateRange": {"start": "2026-08-20", "end": "2026-08-20", "dateField": "created"}}})
    assert len(r["data"]["results"]) == 8, "created 窗口 8-20 应命中全部 8 个文件"
    r = retriever.search({"query": "项目", "page": 1, "pageSize": 50,
                          "filters": {"dateRange": {"start": "2026-08-01", "end": "2026-08-05", "dateField": "created"}}})
    assert r["data"]["results"] == [], "created 窗口 8-01~8-05 应 0 命中（created_at 已改 8-20）"
    # accessed 窗口：8-15 → 命中全部
    r = retriever.search({"query": "项目", "page": 1, "pageSize": 50,
                          "filters": {"dateRange": {"start": "2026-08-15", "end": "2026-08-15", "dateField": "accessed"}}})
    assert len(r["data"]["results"]) == 8, "accessed 窗口 8-15 应命中全部"


def test_contract_filters_directories_recursive(env):
    db, embed, idx, retriever, tmp = env
    # 子目录文件应被父目录递归命中
    sub = tmp / "sub"
    sub.mkdir(exist_ok=True)
    (sub / "子目录文档.md").write_text("# 子目录\n\n这是子目录里的文件。\n", encoding="utf-8")
    idx.index_file(str(sub / "子目录文档.md"))
    r = retriever.search({"query": "子目录", "page": 1, "pageSize": 50, "filters": {"directories": [str(tmp)]}})
    names = [x["fileName"] for x in r["data"]["results"]]
    assert "子目录文档.md" in names, f"递归前缀应命中子目录: {names}"


def test_contract_filters_dir_case_sensitive(env):
    db, embed, idx, retriever, tmp = env
    # Linux 目录大小写敏感：小写路径不应命中大写
    upper = str(tmp).upper()
    r = retriever.search({"query": "子目录", "page": 1, "pageSize": 50, "filters": {"directories": [upper]}})
    names = [x["fileName"] for x in r["data"]["results"]]
    assert "子目录文档.md" not in names, f"大小写敏感: {names}"


def test_contract_pagination(env):
    db, embed, idx, retriever, tmp = env
    r = retriever.search({"query": "项目", "page": 1, "pageSize": 3})
    data = r["data"]
    assert data["pageSize"] == 3
    assert data["total"] >= 5
    assert data["totalPages"] == -(-data["total"] // 3)
    assert len(data["results"]) == 3
    # 第二页
    r2 = retriever.search({"query": "项目", "page": 2, "pageSize": 3})
    assert r2["data"]["results"], "第二页应有结果"
    # 越界页收敛到最后一页
    r3 = retriever.search({"query": "项目", "page": 999, "pageSize": 3})
    assert r3["data"]["page"] == data["totalPages"]


def test_contract_top_k_caps_returned_count(env):
    """回归 B9：top_k 是「单次检索最多返回条数」的硬上限。

    修复前它只当召回池大小用，最终结果条数不受约束（多路合并 + 文件名路不受池
    约束）——实测把 top_k 设成 3，同一查询仍返回 7 条，与界面提示「单次检索最多
    返回的结果数」和 API 契约「top_k：检索返回条数」都不符。
    """
    db, embed, idx, retriever, tmp = env
    saved_k = retriever.top_k
    try:
        # 2026-09-15：只调 top_k，**不动候选池**——返回条数上限与检索深度是两件事。
        retriever.top_k = 3
        data = retriever.search({"query": "项目", "page": 1, "pageSize": 50})["data"]
        assert data["total"] <= 3, f"total 应被 top_k 截断，实际 {data['total']}"
        assert len(data["results"]) <= 3
        retriever.top_k = 1
        data1 = retriever.search({"query": "项目", "page": 1, "pageSize": 50})["data"]
        assert data1["total"] == 1 and len(data1["results"]) == 1
    finally:
        retriever.top_k = saved_k


def test_ranking_does_not_depend_on_top_k(env):
    """回归：同一个查询、同一个文件，**score 不得随返回条数上限变化**。

    2026-09-15 用户实测：Top-K=10 时第 1 名是 `K3-GPU调试笔记-含图.pdf`，调到 5 后
    第 1 名变成一个无关文件。根因是候选池（检索深度）被接在用户的 top_k 上——块级窗口
    从「前 10 块」缩到「前 5 块」，而那个 PDF 的最佳块排在库里第 8 位（cos 0.3951），
    一出窗口它的向量分就凭空消失，只剩文件名那点分，排名掉到第 6（完整现象见
    docs/troubleshooting.md 现象 H）。

    「显示几条」是呈现偏好，不该改变「哪些文件算命中」以及各自的分数。这条测试把两者
    分开：候选池不动，只改 top_k，两次检索共同出现的文件分数必须逐一相等。
    """
    db, embed, idx, retriever, tmp = env
    saved_k = retriever.top_k
    try:
        retriever.top_k = 3
        small = retriever.search({"query": "项目", "page": 1, "pageSize": 50})["data"]
        retriever.top_k = 20
        big = retriever.search({"query": "项目", "page": 1, "pageSize": 50})["data"]

        assert small["total"] < big["total"], "前置：这条查询的命中数须多于 3，否则测不出差异"
        big_by_id = {r["fileId"]: r for r in big["results"]}
        for r in small["results"]:
            assert r["fileId"] in big_by_id, "top_k 小不该让某条命中整体消失（只该少显示几条）"
            assert r["score"] == big_by_id[r["fileId"]]["score"], \
                f"{r['fileName']} 的 score 随 top_k 变了（{r['score']} vs {big_by_id[r['fileId']]['score']}）" \
                "——说明检索深度仍被 top_k 控制"
        # 前 N 名的顺序也必须一致
        assert [r["fileId"] for r in small["results"]] == \
            [r["fileId"] for r in big["results"][: len(small["results"])]], "前几名顺序随 top_k 变了"
    finally:
        retriever.top_k = saved_k


def test_candidate_pool_zero_means_no_truncation(env):
    """候选池 <=0 = 不截断（与 min_similarity<=0 / max_depth=0 同款约定）。

    负数尤其要归一成大整数：BM25 路拿它当 `[:n]` 切片长度，负值会从尾部反删、悄悄丢结果。
    """
    db, embed, idx, retriever, tmp = env
    saved_pool = retriever.candidate_pool
    try:
        retriever.candidate_pool = 0
        assert retriever._pool_size() >= 10 ** 6, "<=0 应归一为不截断"
        ref = retriever.search({"query": "项目", "page": 1, "pageSize": 50})["data"]
        retriever.candidate_pool = -5
        assert retriever._pool_size() >= 10 ** 6, "负数同样是不截断，不得当成负切片"
        again = retriever.search({"query": "项目", "page": 1, "pageSize": 50})["data"]
        assert [r["fileId"] for r in ref["results"]] == [r["fileId"] for r in again["results"]]
    finally:
        retriever.candidate_pool = saved_pool


def test_contract_top_k_caps_filters_only_search(env):
    """纯筛选检索（净化后空 query）同样受 top_k 约束（两条返回路径都要截断）。"""
    db, embed, idx, retriever, tmp = env
    saved_k = retriever.top_k
    try:
        retriever.top_k = 2
        data = retriever.search({
            "query": "", "page": 1, "pageSize": 50, "filters": {"fileTypes": ["md"]},
        })["data"]
        assert data["total"] <= 2 and len(data["results"]) <= 2
    finally:
        retriever.top_k = saved_k


def test_contract_score_is_absolute_and_descending(env):
    """新增字段 score：未归一化的绝对融合分，随结果降序。

    为什么要有 score：confidence 是 min-max 归一化——每次检索头名恒 1.0、末名恒 0.0，
    只能读相对名次，不能读「匹配度」。上游若需要不撒谎的绝对值（排序、分档、调试），
    用 score。本测试用「按 min-max 公式反算 confidence」把归一化语义钉死。
    """
    db, embed, idx, retriever, tmp = env
    _seed(env)
    results = retriever.search({"query": "项目周报", "page": 1, "pageSize": 50})["data"]["results"]
    assert results, "前置：该查询应有结果"

    scores = [x["score"] for x in results]
    for x in results:
        assert "score" in x, "结果项必须含 score 字段（契约新增）"
        # 余弦可为负（噪声文件），故绝对分理论上可略低于 0——它不是百分比
        assert -1.0 <= x["score"] <= 1.0, f"score 应在 -1~1，实际 {x['score']}"
    assert scores == sorted(scores, reverse=True), "结果应按 score 降序"
    # 绝对分**不该**被打到归一化的两个端点（1.0 / 0.0）——那正是 confidence 的特征。
    # 这条抓「score 退化成 confidence」的回归：归一化值必然 min=0.0、max=1.0。
    assert max(scores) < 1.0 or min(scores) > -1.0, \
        "score 被打到归一化端点，疑似与 confidence 同源（绝对分不该被这样拉扯）"

    # confidence 必须等于对本次结果集做的 min-max 归一化（这就是它「只能读名次」的原因）
    lo, hi = min(scores), max(scores)
    assert hi > lo, "前置：结果集内分数应有差异"
    for x in results:
        assert abs(x["confidence"] - (x["score"] - lo) / (hi - lo)) < 1e-3, \
            "confidence 应等于 (score-min)/(max-min)——它衡量的是名次，不是匹配度"
    assert results[0]["confidence"] == 1.0 and results[-1]["confidence"] == 0.0, \
        "min-max 归一化的后果：头名恒 100%、末名恒 0%，与绝对质量无关"


def test_contract_score_is_stable_while_confidence_moves(env):
    """score 是文件的属性，confidence 是结果集的属性——用同一文件跨结果集对比钉死。

    这是「百分比误导人」的根因所在：同一文件、同一查询，只要结果集里多一条或少一条
    垃圾，它的 confidence 就变（甚至被拉到 0%），而 score 纹丝不动。展示 confidence
    等于展示「本次排第几」，不是「这个文件有多相关」。
    """
    db, embed, idx, retriever, tmp = env
    _seed(env)
    full = {x["fileName"]: x for x in
            retriever.search({"query": "项目周报", "page": 1, "pageSize": 50})["data"]["results"]}
    # 用筛选把结果集缩到只剩一个文件（同一次检索语义，不同结果集）
    only_txt = retriever.search(
        {"query": "项目周报", "page": 1, "pageSize": 50, "filters": {"fileTypes": ["txt"]}}
    )["data"]["results"]
    assert len(only_txt) == 1, "前置：该筛选应只剩一个文件"
    target = only_txt[0]["fileName"]
    assert target in full, "前置：该文件在全集里也应出现"

    assert only_txt[0]["score"] == full[target]["score"], \
        f"score 是绝对分，不应随结果集变化（{full[target]['score']} → {only_txt[0]['score']}）"
    assert only_txt[0]["confidence"] != full[target]["confidence"], \
        "confidence 是相对名次，结果集一变就该变——这正是它不能当「匹配度」展示的原因"


def test_contract_errors(env):
    db, embed, idx, retriever, tmp = env
    # 1001: 空 query
    r = retriever.search({"query": ""})
    assert r["code"] == 1001
    # 1001: 非法 page
    r = retriever.search({"query": "项目", "page": 0})
    assert r["code"] == 1001
    # 1002: 非法 fileType
    r = retriever.search({"query": "项目", "filters": {"fileTypes": ["exe"]}})
    assert r["code"] == 1002
    # 1002: dateRange start > end
    r = retriever.search({"query": "项目", "filters": {"dateRange": {"start": "2026-08-10", "end": "2026-08-01"}}})
    assert r["code"] == 1002


def test_contract_empty_result(env):
    """空结果：筛选排除全部文件时 total=0（确定性场景）。"""
    db, embed, idx, retriever, tmp = env
    r = retriever.search({
        "query": "项目", "page": 1, "pageSize": 10,
        "filters": {"directories": ["/nonexistent_dir"]},
    })
    assert r["code"] == 0
    assert r["data"]["total"] == 0
    assert r["data"]["results"] == []


# ---------- 性能基准 ----------
def test_p95_benchmark():
    """P95 ≤ 100ms（1 万块规模，embed 缓存命中的热路径）。

    构造：独立 db，50 个 fake 文件 × 200 块 = 1 万块随机 512 维向量。
    循环 search 同一 query（首次 embed 后走缓存），测余弦+聚合+筛选+排序延迟。
    """
    import numpy as np

    tmp = Path(tempfile.mkdtemp(prefix="bibit_bench_"))
    db = Database(str(tmp / "bench.db"))
    embed = FakeEmbed()
    retriever = Retriever(db, embed, candidate_pool=50, embed_weight=0.7, filename_weight=0.3,
                          min_similarity=0.0)

    # 50 个 fake 文件
    rng = np.random.default_rng(42)
    for i in range(50):
        fid = f"bench_f{i:02d}"
        mtime = f"2026-08-{1 + i % 28:02d}T10:00:00Z"
        db.conn.execute(
            """INSERT INTO files (file_id, path, name, file_type, size, created_at, modified_at,
                                  accessed_at, author, directory, content_hash, summary, indexed_at, vectorize_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fid, f"/bench/f{i:02d}.md", f"f{i:02d}.md", "md", 100, mtime, mtime, mtime, "", "/bench",
             "h", f"摘要{i}", mtime, "completed"),
        )
        # 200 块/文件，块文本含主题词
        vecs = rng.normal(size=(200, 512)).astype(np.float32)
        db.conn.executemany(
            "INSERT INTO chunks (file_id, chunk_index, text, vector) VALUES (?,?,?,?)",
            [(fid, j, f"文档{i}第{j}块：文件搜索语义检索向量数据库。", _pack_vec(vecs[j].tolist())) for j in range(200)],
        )
    db.conn.commit()

    # 预热：加载向量矩阵 + embed 缓存 + jieba 词典（冷启动 200ms 移出查询路径）
    retriever.warmup()
    retriever.embed.embed_one("项目周报与文件搜索")
    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        retriever.search({"query": "项目周报与文件搜索", "page": 1, "pageSize": 10})
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p95 = lat[int(len(lat) * 0.95)]
    print(f"\n[基准] 1万块热路径 P95 = {p95:.1f}ms（20 次，min {lat[0]:.1f} max {lat[-1]:.1f}）")
    assert p95 <= 100, f"P95 {p95:.1f}ms 超过 100ms 预算"
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
