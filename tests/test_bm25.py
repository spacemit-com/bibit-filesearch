"""BM25 稀疏检索单元测试。

覆盖：
- tokenize：中文 jieba + 代码/英文 token 保留（wl_surface / dma_fence）
- BM25 得分：含查询词的文件得分高于不含的
- 文件级聚合：多块文件 max 池化
- remove_file 增量维护
"""
import sys
from pathlib import Path


from bibit_filesearch.bm25 import BM25Index, tokenize  # noqa: E402


def test_tokenize_code_symbols():
    toks = tokenize("调用 wl_surface.commit() 完成提交，dma_fence 同步")
    assert "wl_surface" in toks, "代码符号应完整保留"
    assert "commit" in toks
    assert "dma_fence" in toks
    assert "同步" in toks, "中文应被分词"


def test_bm25_relevant_higher():
    idx = BM25Index()
    idx.add_document("f1", "GTK 窗口 mapped 状态切换与 commit 时序")
    idx.add_document("f2", "Python 脚本计算订单总价")
    scores = idx.search("mapped commit")
    assert "f1" in scores, "含查询词的文件应命中"
    assert scores["f1"] > 0
    # f2 不含查询词 → 0 或不在结果
    assert scores.get("f2", 0.0) == 0.0


def test_bm25_file_aggregation():
    """多块文件 max 池化：任一块命中即文件得分 > 0。"""
    idx = BM25Index()
    idx.add_document("f1", "第一部分 无查询词")
    idx.add_document("f1", "第二部分 含 dma_fence 关键内容")
    idx.add_document("f2", "无关内容")
    scores = idx.search("dma_fence")
    assert "f1" in scores and scores["f1"] > 0
    assert "f2" not in scores


def test_bm25_remove_file():
    idx = BM25Index()
    idx.add_document("f1", "GTK mapped 状态")
    idx.add_document("f2", "mutter 窗口管理")
    assert "f1" in idx.search("mapped")
    idx.remove_file("f1")
    scores = idx.search("mapped")
    assert "f1" not in scores
    # f2 索引应保留
    assert "f2" in idx.search("mutter")
