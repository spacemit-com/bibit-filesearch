"""P12/P13 的回归测试：文件名/路径这条词法通道必须与 BM25 对齐。

改前的两处不一致（2026-09-29 修）：
  · 分词器：BM25 用 `cut_for_search`，这里用 `lcut`（精确模式）
  · 匹配规则：这里子串包含（`w in hay`），BM25 token 相等

于是同一份配置里两个「0.3」语义不同，且短查询能命中、长查询命中不了同一个文件名。
"""
import pytest

from bibit_filesearch.retriever import Retriever


def _mk():
    """只借方法、不建 DB。★ 必须补上 __init__ 会设的缓存字段。"""
    r = Retriever.__new__(Retriever)
    r._name_token_cache = {}
    return r


CANDS = {
    "f1": {"name": "身份证扫描件.pdf", "path": "/home/user/身份证扫描件.pdf"},
    "f2": {"name": "发票.pdf", "path": "/home/user/发票.pdf"},
    "f3": {"name": "notes.md", "path": "/home/user/notes.md"},
}


def test_long_query_still_hits_short_filename_word():
    """核心：查询「居民身份证」必须命中文件名「身份证扫描件.pdf」。

    改前用 `lcut` → words=['居民身份证'] → 子串不在文件名里 → 0 分（漏掉）。
    改后用 `cut_for_search` → 补出子词「身份证」→ 命中。
    """
    s = _mk()._filename_scores("居民身份证", CANDS)
    assert "f1" in s, "★ 长查询应能命中文件名里的短词（这是 P12 修的）"
    assert 0 < s["f1"] <= 1.0


def test_short_query_hits():
    s = _mk()._filename_scores("身份证", CANDS)
    assert "f1" in s and s["f1"] > 0


def test_unrelated_file_scores_absent():
    s = _mk()._filename_scores("身份证", CANDS)
    assert "f3" not in s, "无关文件名不该有分"


def test_score_is_fraction_of_query_tokens():
    """得分 = 命中 token 数 / 查询 token 数（0~1，绝对尺度）。"""
    s = _mk()._filename_scores("发票", CANDS)
    assert s.get("f2") == pytest.approx(1.0), "查询全落在文件名上应为 1.0"


def test_matching_rule_is_token_equality_not_substring():
    """匹配规则是 token 相交，不是任意子串包含。

    '身份证' 作为子串出现在 '居民身份证' 里，但改后要求 **token 相交** ——
    因为两侧都用 cut_for_search，文档侧会补出「身份证」这个子词，所以仍然命中；
    这条测试钉的是"靠子词补齐，不靠子串包含"。
    """
    s = _mk()._filename_scores("身份证", {"f": {"name": "居民身份证.pdf", "path": "居民身份证.pdf"}})
    assert "f" in s, "搜索模式应补出子词「身份证」，与文件名 token 相交"


def test_empty_query_returns_empty():
    assert _mk()._filename_scores("", CANDS) == {}
    assert _mk()._filename_scores("a", CANDS) == {}   # 单字符 token 被丢弃


def test_path_is_searched_too():
    s = _mk()._filename_scores("test", {"f": {"name": "x.md", "path": "/home/user/test/x.md"}})
    assert "f" in s, "路径也应参与匹配"
