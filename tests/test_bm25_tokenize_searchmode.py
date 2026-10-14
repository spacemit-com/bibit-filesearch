"""BM25 中文分词的回归测试（2026-09-29 改用 jieba 搜索模式 cut_for_search）。

守的是一个已实测的召回洞：精确模式对同一串字在文档侧与查询侧切出不同长度的词，
而 BM25 要求 token 精确相等。

    文档「…居民身份证…」 → ['居民身份证']
    查询「身份」         → ['身份']            ← 不等 → 不匹配 → 召回失败

真机后果：id_card_2.jpg 拿不到 BM25 命中豁免，又被门槛（cos < 0.35）挡住，
整份文件从结果里消失；同批 id_card_1 / id_rotate 文本写的是「公民身份号码」，
精确模式恰好切出「身份」所以命中了。

修法用 `cut_for_search`（长词补出词典认可的子词）而非"补相邻双字"：
两者都能补上召回，但双字会产出跨词边界的噪声 token（'民身'、'份号'、'国居'…），
既污染 tf 语义又把索引撑大（实测 avg_len 1369→2576）。搜索模式产出全是真词/真子词。
"""
import pytest

from bibit_filesearch.bm25 import BM25Index, tokenize, _dedup


# ─────────────── tokenize 层 ───────────────

def test_search_mode_fills_segmentation_gap():
    """核心用例：文档切出「居民身份证」时，查询「身份」必须能匹配上。

    实测 jieba 两种模式的切法（别凭直觉假设）：
        精确模式 '居民身份证' → ['居民身份证']
        搜索模式 '居民身份证' → ['居民身份证','居民','身份','身份证']
    关键：靠搜索模式补出的子词「身份」，两侧才对得上。
    """
    doc_tokens = set(tokenize("居民身份证"))
    assert "居民身份证" in doc_tokens, "整词仍在（补子词不该丢整词）"
    assert "身份" in doc_tokens, "★ 搜索模式未补出子词「身份」→ 查询「身份」无法召回"
    assert "身份证" in doc_tokens


def test_no_cross_boundary_noise_tokens():
    """搜索模式**不能**产出跨词边界的噪声 token（这是它优于双字兜底的地方）。

    '民身'（"居民|身份"的跨界残片）、'份号'（"身份|号码"）在语言上不存在。
    """
    for text in ("中华人民共和国居民身份证", "公民身份号码", "居民身份证"):
        toks = set(tokenize(text))
        for noise in ("民身", "份号", "国居", "民共", "和国"):
            assert noise not in toks, "搜索模式产出了跨界噪声 token %r：%r" % (noise, sorted(toks))


def test_query_and_doc_share_token_across_segmentation():
    """查询与文档的 token 集合必须有交集（召回的必要条件）。"""
    q = set(tokenize("身份"))
    for doc in ("公民身份号码 430523197603204314",
                "居民身份证 签发机关 宜昌市公安局",
                "身份证信息 您的身份证",
                "中华人民共和国 居民身份证"):
        assert q & set(tokenize(doc)), "查询「身份」应至少一个 token 与该文档重合：%r" % doc


def test_english_tokens_unchanged():
    """英文/代码 token 的行为不能被这次改动影响。"""
    toks = tokenize("wl_surface dma_fence gsettings")
    assert "wl_surface" in toks and "dma_fence" in toks and "gsettings" in toks


def test_single_char_tokens_still_dropped():
    """单字噪声仍要丢掉（搜索模式补的是 >=2 字的子词）。"""
    toks = tokenize("我 的 书")
    assert not [t for t in toks if len(t) == 1], "单字符 token 应被丢弃：%r" % toks


# ─────────────── 查询侧去重 ───────────────

def test_query_dedup_removes_duplicate_tokens():
    """查询侧去重：搜索模式让一个词以多粒度出现，同一概念不能算多次。

    不去重的话 `search()` 会逐 token 累加倒排贡献 → 该查询分数虚高。
    """
    raw = tokenize("身份证")
    assert len(raw) >= 2, "搜索模式应产出多个粒度：%r" % raw
    deduped = _dedup(raw)
    assert len(deduped) == len(set(deduped))
    # 整词与其子词都在（这是搜索模式的设计，不是 bug）
    assert "身份证" in deduped and "身份" in deduped


def test_document_side_tf_not_deduped():
    """文档侧**不能**去重：tf 是 BM25 的核心分量。"""
    toks = tokenize("身份 身份 身份")
    tf = sum(1 for t in toks if t == "身份")
    assert tf >= 2, "文档侧 tf 被去重抹平了（应保留多次出现）"


# ─────────────── 端到端 ───────────────

def test_end_to_end_recall_across_segmentation():
    """端到端：文档写「居民身份证」时，查询「身份」必须召回该文件。

    这条对应真机上 id_card_2.jpg 消失的那个 BUG。
    """
    bm = BM25Index()
    bm.add_document("doc_a", "蒋飞成 公民身份号码 430523197603204314")   # 精确模式也能召回
    bm.add_document("doc_b", "居民身份证 签发机关 宜昌市公安局")           # 精确模式召回不到
    bm.add_document("doc_c", "PP-OCR 在 RISC-V 上的部署与使用")           # 无关

    hits = bm.search("身份", top_k=100)
    assert "doc_a" in hits, "「公民身份号码」应命中"
    assert "doc_b" in hits, "★「居民身份证」也必须命中（搜索模式修的就是这个）"
    assert "doc_c" not in hits, "无关文档不应命中"


def test_end_to_end_search_mode_hits_more_granularity():
    """搜「身份证」时应同时命中写了「身份证」与只写了「身份号码」的文档。"""
    bm = BM25Index()
    bm.add_document("with_word", "您的身份证 有效期 2026")       # 含整词「身份证」
    bm.add_document("with_sub", "公民身份号码 430523197603204314")  # 只有「身份」
    hits = bm.search("身份证", top_k=100)
    assert "with_word" in hits
    assert "with_sub" in hits, "★ 搜索模式应让「身份证」这个查询也能命中只写「身份号码」的文档"
