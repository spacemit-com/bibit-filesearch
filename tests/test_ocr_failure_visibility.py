"""OCR 失败可见性（rev 5，2026-09-28）。

守的是「引擎唯一化」之后最需要的一条：**OCR 故障不能被显示成「这个文件没有字」**。

旧行为（双引擎时代）：OCR 挂了一律落 unsupported，前端显示「未纳入索引 · 无文本」，
用户无法区分「OCR 坏了」与「这张图本来就没字」。而且 tesseract 回退还会在用户
不知情的前提下用另一个引擎读出（可能不准的）文本，把故障盖住。
现在：有 error → failed（可查、可重试、前端显示失败）；没 error → 才是 no_text。
"""
import os
import sys

import pytest

sys.path.insert(0, "src")

from bibit_filesearch.indexer import Indexer  # noqa: E402


class _Db:
    """最小库桩：只记 upsert 的 status（本测试只关心落成 failed 还是 unsupported）。"""

    def __init__(self):
        self.rows = {}

    def get_file_by_path(self, path):
        return self.rows.get(path)

    def upsert_file(self, meta, content_hash, text, status="", pdf_images="[]"):
        self.rows[meta["path"]] = {
            "file_id": meta["path"], "path": meta["path"],
            "content_hash": content_hash, "vectorize_status": status,
        }
        return meta["path"]

    def mark_attempt_failed(self, file_id, content_hash):
        self.attempts = getattr(self, "attempts", {})
        self.attempts[file_id] = self.attempts.get(file_id, 0) + 1

    def set_vectorize_status(self, file_id, status):
        if file_id in self.rows:
            self.rows[file_id]["vectorize_status"] = status


class _Parsed:
    def __init__(self, text="", error=""):
        self.path = ""
        self.file_type = "png"
        self.text = text
        self.author = ""
        self.error = error
        self.is_image = True
        self.images = []
        self.headings = []
        self.code_symbols = []
        self.raw_meta = {}


def _indexer(monkeypatch, parsed):
    import threading

    import bibit_filesearch.indexer as ix

    db = _Db()
    idx = Indexer.__new__(Indexer)          # 绕过 __init__ 里的重依赖
    idx.db = db
    idx.indexable_types = {"png"}
    idx.chunk_size = 200
    idx.chunk_overlap = 20
    idx.max_chunks_per_file = 64
    idx.embed = None
    idx.duplicate_wait_seconds = 0.1
    # 单飞记账（index_file 入口要用；真构造在 __init__ 里，这里手工补）
    idx._inflight = 0
    idx._inflight_lock = threading.Lock()
    idx._active_paths = {}
    idx._content_hash = lambda p: "H"
    # 2026-09-29：indexer 现在带 min_chars= 调 parse_file（PDF 预算早停），桩收下该 kwarg。
    monkeypatch.setattr(ix, "parse_file", lambda p, **kw: parsed)
    monkeypatch.setattr(ix, "split_by_type", lambda text, ftype, cs, co: [text] if text else [])
    # fileType 按扩展名推，测试才可控（indexer 用它做「是否可索引」判断）
    monkeypatch.setattr(ix, "extract_metadata",
                        lambda p, ftype=None: {"path": p, "name": os.path.basename(p),
                                               "fileType": ftype or str(p).rsplit(".", 1)[-1]})
    return idx, db


def test_ocr_failure_becomes_failed_not_unsupported(tmp_path, monkeypatch):
    """OCR 抛异常 → 文件落 failed，且错误原因可查。"""
    f = tmp_path / "broken.png"
    f.write_bytes(b"x")
    # 2026-09-28（test3）：真实 parser 现在给结构化标记（reason 走前端 locales 出词）
    parsed = _Parsed(
        text="",
        error=(
            "ocrFailed:"
            + '{"kind":"ocrFailed","errorType":"RuntimeError",'
              '"detail":"PP-OCR 不可用：模型加载失败"}'
        ),
    )
    idx, db = _indexer(monkeypatch, parsed)

    res = idx.index_file(str(f))
    assert res["status"] == "failed", "OCR 故障必须落 failed: %r" % res
    assert "PP-OCR" in res["error"]
    assert db.rows[str(f)]["vectorize_status"] == "failed"


def test_empty_image_still_no_text(tmp_path, monkeypatch):
    """真正没字的图（引擎跑成功、结果为空）仍走 no_text —— 不要误报成故障。"""
    f = tmp_path / "blank.png"
    f.write_bytes(b"x")
    parsed = _Parsed(text="", error="")     # 引擎成功但没识别出文字
    idx, db = _indexer(monkeypatch, parsed)

    res = idx.index_file(str(f))
    assert res["status"] == "no_text", "空文本不该被当成 OCR 故障: %r" % res
    assert db.rows[str(f)]["vectorize_status"] == "unsupported"


def test_text_file_error_is_not_routed_as_ocr_failure(tmp_path, monkeypatch):
    """文本类文件的解析错误同样可见（不是只能用于图片）——同一判据、同一出口。"""
    f = tmp_path / "notes.md"
    f.write_bytes(b"x")
    parsed = _Parsed(text="", error="boom")
    parsed.file_type = "md"
    idx, db = _indexer(monkeypatch, parsed)
    idx.indexable_types = {"md"}

    res = idx.index_file(str(f))
    assert res["status"] == "failed"
