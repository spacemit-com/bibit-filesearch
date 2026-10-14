"""解析版本水位线测试（rev 5 从 test_ocr_orientation.py 迁出）。

这些用例守的是「换了识别器 / 修了解析能力之后，**已入库**的文件必须重排一次」——
不重排的话 content_hash 幂等会让老文件永远停在旧文本上，「修了等于没修」。
它们与方向判断无关，所以 tesseract 删掉后仍然有效，原样保留。

★ 迁出时保留了全部 5 个：3 个名字是 test_reconcile_parser_revision_*，
  另 2 个（test_requeue_ocr_text_*、test_reconcile_without_image_types_*）
  名字不带前缀但同属这条链路，一起带走（否则会被误删）。
"""
import sys

sys.path.insert(0, "src")

from bibit_filesearch import parser as P  # noqa: E402

# 背景：光改解析代码对**已入库**的文件无效——content_hash 是文件字节的 hash，
# OCR 结果变了它看不出来 → 重扫判 skipped → 老文本永远留在库里。所以配
# `db.requeue_ocr_text` + `settings.reconcile_parser_revision` 的水位线。

from bibit_filesearch.db import Database          # noqa: E402
from bibit_filesearch.settings import reconcile_parser_revision  # noqa: E402


class _FakeState:
    def __init__(self, **kw):
        self.d = dict(kw)

    def get(self, k, default=None):
        return self.d.get(k, default)

    def set(self, k, v):
        self.d[k] = v


def _row(db, path, ftype, status="completed", pdf_images="[]", hash_="HASH"):
    db.conn.execute(
        "INSERT INTO files (file_id, path, name, file_type, content_hash,"
        " vectorize_status, pdf_images) VALUES (?,?,?,?,?,?,?)",
        (path, path, path.rsplit("/", 1)[-1], ftype, hash_, status, pdf_images))
    db.conn.commit()


def test_requeue_ocr_text_picks_images_and_image_pdfs_only(tmp_path):
    """★ 判据：图片一律要重排；PDF 只在 `pdf_images != '[]'`（= 图片型）时重排。

    有文本层的 PDF 文本来自 pymupdf、与 OCR 无关，必须不动 —— 否则每次升级都白重扫。
    """
    db = Database(str(tmp_path / "c.db"))
    _row(db, "/kb/a.jpg", "jpg")
    _row(db, "/kb/b.png", "png")
    _row(db, "/kb/scan.pdf", "pdf", pdf_images='[{"page_idx": 0, "ocr_text": "x"}]')
    _row(db, "/kb/text.pdf", "pdf", pdf_images="[]")          # 有文本层 → 不动
    _row(db, "/kb/notes.md", "md")                            # 非 OCR 产物 → 不动
    _row(db, "/kb/skip.txt", "txt", status="unsupported")     # 已 unsupported → 不动

    n = db.requeue_ocr_text(["jpg", "png"])
    assert n == 3, "应重排 a.jpg / b.png / scan.pdf 共 3 个"
    assert db.get_file_by_path("/kb/a.jpg")["vectorize_status"] == "queued"
    assert db.get_file_by_path("/kb/a.jpg")["content_hash"] == "", "必须清 hash，否则幂等会 skip"
    assert db.get_file_by_path("/kb/scan.pdf")["vectorize_status"] == "queued"
    assert db.get_file_by_path("/kb/text.pdf")["vectorize_status"] == "completed"
    assert db.get_file_by_path("/kb/notes.md")["vectorize_status"] == "completed"


def test_reconcile_parser_revision_requeues_both_sides(tmp_path):
    """★ 水位线一次跑两边：unsupported 的（旧能力读不到）+ 有 OCR 文本的（旧 OCR 结果）。"""
    db = Database(str(tmp_path / "c.db"))
    _row(db, "/kb/box.docx", "docx", status="unsupported")   # 当时读不到 → 走 unsupported 那条
    _row(db, "/kb/a.jpg", "jpg")                             # 当时读到了旧 OCR 文本 → 走新那条
    state = _FakeState()

    n = reconcile_parser_revision(db, state, ["docx", "jpg"],
                                  image_types=["jpg"])
    assert n == 2, "两条影响面都要被覆盖: %d" % n
    assert db.get_file_by_path("/kb/box.docx")["vectorize_status"] == "queued"
    assert db.get_file_by_path("/kb/a.jpg")["vectorize_status"] == "queued"
    assert state.get("parser_revision") == P.PARSER_REVISION, "水位线要置位"


def test_reconcile_parser_revision_is_idempotent(tmp_path):
    """跑过一次就不再重排（否则每轮启动都白重解析一遍）。"""
    db = Database(str(tmp_path / "c.db"))
    _row(db, "/kb/a.jpg", "jpg")
    state = _FakeState()

    assert reconcile_parser_revision(db, state, ["jpg"], image_types=["jpg"]) == 1
    db.conn.execute("UPDATE files SET vectorize_status='completed', content_hash='H'")
    db.conn.commit()
    assert reconcile_parser_revision(db, state, ["jpg"], image_types=["jpg"]) == 0


def test_reconcile_parser_revision_skips_when_revision_already_seen(tmp_path):
    """已记录过本版本 → 一行都不碰（升级前的老库首次跑才会有动作）。"""
    db = Database(str(tmp_path / "c.db"))
    _row(db, "/kb/a.jpg", "jpg")
    state = _FakeState(parser_revision=P.PARSER_REVISION)
    assert reconcile_parser_revision(db, state, ["jpg"], image_types=["jpg"]) == 0
    assert db.get_file_by_path("/kb/a.jpg")["vectorize_status"] == "completed"


def test_reconcile_without_image_types_still_does_unsupported_pass(tmp_path):
    """不传 image_types（老调用方/测试）→ 只做 unsupported 那条，行为向后兼容。"""
    db = Database(str(tmp_path / "c.db"))
    _row(db, "/kb/box.docx", "docx", status="unsupported")
    _row(db, "/kb/a.jpg", "jpg")
    n = reconcile_parser_revision(db, _FakeState(), ["docx", "jpg"])
    assert n == 1
    assert db.get_file_by_path("/kb/a.jpg")["vectorize_status"] == "completed"


def test_reconcile_requeues_all_docx_including_completed(tmp_path):
    """★ rev 9：有正文的 docx（completed）也要重排——影响面是「所有 docx」。

    这一条是 rev 9 的关键取舍：docx 加了内嵌图 OCR 后，**正文 + 段内插图**的 docx
    文本会变多，而它们手里是 completed（不是 unsupported），`requeue_unsupported`
    一个都命中不到。所以必须直写「所有 docx」，否则这类文件永远留着旧文本、
    新 OCR 代码一次都跑不到（content_hash 是文件字节的 hash，看不出 OCR 结果）。
    """
    db = Database(str(tmp_path / "c.db"))
    _row(db, "/kb/body.docx", "docx", status="completed")   # 正文 + 插图 → completed
    _row(db, "/kb/note.md", "md", status="completed")       # 非 docx → 不受影响
    n = reconcile_parser_revision(db, _FakeState(), ["docx", "md"])
    assert db.get_file_by_path("/kb/body.docx")["vectorize_status"] == "queued"
    assert db.get_file_by_path("/kb/body.docx")["content_hash"] == "", \
        "必须清 hash，否则幂等会判定 skipped"
    assert db.get_file_by_path("/kb/note.md")["vectorize_status"] == "completed"
    assert n == 1, "只有那个 docx 该被重排；实际 %d" % n


def test_reconcile_counts_each_file_once(tmp_path):
    """计数不虚高：一个 unsupported 的 docx 只能被数一次。

    回归（2026-10-09 实踩）：`requeue_all_docx` 与 `requeue_unsupported` 都能命中同一个
    unsupported docx。若 unsupported 那条先跑（把它改成 queued），后面的 requeue_all_docx
    会再数一遍 → 计数 2，而实际只重排了 1 个文件。修法是把 unsupported 放最后。
    """
    db = Database(str(tmp_path / "c.db"))
    _row(db, "/kb/only.docx", "docx", status="unsupported")
    n = reconcile_parser_revision(db, _FakeState(), ["docx"])
    assert n == 1, "同一个文件只能计一次；实际 %d" % n
    assert db.get_file_by_path("/kb/only.docx")["vectorize_status"] == "queued"
