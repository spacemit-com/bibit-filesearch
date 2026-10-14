"""图片型 PDF 的 OCR 文本被 parser 与 indexer 双写 → 同一页入库两份块（2026-09-24 修复）。

## 现象

`bill.pdf`（单页、无文本层、内嵌一张 387×516 的发票截图）在库里是 2 块：

    块0  [第1页图片] 通用机打发时 HAW 2350113022 ...
    块1  [图1页]     通用机打发时 HAW 2350113022 ...

字符级相似度 0.994 —— 同一份 OCR 文本，只是前缀不同。

## 根因（两处，都以为自己是唯一的写入者）

  · `parser._parse_pdf`：图片型 PDF 分支把 OCR 文本拼进 `doc.text`
    （前缀 `[第N页图片]`）—— **必须保留**，那是可检索内容的唯一来源。
  · `indexer.index_file`：又把 `parsed.images` 里的同一份 OCR 文本
    `chunks_text.append(...)` 一次（前缀 `[图N页]`）—— **本次删掉的就是它**。

indexer 那句来自初版（`parser` 那时 `images` 恒空、`_parse_pdf` 返回 `(text, author, [])`），
当时它是「把 OCR 送进向量化内容」的唯一手段；2026-09-07 接入 OCR 后假设失效，语句留着。

## 为什么不能只删代码不改存量

`content_hash` 是**文件字节**的 hash —— 删掉重复块不体现在里面。老库里已经双写的文件
重扫会判「内容未变 → skipped」，重复块永远活着。故配 `CHUNK_REVISION` 水位线 2→3 +
`db.requeue_image_ocr_dupes()`（判据 = indexer 侧独有的 `[图N页]` 前缀）。

与 0.1.9（purge_vectors 没清 hash）、0.1.10（requeue_unsupported）、rev 1/2 是同一类陷阱。

## 为什么不打真 PDF / 不跑真 PP-OCR（2026-09-24 定）

本文件原先用 pymupdf 造真 PDF、真跑 PP-OCR。它自己跑峰值只 27 个 fd，但**放进全量
套件后**把套件峰值从 795 推过 1024（进程 fd 上限）——排在后面的 `test_t3_parser` 真 OCR
用例拿不到 fd（`OCR 失败 ...: [Errno 24] Too many open files`）而失败，之后连 pytest 写
自己的 tmp 目录都失败、整个会话作废。抬 `ulimit -n 8192` 可全过，但那要每个跑套件的人
都知道；所以这里改成 monkeypatch `indexer.parse_file`，直接喂构造好的 `ParsedDocument`。

这样改**不降低**这条测试的效力：被测的正是 `index_file` 对「parser 已把 OCR 放进 text
**且** images 非空」这一输入的处理，而 monkeypatch 喂的正是这个输入 —— 改前会出 2 块、
改后出 1 块，判别力与原版完全一致，只是不花 fd。真 OCR 端到端仍由 `test_t3_parser` 守。
"""
import pytest

from fake_embed import FakeEmbed  # noqa: E402
from bibit_filesearch import indexer as indexer_mod  # noqa: E402
from bibit_filesearch.db import Database  # noqa: E402
from bibit_filesearch.indexer import CHUNK_REVISION, Indexer  # noqa: E402
from bibit_filesearch.parser import ParsedDocument  # noqa: E402
from bibit_filesearch.settings import reconcile_chunk_revision  # noqa: E402

INDEXER_PREFIX = "[图"          # indexer 侧独有前缀（重复块）
PARSER_PREFIX = "[第"           # parser 侧前缀（正本，必须留）

OCR_TEXT = "DupeProbe OCR 2350113022"


class _FakeState:
    def __init__(self):
        self.d = {}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def set(self, k, v):
        self.d[k] = v


def _db(tmp_path) -> Database:
    return Database(str(tmp_path / "c.db"))


def _fake_indexer(db, tmp_path, monkeypatch, *, ocr=OCR_TEXT, images=True,
                  text_has_ocr=True, name="image_only.pdf"):
    """把一个「图片型 PDF」喂给 indexer，但不真解析（见模块 docstring 末节）。

    关键：parser 侧把 OCR 拼进 text（text_has_ocr）+ images 非空（images）——
    这正是 2026-09-07 之后真实的输入形态，也是双写的充分条件。
    """
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 dummy image-only pdf, never parsed\n")

    body = f"{PARSER_PREFIX}1页图片] {ocr}" if text_has_ocr else ""
    parsed = ParsedDocument(
        path=str(p),
        file_type="pdf",
        text=body,
        author="",
        is_image=False,
        images=[{"page_idx": 0, "ocr_text": ocr}] if images else [],
    )
    # 2026-09-29：indexer 现在带 min_chars= 调 parse_file（PDF 预算早停），
    # 桩要收下这个 kwarg（**kwargs 即可，本用例不关心它的值）。
    monkeypatch.setattr(indexer_mod, "parse_file", lambda _path, **kw: parsed)
    return str(p)


# ---------- 一、写入侧：不再双写 ----------

def test_image_pdf_indexes_each_page_once(tmp_path, monkeypatch):
    """★ 核心：parser 已给 OCR 文本 + images 非空时，只出一块（不再有 `[图N页]` 块）。"""
    db = _db(tmp_path)
    idx = Indexer(db, FakeEmbed(), indexable_types={"pdf"})
    r = idx.index_file(_fake_indexer(db, tmp_path, monkeypatch))

    chunks = db.get_chunks(r["file_id"])
    texts = [c["text"] for c in chunks]
    assert len(texts) == 1, f"同一页 OCR 只该有一块: {texts!r}"
    assert OCR_TEXT in texts[0]
    assert not any(t.lstrip().startswith(INDEXER_PREFIX) for t in texts), \
        f"不应再有 indexer 侧前缀的重复块: {texts!r}"


def test_dupe_would_appear_without_the_fix(tmp_path, monkeypatch):
    """★ 判别力自证：把历史上那句 append 放回去，同一输入必须出 2 块。

    这条守的是本条测试的**效力**——防止将来有人把断言改松到「改前改后都能过」。
    """
    db = _db(tmp_path)
    path = _fake_indexer(db, tmp_path, monkeypatch)

    real_index_file = Indexer.index_file

    def index_file_with_legacy_append(self, p, *a, **kw):
        orig = indexer_mod.split_by_type

        def split_with_legacy_dupe(text, ftype, *args, **kwargs):
            out = orig(text, ftype, *args, **kwargs)
            # 复刻修复前 indexer.py 的行为：把 images 里的 OCR 再 append 一块
            out = list(out) + [f"{INDEXER_PREFIX}1页] {OCR_TEXT}"]
            return out

        indexer_mod.split_by_type = split_with_legacy_dupe
        try:
            return real_index_file(self, p, *a, **kw)
        finally:
            indexer_mod.split_by_type = orig

    monkeypatch.setattr(Indexer, "index_file", index_file_with_legacy_append)
    r = Indexer(db, FakeEmbed(), indexable_types={"pdf"}).index_file(path)
    texts = [c["text"] for c in db.get_chunks(r["file_id"])]
    assert len(texts) == 2, f"复刻旧行为时应当出重复块: {texts!r}"


def test_image_pdf_keeps_provenance(tmp_path, monkeypatch):
    """修完仍要保住图生文溯源：files.pdf_images 照常落 OCR 文本。"""
    db = _db(tmp_path)
    idx = Indexer(db, FakeEmbed(), indexable_types={"pdf"})
    r = idx.index_file(_fake_indexer(db, tmp_path, monkeypatch))

    row = db.get_file(r["file_id"])
    assert row["pdf_images"] != "[]"
    assert OCR_TEXT in row["pdf_images"], "溯源用的 OCR 文本不该被一起删掉"


# ---------- 二、判据：只挑重复块所在文件 ----------

def _dupe_row(db: Database, path: str = "/kb/legacy.pdf", *, with_dupe: bool = True,
              status: str = "completed") -> str:
    """手工造一行「已被旧代码双写」的库记录（直插 SQL，模仿老库形态）。

    `with_dupe=False` → 只有 parser 侧那块（正本），用于验证判据不误伤。
    """
    db.conn.execute(
        "INSERT INTO files (file_id, path, name, file_type, content_hash, vectorize_status)"
        " VALUES (?,?,?,'pdf','HASHED',?)",
        (path, path, path.rsplit("/", 1)[-1], status))
    fid = db.conn.execute("SELECT file_id FROM files WHERE path=?", (path,)).fetchone()[0]
    rows = [(0, f"{PARSER_PREFIX}1页图片] {OCR_TEXT}")]
    if with_dupe:
        rows.append((1, f"{INDEXER_PREFIX}1页] {OCR_TEXT}"))
    for i, text in rows:
        db.conn.execute(
            "INSERT INTO chunks (file_id, chunk_index, text, vector) VALUES (?,?,?,?)",
            (fid, i, text, b"\x00\x01"))
    db.conn.commit()
    return fid


def test_requeue_picks_image_ocr_dupes(tmp_path):
    """★ 核心：老库里带 `[图N页]` 前缀块的文件被挑出，且清 hash（否则幂等会 skip）。"""
    db = _db(tmp_path)
    _dupe_row(db)

    assert db.requeue_image_ocr_dupes() == 1
    row = db.get_file_by_path("/kb/legacy.pdf")
    assert row["vectorize_status"] == "queued"
    assert row["content_hash"] == ""


def test_requeue_ignores_files_without_dupes(tmp_path, monkeypatch):
    """★ 不误伤：修好后的图片型 PDF 与普通 md 都不被重排。"""
    db = _db(tmp_path)
    idx = Indexer(db, FakeEmbed(), indexable_types={"pdf", "md"})

    md = tmp_path / "plain.md"
    md.write_text("# 标题\n\n正文内容。\n", encoding="utf-8")
    idx.index_file(str(md))

    pdf_path = _fake_indexer(db, tmp_path, monkeypatch)
    idx.index_file(pdf_path)

    assert db.requeue_image_ocr_dupes() == 0
    assert db.get_file_by_path(str(md))["vectorize_status"] == "completed"
    assert db.get_file_by_path(pdf_path)["vectorize_status"] == "completed"


def test_requeue_only_matches_indexer_prefix(tmp_path):
    """判据的边界：`[第N页图片]`（parser 正本）不触发重排 —— 它是我们要保留的那份。"""
    db = _db(tmp_path)
    _dupe_row(db, "/kb/only-parser.pdf", with_dupe=False)

    assert db.requeue_image_ocr_dupes() == 0
    assert db.get_file_by_path("/kb/only-parser.pdf")["vectorize_status"] == "completed"


# ---------- 三、水位线 ----------

def test_reconcile_rev3_runs_once(tmp_path):
    """水位线 2→3：跑一次把重复块文件重排，再跑零代价。"""
    db = _db(tmp_path)
    _dupe_row(db)
    state = _FakeState()
    state.set("chunk_revision", 2)          # 上一版水位线

    assert reconcile_chunk_revision(db, state) == 1
    assert state.get("chunk_revision") == CHUNK_REVISION == 3
    assert reconcile_chunk_revision(db, state) == 0


# ---------- 四、端到端：老库态 → 重排 → 只剩一份 ----------

def test_end_to_end_reindex_removes_dupe(tmp_path, monkeypatch):
    """★ 端到端：老库的双写产物，经水位线重排 + 重索引后，只剩 parser 那一份。"""
    db = _db(tmp_path)
    path = _fake_indexer(db, tmp_path, monkeypatch)
    idx = Indexer(db, FakeEmbed(), indexable_types={"pdf"})
    fid = idx.index_file(path)["file_id"]

    # 手工把库改回「旧代码双写」的状态（模拟升级前入库），并保持 content_hash 不变
    db.replace_chunks(fid, [
        (0, f"{PARSER_PREFIX}1页图片] {OCR_TEXT}", None),
        (1, f"{INDEXER_PREFIX}1页] {OCR_TEXT}", None),
    ])
    db.conn.execute("UPDATE files SET total_chunks=2 WHERE file_id=?", (fid,))
    db.conn.commit()

    # 不改内容直接重索引 → 幂等 skip（这正是「只改代码不改存量」失败的形态）
    assert Indexer(db, FakeEmbed(), indexable_types={"pdf"}).index_file(path)["status"] == "skipped"

    # 水位线触发重排后才真正重切
    assert reconcile_chunk_revision(db, _FakeState()) == 1
    r2 = Indexer(db, FakeEmbed(), indexable_types={"pdf"}).index_file(path)
    assert r2["status"] in ("indexed", "updated")
    texts = [c["text"] for c in db.get_chunks(fid)]
    assert not any(t.lstrip().startswith(INDEXER_PREFIX) for t in texts), \
        f"重索引后不应再有重复块: {texts!r}"
    assert OCR_TEXT in "\n".join(texts), "OCR 内容本身不能丢"
