"""截断（内容不完整）与错误明细分离 + 失败分类修正（2026-09-21）。

用户实测暴露的三个叠在一起的问题：
  ① 界面「失败文件 1」点开却列了 12 条——计数取 vectorizePhase.failed（真失败），
     列表取 scanPhase.errorDetails（含截断），两者数据源不同；
  ② 11 条「块超过 64」被列在「失败文件」下，但那些文件其实**索引成功、可检索**，
     只是内容不全（indexer 刻意不用独立状态，免得整份搜不到）；
  ③ 剩下那条被标成「错误解析」，实际是 embed 过载超时——worker 把除
     PermissionError 外的一切异常都归成 "parse"，名不符实。

改法：
  · 截断移出 errorDetails → vectorizePhase.truncated / truncationDetails，
    且改按 DB 求值（持久，跨扫描轮次可见）；
  · 服务侧失败抛 EmbedError → 单列 kind="embed"。
"""
import tempfile
import time
from pathlib import Path

import pytest

from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer
from bibit_filesearch.model_client import EmbedError
from bibit_filesearch.retriever import Retriever
from bibit_filesearch.scanner import ScanService


class _Cfg:
    def __init__(self, dirs, **scan_cfg):
        self.index_dirs = dirs
        self.exclude_patterns = [".git"]

        class _Scan:
            def __init__(self, d):
                self._d = {"io_throttle_ms": 0, **d}

            def get(self, k, default=None):
                return self._d.get(k, default)

        self.scan = _Scan(scan_cfg)


class _Embed:
    """可控 embed：fail_with 非空时按该异常失败。"""

    def __init__(self, fail_with=None):
        self.dim = 8
        self.fail_with = fail_with

    def _vec(self, text):
        return [float((hash((text, i)) % 97) + 1) for i in range(self.dim)]

    def embed_one(self, text):
        if self.fail_with is not None:
            raise self.fail_with
        return self._vec(text)

    def embed_batch(self, texts):
        if self.fail_with is not None:
            raise self.fail_with
        return [self._vec(t) for t in texts]

    def ping(self):
        return self.fail_with is None

    def close(self):
        pass


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_trunc_"))
    db = Database(str(tmp / "t.db"))
    yield tmp, db
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _wait_done(scanner, timeout=30):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if scanner.status == "done":
            return True
        time.sleep(0.1)
    return scanner.status == "done"


def _mk(tmp, db, embed, **scan_cfg):
    work = tmp / "work"
    work.mkdir(exist_ok=True)
    indexer = Indexer(db, embed, chunk_size=10, chunk_overlap=0, **scan_cfg)
    retriever = Retriever(db, embed)
    indexer.set_on_indexed(retriever.on_file_indexed)
    indexer.set_on_deleted(retriever.on_file_deleted)
    return work, ScanService(_Cfg([str(work)], **scan_cfg), db, indexer)


def _mk2(tmp, db, embed, scanner):
    """已建好的 scanner 复用：返回索引器（便于直接调 index_file）。"""
    return Indexer(db, embed, chunk_size=10, chunk_overlap=0, max_chunks_per_file=3)


# ---------------------------------------------------------------- A：db.truncated_files
def test_truncated_files_lists_only_incomplete(env):
    """只列「已知总块数且已存更少」的文件；完整的、total 未知的都不列。"""
    tmp, db = env
    work = tmp / "work"
    work.mkdir()
    emb = _Embed()
    idx = Indexer(db, emb, chunk_size=10, chunk_overlap=0, max_chunks_per_file=3)
    big = work / "big.txt"
    big.write_text("x" * 100, encoding="utf-8")        # 10 块 → 截到 3
    small = work / "small.txt"
    small.write_text("y" * 30, encoding="utf-8")       # 3 块 → 完整
    idx.index_file(str(big))
    idx.index_file(str(small))

    got = db.truncated_files()
    paths = [t["path"] for t in got]
    assert paths == [str(big)], f"只应列出被截断的：{got}"
    assert got[0]["indexed"] == 3 and got[0]["total"] == 10


def test_truncated_files_ignores_unknown_total(env):
    """total_chunks=0（升级前入库、尚未重算）不参与——那是「未知」不是「未截断」。"""
    tmp, db = env
    db.conn.execute(
        "INSERT INTO files (file_id, path, name, file_type, content_hash, vectorize_status)"
        " VALUES ('f1','/kb/old.md','old.md','md','H','completed')")
    db.conn.commit()
    assert db.truncated_files() == []


# ---------------------------------------------------------------- A：progress 契约
def test_truncation_reported_separately_from_errors(env):
    """★ 核心断言：截断出现在 vectorizePhase.truncationDetails，**不在** errorDetails。"""
    tmp, db = env
    work, scanner, *_ = _mk(tmp, db, _Embed(), max_chunks_per_file=3)
    (work / "big.txt").write_text("x" * 100, encoding="utf-8")

    scanner.start()
    assert _wait_done(scanner)

    p = scanner.progress()
    vp, sp = p["vectorizePhase"], p["scanPhase"]
    # 截断：计数与明细同源同长
    assert vp["truncated"] == 1, f"应有 1 个被截断：{vp}"
    assert len(vp["truncationDetails"]) == vp["truncated"]
    assert vp["truncationDetails"][0]["total"] == 10
    # 失败：截断**不**算失败
    assert vp["failed"] == 0, f"截断不是失败：{vp}"
    # 错误明细：不得出现 oversized
    kinds = [d["kind"] for d in sp["errorDetails"]]
    assert "oversized" not in kinds, f"截断不该进错误明细：{sp['errorDetails']}"


def test_truncation_count_matches_list_length(env):
    """★ 用户看到的矛盾（按钮写 1、列表 12 条）不再可能出现：计数与列表同源。"""
    tmp, db = env
    work, scanner, *_ = _mk(tmp, db, _Embed(), max_chunks_per_file=3)
    for i in range(3):
        (work / f"big{i}.txt").write_text("x" * 100, encoding="utf-8")
    (work / "small.txt").write_text("y" * 30, encoding="utf-8")

    scanner.start()
    assert _wait_done(scanner)
    vp = scanner.progress()["vectorizePhase"]
    assert vp["truncated"] == 3
    assert len(vp["truncationDetails"]) == 3
    assert vp["failed"] == 0


def test_truncation_survives_rescan(env):
    """截断是**持久**属性：第二轮扫描（hash 全跳过）后仍然可见。

    这是与 errorDetails 的关键差别——后者按契约每轮清空，于是「文件只索引了一部分」
    这件事在扫描结束后就再也看不到了。
    """
    tmp, db = env
    work, scanner, *_ = _mk(tmp, db, _Embed(), max_chunks_per_file=3)
    (work / "big.txt").write_text("x" * 100, encoding="utf-8")
    scanner.start()
    assert _wait_done(scanner)
    assert scanner.progress()["vectorizePhase"]["truncated"] == 1

    scanner.restart()
    assert _wait_done(scanner)
    # 第二轮：内容没变 → 全部 skipped，错误明细清空，但截断仍可见
    assert scanner.progress()["scanPhase"]["errorDetails"] == []
    assert scanner.progress()["vectorizePhase"]["truncated"] == 1


# ---------------------------------------------------------------- D：失败分类
def test_embed_error_gets_its_own_kind(env):
    """★ 服务侧失败标成 "embed"，不再冒充「解析」错误。"""
    tmp, db = env
    emb = _Embed()
    work, scanner, *_ = _mk(tmp, db, emb)
    (work / "a.md").write_text("# 甲\n\n内容。\n", encoding="utf-8")
    emb.fail_with = EmbedError("单条输入 7 字符在 60.0s 内未算完——embed 服务可能过载或卡死")

    scanner.start()
    assert _wait_done(scanner)
    kinds = [d["kind"] for d in scanner.progress()["scanPhase"]["errorDetails"]]
    assert kinds == ["embed"], f"应标 embed：{kinds}"


def test_plain_exception_still_parse(env):
    """对照组：非服务侧异常仍归 "parse"（别把标签改宽了）。"""
    tmp, db = env
    emb = _Embed()
    work, scanner, *_ = _mk(tmp, db, emb)
    (work / "a.md").write_text("# 甲\n\n内容。\n", encoding="utf-8")
    emb.fail_with = ValueError("解析器内部错误")

    scanner.start()
    assert _wait_done(scanner)
    kinds = [d["kind"] for d in scanner.progress()["scanPhase"]["errorDetails"]]
    assert kinds == ["parse"], f"应仍归 parse：{kinds}"


def test_embed_error_is_runtime_error():
    """EmbedError 继承 RuntimeError：既有 `except RuntimeError`/测试断言不受影响。"""
    assert issubclass(EmbedError, RuntimeError)
