"""「失败文件」明细必须与失败计数同源（2026-09-28 真机实测）。

某台设备实测（装早期版本）：
    状态胶囊 = 「已完成（4 个失败）」
    「失败文件」统计格 = 0
    点「查看明细」= 空

根因：两处数据源不同。
    计数 `vectorizePhase.failed`  ← 按 DB 求值（持久、准）
    明细 `scanPhase.errorDetails` ← 内存流水，只有遍历/embed **抛异常**才写，
                                    且每轮 start() 清空
而解析失败（PDF 损坏 / xlsx 缺 [Content_Types].xml）走的是 index_file
**正常返回** status=failed 的那条路 —— 它从不写内存流水。于是：
    库里 4 行 failed（计数对） / 内存流水 0 条（明细空）

修法：失败明细改为按 DB 求值（files.fail_reason），与 truncated_files 同一范式；
内存流水里「有路径且不在 DB」的条目仍补在后面（保护「根本没进库」的失败，
例如文件名不是 UTF-8 —— 那种连行都没有，DB 永远数不到）。

本文件守的就是：**格上的数与展开的列表同源**，且原因可读。
"""
import tempfile
import time
from pathlib import Path

import pytest

from bibit_filesearch.db import Database  # noqa: E402
from bibit_filesearch.indexer import Indexer  # noqa: E402
from bibit_filesearch.scanner import ScanService  # noqa: E402


class _Embed:
    def __init__(self):
        self.dim = 8

    def _vec(self, text):
        return [float((hash((text, i)) % 97) + 1) for i in range(self.dim)]

    def embed_one(self, text):
        return self._vec(text)

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]

    def ping(self):
        return True

    def close(self):
        pass


class _Cfg:
    def __init__(self, dirs):
        self.index_dirs = dirs
        self.exclude_patterns = [".git"]

        class _Scan:
            def get(self, k, d=None):
                return {"io_throttle_ms": 0, "max_attempts": 3}.get(k, d)

        self.scan = _Scan()


def _mk_env(tmp: Path):
    db = Database(str(tmp / "t.db"))
    embed = _Embed()
    indexer = Indexer(db, embed)
    return db, indexer


def _wait_done(scanner, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if scanner.progress()["status"] == "done":
            return True
        time.sleep(0.05)
    return False


def _mk_broken(tmp: Path) -> Path:
    """一个**解析必失败**的文件：扩展名是 pdf，内容不是 PDF。

    这条路径正是真机那 4 个的形态 —— parser 抛异常、indexer 正常返回
    status=failed（走 parsed.error 分支），**不经过** errorDetails 那条内存流水。
    """
    p = tmp / "kb" / "broken.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"this is not a pdf at all")
    return p


def test_failed_count_and_details_come_from_same_source(tmp_path):
    """★ 核心：库里有 N 行 failed → 明细就有 N 条，且原因非空。

    红测判据：撤掉 scanner.progress 里 failed_details 的 DB 求值（退回只用内存
    流水）后，本用例必然失败 —— 因为解析失败从不写内存流水。
    """
    db, indexer = _mk_env(tmp_path)
    _mk_broken(tmp_path)
    scanner = ScanService(_Cfg([str(tmp_path / "kb")]), db, indexer)

    scanner.start()
    assert _wait_done(scanner), "扫描未在超时内完成"
    p = scanner.progress()

    failed = p["vectorizePhase"]["failed"]
    assert failed == 1, f"该文件应落 failed：{p['vectorizePhase']}"

    # ★ 异常就落在这里：修复前 len(details) == 0 而 failed == 1
    details = p["scanPhase"]["errorDetails"]
    assert len(details) == failed, (
        f"格上的数（failed={failed}）与明细条数（{len(details)}）必须同源；"
        f"修复前这里会是 0 条：{details}"
    )
    assert details[0]["reason"], "明细必须带原因（否则点开只看到路径，用户仍不知为何失败）"
    assert details[0]["path"].endswith("broken.pdf")


def test_failed_reason_persists_across_rescan(tmp_path):
    """原因落库 → 重扫（内容没变、跳过重试）后明细仍可读。

    内存流水的短板正在这里：扫描一结束/下一轮 start() 就清空。
    """
    db, indexer = _mk_env(tmp_path)
    _mk_broken(tmp_path)
    scanner = ScanService(_Cfg([str(tmp_path / "kb")]), db, indexer)

    scanner.start()
    assert _wait_done(scanner)
    first = scanner.progress()["scanPhase"]["errorDetails"]
    assert first and first[0]["reason"]

    scanner.start()                     # 第二轮：hash 未变 → 不会重试
    assert _wait_done(scanner)
    second = scanner.progress()["scanPhase"]["errorDetails"]
    assert [d["path"] for d in second] == [d["path"] for d in first], (
        f"重扫后明细不该消失（原因已落库）：{second}"
    )


def test_failed_details_stay_visible_after_restart(tmp_path):
    """restart()（改配置路径）也保留持久失败明细。"""
    db, indexer = _mk_env(tmp_path)
    _mk_broken(tmp_path)
    scanner = ScanService(_Cfg([str(tmp_path / "kb")]), db, indexer)

    scanner.start()
    assert _wait_done(scanner)
    assert scanner.progress()["scanPhase"]["errorDetails"], "首轮应有明细"

    scanner.restart()
    assert _wait_done(scanner)
    assert scanner.progress()["scanPhase"]["errorDetails"], (
        "restart 后持久失败明细不该被清空（清空的只是**本轮内存流水**）"
    )


def test_memory_only_failures_are_kept(tmp_path):
    """对照组：**没进库**的失败（文件名不是 UTF-8 那种）仍要从内存流水补进明细。

    否则修「库侧明细」会把这一类从界面上抹掉 —— 它恰恰是最需要可见的一类。
    """
    db, indexer = _mk_env(tmp_path)
    scanner = ScanService(_Cfg([str(tmp_path / "kb")]), db, indexer)
    ghost = tmp_path / "kb" / "ghost.txt"
    ghost.parent.mkdir(parents=True, exist_ok=True)
    ghost.write_text("x", encoding="utf-8")

    scanner.record_error("parse", str(ghost), "filenameUnencodable:{...}")
    details = scanner.progress()["scanPhase"]["errorDetails"]
    assert [d["path"] for d in details] == [str(ghost)], (
        f"库里没有行、只在内存流水里的失败也必须可见：{details}"
    )
    assert details[0]["reason"] == "filenameUnencodable:{...}"


def test_no_double_counting_when_both_sources_have_it(tmp_path):
    """同一条失败两处都有 → 明细里只出现一次（DB 那条为准，带持久原因）。"""
    db, indexer = _mk_env(tmp_path)
    broken = _mk_broken(tmp_path)
    scanner = ScanService(_Cfg([str(tmp_path / "kb")]), db, indexer)
    scanner.start()
    assert _wait_done(scanner)

    # 手工再往内存流水里塞同一条（模拟 embed 异常与解析失败撞在同一文件上）
    scanner.record_error("parse", str(broken), "另一条内存原因")
    details = scanner.progress()["scanPhase"]["errorDetails"]
    paths = [d["path"] for d in details]
    assert paths.count(str(broken)) == 1, f"同一条不该重复：{details}"
    assert details[0]["reason"] != "另一条内存原因", (
        "两处都有时应以 DB 那条为准（持久原因是用户最该看到的）"
    )


def test_deleted_file_drops_out_of_details(tmp_path):
    """文件删掉后不再列在失败明细里（与 errorDetails 原口径一致）。"""
    db, indexer = _mk_env(tmp_path)
    broken = _mk_broken(tmp_path)
    scanner = ScanService(_Cfg([str(tmp_path / "kb")]), db, indexer)
    scanner.start()
    assert _wait_done(scanner)
    assert scanner.progress()["scanPhase"]["errorDetails"], "先要有明细"

    broken.unlink()
    assert scanner.progress()["scanPhase"]["errorDetails"] == [], (
        "磁盘上已不存在的路径不该还挂在失败明细里"
    )


def test_failed_reason_is_structured_not_a_bare_sentence(tmp_path):
    """★ 2026-09-28（test3）：原因必须是**结构化标记**，不是裸英文句子。

    实测：界面选中文，明细右侧却是 `Failed to open file '…' as type pdf.`
    —— 后端把库的原始报错当句子输出，前端只能照打。
    现在 parser 给 `parseFailed:{"kind":…,"errorType":…,"detail":…}`，
    词由前端 locales 按语言拼（与 `filenameUnencodable:` 同一套做法）。

    红测判据：把 parser 的 describe_parse_failure 换回 `str(e)[:80]` 必失败。
    """
    import json as _json

    db, indexer = _mk_env(tmp_path)
    _mk_broken(tmp_path)
    scanner = ScanService(_Cfg([str(tmp_path / "kb")]), db, indexer)
    scanner.start()
    assert _wait_done(scanner)

    reason = scanner.progress()["scanPhase"]["errorDetails"][0]["reason"]
    assert reason.startswith("parseFailed:"), (
        f"原因应是结构化标记（前端据此按语言出词），实际：{reason!r}"
    )
    payload = _json.loads(reason[len("parseFailed:"):])
    assert payload["kind"] == "parseFailed"
    assert payload["errorType"], "应带错误类型（排障用）"
    assert payload["detail"], "应带原始细节（翻译会丢信息，必须保留）"
    # 壳在文案里，标记本身不该是一句可直接展示的英文
    assert not reason.startswith("Failed to open file"), reason


def test_ocr_failure_reason_is_structured(tmp_path, monkeypatch):
    """OCR 失败同样结构化（`ocrFailed:`），覆盖原先的 `ocr: ` 拼句。"""
    import json as _json

    from bibit_filesearch import parser as P

    def _boom(_p):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(P, "_ocr_image", _boom)
    img = tmp_path / "kb" / "pic.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    doc = P.parse_file(img)
    assert doc.error.startswith("ocrFailed:"), doc.error
    payload = _json.loads(doc.error[len("ocrFailed:"):])
    assert payload["detail"] == "engine exploded"
