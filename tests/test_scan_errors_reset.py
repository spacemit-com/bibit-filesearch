"""扫描错误明细重置测试（2026-09-09 用户实测暴露）。

现象：失败文件归 0（重跑成功）后，GUI 展开明细仍是上一轮失败项。
根因：ScanService._error_logs 只在 __init__ 建空列表，start()/restart() 不清，
而 progress() 的 errorDetails 契约是"本次扫描"的错误 → 跨轮次泄漏。

断言：新一轮扫描（start/restart）后 errorDetails 为空；
失败轮次仍有明细（对照组，防止"清空"改成"永不记录"）。
"""
import tempfile
import time
from pathlib import Path

import pytest


from bibit_filesearch.db import Database  # noqa: E402
from bibit_filesearch.indexer import Indexer  # noqa: E402
from bibit_filesearch.retriever import Retriever  # noqa: E402
from bibit_filesearch.scanner import ScanService  # noqa: E402


class _SwitchableEmbed:
    """可控 embed：down=True 时抛连接错误（模拟 8082 掉线），False 时正常向量化。"""

    def __init__(self):
        self.down = True
        self.dim = 8

    def _vec(self, text: str):
        return [float((hash((text, i)) % 97) + 1) for i in range(self.dim)]

    def embed_one(self, text: str):
        if self.down:
            raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")
        return self._vec(text)

    def embed_batch(self, texts):
        if self.down:
            raise RuntimeError("embed 请求失败（重试 2 次）: [Errno 111] Connection refused")
        return [self._vec(t) for t in texts]

    def ping(self) -> bool:
        return not self.down

    def close(self) -> None:
        pass


class _Cfg:
    def __init__(self, dirs):
        self.index_dirs = dirs
        self.exclude_patterns = [".git"]

        class _Scan:
            def get(self, k, d=None):
                return {"io_throttle_ms": 0}.get(k, d)

        self.scan = _Scan()


def _wait_done(scanner, timeout=30):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if scanner.status == "done":
            return True
        time.sleep(0.1)
    return scanner.status == "done"


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_errreset_"))
    db = Database(str(tmp / "test.db"))
    embed = _SwitchableEmbed()
    yield tmp, db, embed
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _mk_scanner(tmp, db, embed):
    work = tmp / "work"
    work.mkdir(exist_ok=True)
    (work / "a.md").write_text("# 文档甲\n\n内容甲。\n", encoding="utf-8")
    (work / "b.txt").write_text("内容乙。\n", encoding="utf-8")
    indexer = Indexer(db, embed)
    retriever = Retriever(db, embed)
    indexer.set_on_indexed(retriever.on_file_indexed)
    indexer.set_on_deleted(retriever.on_file_deleted)
    return ScanService(_Cfg([str(work)]), db, indexer)


def test_error_details_cleared_on_rescan(env):
    """embed 掉线扫描（明细非空）→ embed 恢复后重扫 → 明细清空、失败归 0。"""
    tmp, db, embed = env
    scanner = _mk_scanner(tmp, db, embed)

    # 第一轮：embed 掉线 → 全部失败，明细有记录
    embed.down = True
    scanner.start()
    assert _wait_done(scanner, timeout=30)
    p1 = scanner.progress()
    assert p1["vectorizePhase"]["failed"] >= 1, f"掉线应产生失败：{p1}"
    assert len(p1["scanPhase"]["errorDetails"]) >= 1, f"应记录明细：{p1}"

    # 第二轮：embed 恢复 → 失败归 0，明细必须清空（不能留上一轮）
    embed.down = False
    scanner.start()
    assert _wait_done(scanner, timeout=30)
    p2 = scanner.progress()
    assert p2["vectorizePhase"]["failed"] == 0, f"恢复后应无失败：{p2}"
    assert p2["scanPhase"]["errorDetails"] == [], \
        f"新一轮扫描应清空错误明细（实际 {p2['scanPhase']['errorDetails']}）"


def test_error_details_cleared_on_restart(env):
    """restart()（改配置路径）同样清空明细。"""
    tmp, db, embed = env
    scanner = _mk_scanner(tmp, db, embed)

    embed.down = True
    scanner.start()
    assert _wait_done(scanner, timeout=30)
    assert scanner.progress()["scanPhase"]["errorDetails"], "首轮应有明细"

    embed.down = False
    scanner.restart()
    assert _wait_done(scanner, timeout=30)
    assert scanner.progress()["scanPhase"]["errorDetails"] == []
