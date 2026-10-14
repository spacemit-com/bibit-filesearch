"""「未纳入索引」增量刷新：把文件移出被排除目录后，计数与明细必须跟着收敛。

现象（用户实测 2026-10-10）：`/home/user/test/excluded/` 下的文件 + 库侧 unsupported
构成 UI 的「未纳入索引 14」。用户 `mv excluded/* .` 之后，这些文件被 watcher 增量索引
进库（「已索引」涨到 6），但「未纳入索引」仍显示 14（明细也是旧的），要等下一次**全盘
扫描**才会重算。

根因：遍历侧那份读数（`_excluded_count` / `_excluded_details` / `_excluded_kinds`）
只在 `scanner.start()` / `restart()` 开头清零，且**只有全盘 `_walk_dir` 会往里加**。
watcher 的增量路径（offer → 索引 → completed）完全不碰它 → 文件不再命中排除规则后，
这一格仍停在上一轮遍历的值。

修法：watcher 在增删移事件上打「需要重算」标记（`request_excluded_refresh`），索引池
在静默时做一次全量重算（`recount_excluded`，与 OCR 空闲释放同一个收尾点）。

本文件钉住契约：**文件不再命中排除规则时，未纳入索引的计数与明细要即时收敛。**
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer
from bibit_filesearch.scanner import ScanService


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
    def __init__(self, dirs, excludes=None, max_depth=0, indexable_types=None):
        self.index_dirs = [str(d) for d in dirs]
        self.exclude_patterns = list(excludes or [])
        self.indexable_types = list(indexable_types or ["txt", "md", "pdf"])
        self.scan = {"io_throttle_ms": 0, "max_depth": max_depth, "max_attempts": 3}


@pytest.fixture()
def env():
    # ★ 库目录与索引目录分离：否则 test.db / -wal / -shm 落在索引目录里，会被算成
    #   「格式不支持」，把断言要观察的排除计数搅浑（踩过一次）。
    tmp = Path(tempfile.mkdtemp(prefix="bibit_notidx_refresh_"))
    kb = tmp / "kb"
    kb.mkdir()
    db = Database(str(tmp / "data" / "test.db"))
    embed = _Embed()
    idx = Indexer(db, embed, indexable_types={"txt", "md", "pdf"})
    cfg = _Cfg([kb], excludes=["excluded"])
    scanner = ScanService(cfg, db, idx)
    # 测试里关掉节流（真实默认 2s），让重算在断言窗口内必然发生一次。
    _orig = scanner.recount_excluded

    def _no_throttle(*, force=False, throttle_secs=0.0):
        return _orig(force=force, throttle_secs=0.0)

    scanner.recount_excluded = _no_throttle          # type: ignore[assignment]
    yield kb, db, idx, scanner
    embed.close()
    db.close()
    shutil.rmtree(tmp, ignore_errors=True)


def _wait(pred, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return bool(pred())


def _phase(s):
    return s.progress()["notIndexedPhase"]


def test_not_indexed_recovers_after_files_leave_excluded_dir(env):
    """移出被排除目录后，未纳入索引必须收敛（未修时停在旧值 → 红）。"""
    kb, db, idx, scanner = env
    ex = kb / "excluded"
    (ex / "picture").mkdir(parents=True)
    for i in range(3):
        (ex / f"junk{i}.txt").write_text(f"junk {i}\n", encoding="utf-8")
    (ex / "picture" / "deep.txt").write_text("deep\n", encoding="utf-8")
    (kb / "keep.md").write_text("kept\n", encoding="utf-8")

    scanner.start()
    try:
        assert _wait(lambda: _phase(scanner)["total"] >= 4), \
            "被排除目录里的 4 个文件应计入未纳入索引；实际 %s" % _phase(scanner)
        assert _phase(scanner)["byKind"].get("rule", 0) >= 4, _phase(scanner)

        # 用户操作：mv excluded/* .  —— 文件移出被排除目录
        moved = []
        for f in list(ex.rglob("*.txt")):
            dst = kb / f.name
            shutil.move(str(f), str(dst))
            moved.append(dst)

        # 增量路径：watcher 的等价动作 = 打标记 + offer
        scanner.request_excluded_refresh()
        for p in moved:
            scanner._enqueue_file(str(p))

        assert _wait(lambda: all(
            (db.get_file_by_path(str(p)) or {}).get("vectorize_status") == "completed"
            for p in moved), timeout=15), "移出的文件应被索引成 completed"

        # ★ 契约：不再命中排除规则的文件，必须从「未纳入索引」里消失
        assert _wait(lambda: _phase(scanner)["total"] == 0, timeout=10), (
            "文件移出被排除目录后，未纳入索引必须收敛到 0；"
            "实际仍是 %s（明细 %d 条）—— 这就是 UI 停在 14 的根因"
            % (_phase(scanner)["total"], len(_phase(scanner)["details"]))
        )
        assert _phase(scanner)["details"] == [], "明细也必须收敛，否则展开还是旧的那几条"
        assert _phase(scanner)["byKind"].get("rule", 0) == 0, _phase(scanner)
    finally:
        scanner.stop()


def test_recount_reflects_newly_excluded_dir(env):
    """反向：新建一个命中排除规则的目录后，重算要能把它算进来。"""
    kb, db, idx, scanner = env
    (kb / "keep.md").write_text("k\n", encoding="utf-8")
    scanner.start()
    try:
        assert _wait(lambda: _phase(scanner)["total"] == 0), _phase(scanner)
        ex = kb / "excluded"
        ex.mkdir()
        (ex / "a.txt").write_text("a\n", encoding="utf-8")
        (ex / "b.txt").write_text("b\n", encoding="utf-8")
        scanner.request_excluded_refresh()
        assert _wait(lambda: _phase(scanner)["total"] == 2, timeout=10), _phase(scanner)
        assert _phase(scanner)["byKind"].get("rule", 0) == 2, _phase(scanner)
    finally:
        scanner.stop()


def test_recount_skips_while_scanning(env):
    """全盘遍历进行中不做重算（那份读数由遍历自己填，别抢）。"""
    kb, db, idx, scanner = env
    (kb / "keep.md").write_text("k\n", encoding="utf-8")
    scanner.request_excluded_refresh()
    scanner._status = "scanning"
    assert scanner.recount_excluded(force=True, throttle_secs=0.0) is False, \
        "扫描进行中不该重算（会被遍历覆盖成半成品）"
