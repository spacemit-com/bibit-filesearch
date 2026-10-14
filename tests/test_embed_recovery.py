"""embed 恢复看门狗测试（2026-09-09）。

契约：embed 从 down → up 跳变、且库里有 failed/queued 文件时自动重建索引；
其余情况（无跳变 / 无待重跑文件 / 扫描进行中或暂停）不得触发。
"""
import tempfile
import time
from pathlib import Path

import pytest


from bibit_filesearch.db import Database  # noqa: E402
from bibit_filesearch.indexer import Indexer  # noqa: E402
from bibit_filesearch.recovery import EmbedRecovery  # noqa: E402
from bibit_filesearch.retriever import Retriever  # noqa: E402
from bibit_filesearch.scanner import ScanService  # noqa: E402


class _SwitchableEmbed:
    """可切换可用性的 embed 替身。"""

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


class _SpyScanner:
    """记录 start() 调用次数的 scanner 替身。"""

    def __init__(self, status="done"):
        self.status = status
        self.starts = 0

    def start(self):
        self.starts += 1
        return {"status": "scanning", "message": "扫描已启动"}


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_recovery_"))
    db = Database(str(tmp / "test.db"))
    embed = _SwitchableEmbed()
    yield tmp, db, embed
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _pending_file(db, tmp, status="failed"):
    """造一条指定状态的文件记录（watchdog 只查 vectorize_status 计数）。"""
    path = tmp / f"{status}.md"
    path.write_text("x", encoding="utf-8")
    db.upsert_file(
        {
            "fileId": "", "filePath": str(path), "fileName": path.name, "fileType": "md",
            "fileSize": 1, "createdAt": "2026-09-09T00:00:00", "modifiedAt": "2026-09-09T00:00:00",
            "accessedAt": "2026-09-09T00:00:00", "author": "", "directory": str(tmp),
        },
        "", "", status=status,
    )
    return path


def _wait_probes(rec, n, timeout=5.0):
    """确定性等待：轮询到探测次数 >= n（不靠 sleep 撞时机）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and rec.probes < n:
        time.sleep(0.01)


def test_recovers_on_down_to_up_transition(env):
    """down → up 跳变 + 有 failed 文件 → 触发一次 start()。"""
    tmp, db, embed = env
    _pending_file(db, tmp, "failed")
    scanner = _SpyScanner()
    rec = EmbedRecovery(embed, db, scanner, interval=0.05)
    embed.down = True
    rec.start()          # 基线：down
    _wait_probes(rec, 2)
    assert scanner.starts == 0, "掉线期间不应触发"
    embed.down = False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and scanner.starts == 0:
        time.sleep(0.02)
    rec.stop()
    assert scanner.starts == 1, f"恢复后应触发一次（实际 {scanner.starts}）"


def test_no_trigger_without_pending_files(env):
    """跳变但无 failed/queued 文件 → 不触发（避免无谓全盘重扫）。"""
    tmp, db, embed = env
    scanner = _SpyScanner()
    rec = EmbedRecovery(embed, db, scanner, interval=0.05)
    embed.down = True
    rec.start()
    _wait_probes(rec, 2)
    embed.down = False
    _wait_probes(rec, 4)
    rec.stop()
    assert scanner.starts == 0, "无待重跑文件不应触发"


def test_no_trigger_when_already_up(env):
    """一直可用（无跳变）→ 不触发，即使有 failed 文件。"""
    tmp, db, embed = env
    _pending_file(db, tmp, "failed")
    scanner = _SpyScanner()
    embed.down = False
    rec = EmbedRecovery(embed, db, scanner, interval=0.05)
    rec.start()
    _wait_probes(rec, 4)
    rec.stop()
    assert scanner.starts == 0, "无跳变不应触发（否则永久失败文件被无限重扫）"


def test_skips_when_scanning_or_paused(env):
    """跳变但扫描进行中/暂停 → 跳过，不打断用户操作。"""
    for status in ("scanning", "paused"):
        tmp, db, embed = env
        _pending_file(db, tmp, "queued")
        scanner = _SpyScanner(status=status)
        rec = EmbedRecovery(embed, db, scanner, interval=0.05)
        embed.down = True
        rec.start()
        _wait_probes(rec, 2)
        embed.down = False
        _wait_probes(rec, 4)
        rec.stop()
        assert scanner.starts == 0, f"status={status} 时不应触发"


def test_default_interval_is_five_seconds(env):
    """生产默认探测间隔 = 5s（2026-09-09 用户反馈延迟后由 30s 收紧）。

    这条守的是「默认值」本身：EmbedRecovery 的默认参数与 main.create_app 的
    config 兜底必须都是 5，否则用户 start embed 后又要白等最多 30s 的探测节拍。
    两处任一被改回 30 即变红。
    """
    tmp, db, embed = env
    rec = EmbedRecovery(embed, db, _SpyScanner())  # 不传 interval → 用默认
    assert rec.interval == 5.0, f"EmbedRecovery 默认间隔应为 5s（实际 {rec.interval}）"
    rec.stop()

    # main.create_app 的兜底：配置里没有 scan.embed_recovery_interval 时应为 5
    import inspect

    from bibit_filesearch import main as main_mod

    src = inspect.getsource(main_mod.create_app)
    assert 'embed_recovery_interval", 5' in src, \
        "create_app 的 embed_recovery_interval 兜底应为 5（改回 30 会重现 25s 延迟）"
    assert 'embed_recovery_interval", 30' not in src, "不得回退到 30s"
