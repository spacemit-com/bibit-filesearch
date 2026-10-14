"""全盘扫描 + 队列 + 管理接口测试。

覆盖：
- 启动自动扫描配置目录；两阶段进度接口（遍历 + 向量化）
- 暂停/继续作用于整个任务（遍历与 worker 都停）
- 遍历完成 → 队列消费完毕 → done；逐文件立即可检索
- 持久化：重启后从队列续跑（不重复已完成）
- 删除清理（磁盘删文件 → 索引移除）
- hash 跳过（mtime 变内容不变不重算）
- 不可索引类型仅元数据
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest


from fake_embed import FakeEmbed  # noqa: E402
from bibit_filesearch.db import Database
from bibit_filesearch.indexer import Indexer  # noqa: E402
from bibit_filesearch.scanner import ScanService  # noqa: E402
from bibit_filesearch.retriever import Retriever  # noqa: E402



class FakeConfig:
    """最小配置替身（避免依赖 config.yaml 的真实路径）。"""

    def __init__(self, index_dirs, excludes=None):
        self.index_dirs = index_dirs
        self.exclude_patterns = excludes or [".git", "__pycache__"]

        class _Scan:
            def get(self, k, d=None):
                return {"io_throttle_ms": 0}.get(k, d)

        self.scan = _Scan()


@pytest.fixture()
def env():
    """每个测试独立 db + tmp（扫描测试会污染 files 表，必须隔离）。"""
    tmp = Path(tempfile.mkdtemp(prefix="bibit_t7_"))
    db = Database(str(tmp / "test.db"))
    embed = FakeEmbed()
    yield tmp, db, embed
    embed.close()
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def _wait_status(scanner, target, timeout=30):
    """轮询等待状态到达 target。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if scanner.status == target:
            return True
        time.sleep(0.1)
    return scanner.status == target


def test_scan_full_flow(env):
    tmp, db, embed = env
    # 造测试目录（3 个文件 + 1 个排除目录 + 1 个不可索引）
    work = tmp / "work"
    (work / "sub").mkdir(parents=True, exist_ok=True)
    (work / "a.md").write_text("# 文档甲\n\n内容甲。\n", encoding="utf-8")
    (work / "sub" / "b.md").write_text("# 文档乙\n\n内容乙。\n", encoding="utf-8")
    (work / "c.py").write_text("def func():\n    pass\n", encoding="utf-8")
    (work / "data.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 16)
    (work / ".git").mkdir(exist_ok=True)
    (work / ".git" / "config").write_text("exclude", encoding="utf-8")

    indexer = Indexer(db, embed)
    retriever = Retriever(db, embed)
    indexer.set_on_indexed(retriever.on_file_indexed)
    indexer.set_on_deleted(retriever.on_file_deleted)
    scanner = ScanService(FakeConfig([str(work)]), db, indexer)

    # 启动
    r = scanner.start()
    assert r["status"] == "scanning"
    # 等待 done（遍历 + 队列消费完毕）
    assert _wait_status(scanner, "done", timeout=60), f"扫描应完成，当前 {scanner.status}"

    # 两阶段进度：3 个可索引文件已入库（zip 仅元数据）
    prog = scanner.progress()
    assert prog["scanPhase"]["walked"] >= 3
    assert prog["vectorizePhase"]["completed"] >= 3
    # .git 被排除
    assert db.get_file_by_path(str(work / ".git" / "config")) is None
    # zip 仅元数据（queued 不在 completed，但文件在库中）
    assert db.get_file_by_path(str(work / "data.zip")) is not None
    # 检索立即可用
    res = retriever.search({"query": "文档甲", "page": 1, "pageSize": 10})
    assert res["data"]["total"] >= 1


def test_scan_pause_resume(env):
    tmp, db, embed = env
    work = tmp / "pause_work"
    work.mkdir(exist_ok=True)
    # 造足够多的文件让遍历时间 > 暂停窗口
    for i in range(30):
        (work / f"f{i:02d}.md").write_text(f"# 文档{i}\n\n内容{i}。\n", encoding="utf-8")

    indexer = Indexer(db, embed)
    retriever = Retriever(db, embed)
    indexer.set_on_indexed(retriever.on_file_indexed)
    indexer.set_on_deleted(retriever.on_file_deleted)
    scanner = ScanService(FakeConfig([str(work)]), db, indexer)

    scanner.start()
    # 立即暂停（遍历尚未完成；pause 幂等：done 后 pause 返回当前状态）
    r = scanner.pause()
    assert r["status"] in ("paused", "done"), f"暂停应生效: {r}"
    if r["status"] == "done":
        # 已完成则直接通过（30 文件太小，扫描太快）
        pytest.skip("扫描过快已完成，跳过暂停验证")
    # 暂停后：遍历/worker 在检查点停下。pause() 是异步的——清 event 时正在入队的
    # 那个文件（_walk_dir 的 wait 检查点之间）会走完才停，walked 允许 +1（in-flight）；
    # 之后必须真正冻结。轮询至 walked 连续两次相等 = 冻结成立；若 pause 失效，
    # walked 会一路涨到 total，deadline 超时即失败（2026-09-04：FakeEmbed 扫描极快，
    # 原"两次快照立即相等"断言 racy——pause 常落在文件间隙触发 +1）。
    deadline = time.monotonic() + 3.0
    prev_walked = None
    while True:
        p = scanner.progress()
        assert p["status"] == "paused", f"暂停态丢失: {p}"
        if p["scanPhase"]["walked"] == prev_walked:
            break  # 遍历侧已冻结
        assert time.monotonic() < deadline, f"暂停未冻结进度: {p}"
        prev_walked = p["scanPhase"]["walked"]
        time.sleep(0.2)
    # 继续 → 最终完成
    r = scanner.resume()
    assert r["status"] == "scanning"
    assert _wait_status(scanner, "done", timeout=60)
    assert scanner.progress()["vectorizePhase"]["completed"] == 30


def test_scan_delete_prune(env):
    tmp, db, embed = env
    work = tmp / "prune_work"
    work.mkdir(exist_ok=True)
    p = work / "gone.md"
    p.write_text("# 将删除\n\n内容。\n", encoding="utf-8")

    indexer = Indexer(db, embed)
    retriever = Retriever(db, embed)
    indexer.set_on_indexed(retriever.on_file_indexed)
    indexer.set_on_deleted(retriever.on_file_deleted)
    scanner = ScanService(FakeConfig([str(work)]), db, indexer)

    scanner.start()
    assert _wait_status(scanner, "done", timeout=60)
    fid = db.get_file_by_path(str(p))["file_id"]
    assert db.get_chunks(fid) != []

    # 磁盘删除 → 再次扫描 → 索引清理
    p.unlink()
    scanner.start()
    assert _wait_status(scanner, "done", timeout=60)
    assert db.get_file_by_path(str(p)) is None, "删除文件应从索引移除"
    # chunks 也清掉
    assert db.get_chunks(fid) == []


def test_scan_restart_resume_queue(env):
    """持久化续跑：模拟重启（新实例同 db），已完成的不重算、队列继续消费。"""
    tmp, db, embed = env
    work = tmp / "restart_work"
    work.mkdir(exist_ok=True)
    for i in range(20):
        (work / f"r{i:02d}.md").write_text(f"# 重启文档{i}\n\n内容{i}。\n", encoding="utf-8")

    # 第一轮：跑一半就"重启"（新 Indexer/Scanner，同 db）
    idx1 = Indexer(db, embed)
    r1 = Retriever(db, embed)
    idx1.set_on_indexed(r1.on_file_indexed)
    idx1.set_on_deleted(r1.on_file_deleted)
    sc1 = ScanService(FakeConfig([str(work)]), db, idx1)
    sc1.start()
    assert _wait_status(sc1, "done", timeout=60)

    # 第二轮：新实例（模拟重启），应全部 skipped（hash 未变），不再重复向量化
    idx2 = Indexer(db, embed)
    r2 = Retriever(db, embed)
    idx2.set_on_indexed(r2.on_file_indexed)
    idx2.set_on_deleted(r2.on_file_deleted)
    sc2 = ScanService(FakeConfig([str(work)]), db, idx2)
    sc2.start()
    assert _wait_status(sc2, "done", timeout=60)
    # 第二轮不应产生新向量化（全部 skipped → queued 清空，completed 保持 20）
    prog = sc2.progress()
    assert prog["vectorizePhase"]["completed"] == 20
    assert prog["vectorizePhase"]["queued"] == 0


def test_scan_http_api(env, monkeypatch):
    """管理接口 HTTP 冒烟（复用 test_t5_http 的模式，独立服务实例）。

    2026-09-04：create_app 从 config 构造 EmbedClient 连真实 :8082——
    测试用 monkeypatch 把 model_client.EmbedClient 换成 FakeEmbed
    （函数级 import，create_app 每次调用时取当前属性，patch 生效）。
    """
    import bibit_filesearch.model_client as mc
    from fake_embed import FakeEmbed

    monkeypatch.setattr(mc, "EmbedClient", FakeEmbed)
    tmp, db, embed = env
    work = tmp / "http_work"
    work.mkdir(exist_ok=True)
    (work / "api.md").write_text("# API 文档\n\n扫描接口。\n", encoding="utf-8")

    import json
    import urllib.request
    import uvicorn

    from bibit_filesearch.config import load_config
    from bibit_filesearch.main import create_app

    # 临时配置：只扫 work
    import yaml

    cfg_path = tmp / "scan_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"index_dirs": [str(work)], "storage": {"db_path": str(tmp / "http_scan.db")}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    app = create_app(str(cfg_path))
    port = 18053
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    time.sleep(1.5)
    base = f"http://127.0.0.1:{port}"

    def post(path):
        req = urllib.request.Request(base + path, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def get(path):
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return json.load(r)

    try:
        r = post("/api/scan")
        # 2026-09-01 契约统一：{code,message,data} 三层包装
        assert r["code"] == 0
        assert r["data"]["status"] == "scanning"
        # 等 done。2026-09-23（池子化）：这条用例在全量套件里约 1/5~1/10
        # 失败于 `completed >= 1`，而单独跑 10/10 通过。失败时把整帧打出来，
        # 否则只剩一个「assert 0 >= 1」无法判断 done 是谁落的、行都去哪了。
        frames = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            p = get("/api/scan/progress")["data"]
            frames.append(p)
            if p["status"] == "done":
                break
            time.sleep(0.2)
        prog = get("/api/scan/progress")
        assert prog["code"] == 0
        if prog["data"]["vectorizePhase"]["completed"] < 1:
            import json as _j

            dbg = {
                "first_frame": frames[0] if frames else None,
                "last_frames": frames[-6:],
                "final": prog["data"],
                "walked_hist": [(f["scanPhase"]["walked"], f["scanPhase"]["total"],
                                 f["status"],
                                 f["vectorizePhase"]["completed"],
                                 f["vectorizePhase"].get("pending"),
                                 f["vectorizePhase"]["failed"],
                                 f["vectorizePhase"]["unsupported"]) for f in frames],
                "db_path": str(tmp / "http_scan.db"),
                "index_dirs_seen": get("/api/settings")["data"].get("index_dirs")
                if get("/api/settings")["code"] == 0 else None,
            }
            Path("/tmp/t7_http_fail.json").write_text(_j.dumps(dbg, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
            print("★ 诊断帧已写 /tmp/t7_http_fail.json")
        assert prog["data"]["status"] == "done"
        assert prog["data"]["vectorizePhase"]["completed"] >= 1
        # 搜索命中
        sr = post("/api/search")  # 无 body，用 GET 替代
        # 直接调 search
        req = urllib.request.Request(
            base + "/api/search",
            data=json.dumps({"query": "扫描接口", "page": 1, "pageSize": 5}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as rr:
            sres = json.load(rr)
        assert sres["data"]["total"] >= 1
    finally:
        srv.should_exit = True
        t.join(timeout=3)
