"""非 UTF-8 文件名（GBK 等）导致的接口 500 与审计日志静默丢失（2026-09-21）。

背景（真机实测，169 上 `test2/cuda/docs/` 下两个 GBK 文件名的 md）：
  文件系统允许文件名不是 UTF-8；`os.walk` 通过 surrogateescape 把它变成含 `\\udcc9`
  的 str。这种字符串**能碰文件系统**（`os.path.exists`/`open` 都可用），但**不能编码**，
  于是：
    · `_enqueue_file` → `db.get_file_by_path` → sqlite3 参数编码 → UnicodeEncodeError
      → 该文件永远进不了库（每轮扫描都失败）；
    · 这条失败明细带着含 surrogate 的 path 进 `_error_logs`，`progress()` 原样返回,
      FastAPI 的 `JSONResponse.render()` 里 `.encode("utf-8")` 抛错 → **整个
      /api/scan/progress 500**（实测 17:54:22 起持续 500），前端 refreshStatus 的
      catch 把「进度查不到」当成「服务不可用」；
    · `_record_error` 写 scan_errors.log 时 `open(..., encoding="utf-8")` 也编不出来，
      而它包在 `except Exception: pass` 里 → **该失败在任何审计文件里都查不到**。

本文件钉四件事：
  ① display_path 的语义（能编码、不丢原始字节的判据）；
  ② progress() 的输出必须「可 JSON 序列化」且**端点不 500**（真造一个非法名文件）；
  ③ 判存在仍用原路径（不能因为安全化就把条目从界面上抹掉）；
  ④ _record_error 落盘不再静默失败，且写明原因。
"""
import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from bibit_filesearch.pathutil import (
    describe_unencodable_filename,
    display_path,
    is_utf8_safe,
)

# GBK 的「深度拷问清单.md」——真机上就是这么两个名字
GBK_NAME = b"\xc9\xee\xb6\xc8\xbf\xbd\xce\xca\xc7\xe5\xb5\xa5.md"


def _make_non_utf8_file(dir_path: Path, name_bytes: bytes = GBK_NAME) -> str:
    """在 dir_path 下造一个文件名为非 UTF-8 字节的文件，返回 os.fsdecode 后的 str。"""
    raw = os.fsencode(str(dir_path)) + b"/" + name_bytes
    fd = os.open(raw, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, "# 内容\nCUDA 深度拷问清单\n".encode("utf-8"))
    finally:
        os.close(fd)
    return os.fsdecode(raw)


# ─────────────────────────────────────────────────────────────
# 一、display_path 的语义
# ─────────────────────────────────────────────────────────────

def test_utf8_path_passes_through_unchanged():
    """合法路径原样返回（绝大多数情况零改动，不能引入无谓的替换）。"""
    for p in ("/home/user/test2/a.txt", "/tmp/中文目录/文件.md", ""):
        assert display_path(p) == p


def test_non_utf8_path_becomes_encodable():
    """含 surrogate 的路径必须变成能 encode('utf-8') 的字符串。"""
    bad = os.fsdecode(b"/kb/\xc9\xee\xb6\xc8\xbf\xbd\xce\xca\xc7\xe5\xb5\xa5.md")
    assert not is_utf8_safe(bad), "前提：这个 path 确实编不了码"
    got = display_path(bad)
    assert is_utf8_safe(got)
    got.encode("utf-8")  # 不应抛
    assert "/kb/" in got, "目录部分必须保留，便于人眼定位"
    assert "\ufffd" in got, "非法字节替换为 U+FFFD"


def test_display_path_keeps_prefix_and_suffix():
    """替换只发生在非法字节上，前后文保留。"""
    bad = os.fsdecode(b"/a/b/CUDA_GPU_\xd3\xb2\xbc\xfe.md")
    got = display_path(bad)
    assert got.startswith("/a/b/CUDA_GPU_")
    assert got.endswith(".md")


def test_describe_explains_gbk_name():
    """返回**结构化事实**（不是成句文案）：用户要能看到「按 GBK 读是什么」。

    2026-09-23 改契约：此前这里返回一句拼好的中文，界面原样打印 → 界面切英文时那一行
    仍是中文。现在只给事实，词由界面按语言拼（见 locales failedFiles.reason*）。
    """
    bad = os.fsdecode(b"/kb/\xc9\xee\xb6\xc8\xbf\xbd\xce\xca\xc7\xe5\xb5\xa5.md")
    info = describe_unencodable_filename(bad)
    assert info["kind"] == "unencodableFilename"
    assert "深度拷问清单" in info["guess"], f"应给出按常见编码的读法，实际: {info}"
    assert info["encoding"] == "gb18030", info
    assert "\\xc9" in info["base"], "原始字节要留着，排查时靠它定位"
    json.dumps(info, ensure_ascii=False).encode("utf-8")  # 必须能编码出接口


def test_describe_is_encodable_even_for_binary_junk():
    """极端脏名（连 gb18030 都读不出）也要给出可编码的结构，不能反过来炸。"""
    bad = os.fsdecode(b"/kb/\xff\xfe\x00\x80.md")
    info = describe_unencodable_filename(bad)
    json.dumps(info, ensure_ascii=False).encode("utf-8")
    assert info["kind"] == "unencodableFilename"
    assert info["base"]


def test_error_detail_uses_structured_reason_not_chinese_prose():
    """失败明细的 reason 不得是写死的中文成句文案（用户切英文时会漏中文）。"""
    import json as _json

    from bibit_filesearch.scanner import ScanService

    kb = "/tmp/kb-just-a-name"
    bad = os.fsdecode(b"/tmp/\xc9\xee\xb6\xc8.md")
    marker = ScanService._unencodable_marker(None, bad)
    assert marker.startswith("filenameUnencodable:"), marker
    payload = _json.loads(marker.split(":", 1)[1])
    assert payload["kind"] == "unencodableFilename"
    assert "深度" in payload["guess"]
    # 关键：整串里不能出现「请把该文件重命名」这类成品中文句子
    assert "重命名" not in marker, "成品文案应由前端 locales 提供，后端只给事实"
    assert kb  # 保留形参语义（本用例不依赖真实文件）


# ─────────────────────────────────────────────────────────────
# 二、端点级：/api/scan/progress 不得 500
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def app_ctx(tmp_path, monkeypatch):
    """与 test_health_deep 同构：临时配置 + create_app + TestClient。

    `BIBIT_FILESEARCH_DATA_ROOT` 必须一起设：create_app 的 data_root 优先级高于
    config.storage.db_path（main.py:98 `env 覆盖 > config.storage.db_path`），只改配置
    的话数据目录仍是 `~/.local/share/bibit-filesearch` —— 测试会写到**真实库目录**的
    scan_errors.log 上（实测踩到：断言读不到文件，同时污染了本机 root 的日志）。
    """
    data = tmp_path / "data"
    data.mkdir()
    kb = tmp_path / "kb"
    kb.mkdir()
    cfg = {
        "storage": {"db_path": str(data / "scanpath.db")},
        "server": {"port": 18099, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "top_k": 50,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    monkeypatch.setenv("BIBIT_CONFIG", str(cfg_path))
    monkeypatch.setenv("BIBIT_FILESEARCH_DATA_ROOT", str(data))
    from fastapi.testclient import TestClient
    from bibit_filesearch.main import create_app

    app = create_app(str(cfg_path))
    assert Path(app.state.db.db_path) == data / "filesearch.db", (
        "前提：测试必须落在临时目录，不能碰真实库（否则污染 ~/.local/share 并断言失败）"
    )
    try:
        yield app, TestClient(app), kb, data
    finally:
        if hasattr(app.state, "embed"):
            try:
                app.state.embed.close()
            except Exception:  # noqa: BLE001
                pass
        for name in ("watcher", "scanner", "recovery"):
            obj = getattr(app.state, name, None)
            stop = getattr(obj, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001
                    pass


def test_progress_endpoint_ok_with_non_utf8_error_detail(app_ctx):
    """明细里带非 UTF-8 路径时，端点必须仍返回 200（这就是用户看到的 500）。"""
    app, client, kb, data = app_ctx
    bad = _make_non_utf8_file(kb)
    scanner = app.state.scanner
    assert os.path.exists(bad), "前提：该路径在文件系统上确实存在"

    scanner._record_error("other", bad, "boom")
    assert scanner._error_logs, "前提：明细已入内存"

    res = client.get("/api/scan/progress")
    assert res.status_code == 200, (
        f"含非 UTF-8 路径的明细不得让端点 500（实际 {res.status_code}）"
    )
    body = res.json()
    assert body["code"] == 0
    # 载荷必须真的能序列化/编码（这就是 500 的触发点）
    json.dumps(body, ensure_ascii=False).encode("utf-8")


def test_progress_keeps_non_utf8_entry_visible(app_ctx):
    """安全化不等于丢弃：那条明细要留在界面上（路径判存在用原路径）。"""
    app, client, kb, data = app_ctx
    bad = _make_non_utf8_file(kb)
    scanner = app.state.scanner
    scanner._record_error("other", bad, "boom")

    details = client.get("/api/scan/progress").json()["data"]["scanPhase"]["errorDetails"]
    assert len(details) == 1, "文件确实存在 → 明细不应被剔除（判存在必须用原路径）"
    assert "\ufffd" in details[0]["path"], "出接口的路径应已安全化"
    details[0]["path"].encode("utf-8")


def test_progress_drops_entry_when_file_really_gone(app_ctx):
    """对照组：文件真的不存在时明细仍要被剔除（旧行为不能因本次修复而丢）。"""
    app, client, kb, data = app_ctx
    bad = _make_non_utf8_file(kb)
    scanner = app.state.scanner
    scanner._record_error("other", bad, "boom")
    os.unlink(os.fsencode(bad))

    details = client.get("/api/scan/progress").json()["data"]["scanPhase"]["errorDetails"]
    assert details == [], "文件已删 → 明细应剔除"


def test_progress_endpoint_ok_with_non_utf8_truncation(app_ctx, monkeypatch):
    """截断明细走的是另一条路（db.truncated_files → files.path），同样要能出接口。

    这里用 monkeypatch 而不是真往库里插：`files.path` 含 surrogate 时 sqlite3 **插入就
    会失败**（要求合法 UTF-8）——所以现实里库里不会出现这种路径，progress() 里对
    truncations 的 display_path 是纵深防御。用替身才能钉住那道防线本身。
    """
    app, client, kb, data = app_ctx
    bad = _make_non_utf8_file(kb)
    db = app.state.db
    monkeypatch.setattr(db, "truncated_files",
                        lambda limit=200: [{"path": bad, "total": 99, "indexed": 1}])

    res = client.get("/api/scan/progress")
    assert res.status_code == 200, f"截断明细含非法路径也不得 500（实际 {res.status_code}）"
    body = res.json()
    json.dumps(body, ensure_ascii=False).encode("utf-8")
    vp = body["data"]["vectorizePhase"]
    assert vp["truncated"] == 1, "计数与明细同源同长"
    assert "\ufffd" in vp["truncationDetails"][0]["path"]


# ─────────────────────────────────────────────────────────────
# 三、审计日志：落盘不得静默失败
# ─────────────────────────────────────────────────────────────

def test_record_error_writes_log_for_non_utf8_path(app_ctx):
    """非 UTF-8 路径必须真的写进 scan_errors.log（此前被 except: pass 吞掉）。"""
    app, client, kb, data = app_ctx
    bad = _make_non_utf8_file(kb)
    app.state.scanner._record_error("other", bad, "surrogates not allowed")

    log = Path(data) / "scan_errors.log"
    assert log.exists(), "审计文件必须被创建（旧代码这里静默失败）"
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "[other]" in text
    assert "\ufffd" in text, "路径已安全化后落盘"


def test_record_error_log_written_when_path_is_valid(app_ctx):
    """合法路径的行为不变（回归）。"""
    app, client, kb, data = app_ctx
    p = kb / "normal.md"
    p.write_text("x", encoding="utf-8")
    app.state.scanner._record_error("parse", str(p), "解析失败")
    text = (Path(data) / "scan_errors.log").read_text(encoding="utf-8", errors="replace")
    assert str(p) in text and "解析失败" in text


# ─────────────────────────────────────────────────────────────
# 四、遍历失败时给出人能看懂的原因
# ─────────────────────────────────────────────────────────────

def test_walk_error_reason_is_readable_for_non_utf8(app_ctx):
    """_enqueue_file 抛 UnicodeEncodeError 时，明细里的原因要能读懂。"""
    app, client, kb, data = app_ctx
    bad = _make_non_utf8_file(kb)
    scanner = app.state.scanner

    try:
        scanner._enqueue_file(bad)
        raise AssertionError("前提不成立：该路径本应因编码失败而入库失败")
    except UnicodeEncodeError as e:
        reason = scanner._explain_walk_error(bad, e)

    assert "深度拷问清单" in reason, f"应给出按常见编码的读法，实际: {reason}"
    reason.encode("utf-8")


def test_walk_error_reason_passthrough_for_other_exceptions(app_ctx):
    """其它异常原样透传（不得被本次改动改口径）。"""
    app, client, kb, data = app_ctx
    scanner = app.state.scanner
    assert scanner._explain_walk_error("/x", PermissionError("nope")) == "nope"


# ─────────────────────────────────────────────────────────────
# 五、watcher 增量路径（inotify）同一条口径
# ─────────────────────────────────────────────────────────────

def test_watcher_mark_failed_explains_non_utf8(app_ctx):
    """watcher 的增量失败明细也要给能读懂的原因（同 scanner 口径）。

    走 inotify 新建事件同样会踩编码：这类路径进不了 sqlite，原始异常只是
    'utf-8' codec can't encode...，界面上用户不知道要改哪个名字。
    """
    app, client, kb, data = app_ctx
    bad = _make_non_utf8_file(kb)
    w = app.state.watcher
    assert w is not None and hasattr(w, "_scanner_ref"), (
        "前提：watcher 需要能记明细（_scanner_ref）"
    )
    err = UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")
    w._mark_failed(bad, err)

    details = app.state.scanner._error_logs
    assert any("深度拷问清单" in d["reason"] for d in details), (
        f"明细里应给出「按 GBK 读是什么」，实际: {[d['reason'] for d in details]}"
    )
