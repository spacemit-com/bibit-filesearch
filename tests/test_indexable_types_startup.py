"""indexable_types 必须在 scanner.start() 之前生效（2026-09-09 用户实测暴露）。

现象：配置勾选了 png，11 个文件只有 6 个 completed，ocr-test.png 被标 unsupported。
根因：main.create_app 里 `indexer.indexable_types = set(types)` 在 `scanner.start()`
之后——遍历线程已开跑，用 Indexer 构造默认值（DEFAULT_INDEXABLE_TYPES）判定类型，
窗口内的文件被误标 unsupported。窗口 ≈ recovery.start() 首次同步探测耗时。

断言：scanner.start() 被调用时，indexer.indexable_types 已等于运行时配置值。
"""
import os

import pytest
import yaml


def _make_app(tmp_path, indexable_types):
    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 甲\n\n内容。\n", encoding="utf-8")
    cfg = {
        "storage": {"db_path": str(tmp / "search.db")},
        "server": {"port": 18097, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "auto_index": False,
        "indexable_types": indexable_types,
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    os.environ["BIBIT_FILESEARCH_DATA_ROOT"] = str(tmp)

    from bibit_filesearch.main import create_app

    return create_app(str(cfg_path))


def test_indexable_types_applied_before_scanner_start(tmp_path, monkeypatch):
    """scanner.start() 那一刻 indexable_types 必须已是配置值（非构造默认）。"""
    from bibit_filesearch.scanner import ScanService

    seen: list[set[str]] = []
    orig_start = ScanService.start

    def spy_start(self):
        seen.append(set(self.indexer.indexable_types))
        return orig_start(self)

    monkeypatch.setattr(ScanService, "start", spy_start)

    app = _make_app(tmp_path, ["md", "png"])
    try:
        assert seen, "scanner.start() 未被调用——测试前提不成立"
        assert seen[0] == {"md", "png"}, (
            f"scanner.start() 时 indexable_types={sorted(seen[0])}，"
            "应为 {'md','png'}（配置值）——赋值必须早于遍历线程启动，"
            "否则窗口内的文件被按构造默认值误判 unsupported"
        )
    finally:
        app.state.watcher.stop()
        app.state.scanner._status = "idle"


def test_indexable_types_default_when_unset(tmp_path, monkeypatch):
    """配置未给 indexable_types 时，保持 Indexer 构造默认（不误清空）。"""
    from bibit_filesearch.indexer import DEFAULT_INDEXABLE_TYPES
    from bibit_filesearch.scanner import ScanService

    seen: list[set[str]] = []
    orig_start = ScanService.start

    def spy_start(self):
        seen.append(set(self.indexer.indexable_types))
        return orig_start(self)

    monkeypatch.setattr(ScanService, "start", spy_start)

    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 甲\n", encoding="utf-8")
    cfg = {
        "storage": {"db_path": str(tmp / "search.db")},
        "server": {"port": 18096, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "auto_index": False,
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    os.environ["BIBIT_FILESEARCH_DATA_ROOT"] = str(tmp)

    from bibit_filesearch.main import create_app

    app = create_app(str(cfg_path))
    try:
        assert seen[0] == set(DEFAULT_INDEXABLE_TYPES), (
            "未配置 indexable_types 时应保持构造默认值"
        )
    finally:
        app.state.watcher.stop()
        app.state.scanner._status = "idle"


@pytest.mark.parametrize("types", [["md"], ["md", "png"], []])
def test_indexable_types_empty_or_subset(tmp_path, monkeypatch, types):
    """空列表是有效意图（取消全部格式）——不应因 falsy 被忽略而回落默认。

    2026-10-08（审查 R02）：本用例原先把空列表的旧行为写成断言
    （`assert seen[0]`，即「空列表回落构造默认」），把 bug 固化成预期。
    修 `main.create_app` 的判据为 `is not None` 后，空列表必须原样生效。
    """
    from bibit_filesearch.scanner import ScanService

    seen: list[set[str]] = []
    orig_start = ScanService.start

    def spy_start(self):
        seen.append(set(self.indexer.indexable_types))
        return orig_start(self)

    monkeypatch.setattr(ScanService, "start", spy_start)

    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 甲\n", encoding="utf-8")
    cfg = {
        "storage": {"db_path": str(tmp / "search.db")},
        "server": {"port": 18095, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "auto_index": False,
        "indexable_types": types,
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    os.environ["BIBIT_FILESEARCH_DATA_ROOT"] = str(tmp)

    from bibit_filesearch.main import create_app

    app = create_app(str(cfg_path))
    try:
        assert seen[0] == set(types), (
            f"indexable_types 配置为 {types!r}，scanner.start() 时实为 "
            f"{sorted(seen[0])}——空列表是「取消全部格式」的明确意图，不得回落默认"
        )
    finally:
        app.state.watcher.stop()
        app.state.scanner._status = "idle"


def test_indexable_types_empty_keeps_file_out_of_the_index(tmp_path, monkeypatch):
    """端到端：全部格式取消后，重启时不再把 txt 索引成 completed。

    这是用户能看见的那一面——设置页显示全部取消，重启后文件仍能被搜到。
    仅断言 `indexer.indexable_types == set()` 不够：要证明文件的**内容索引**
    没有落库（vectorize_status 不是 completed / 无块）。
    """
    from unittest.mock import patch

    from bibit_filesearch.main import create_app

    tmp = tmp_path / "data"
    tmp.mkdir()
    kb = tmp / "kb"
    kb.mkdir()
    victim = kb / "should-not-be-indexed.txt"
    victim.write_text("这段内容不该进入索引。\n", encoding="utf-8")

    cfg = {
        "storage": {"db_path": str(tmp / "search.db")},
        "server": {"port": 18094, "cors_origins": []},
        "index_dirs": [str(kb)],
        "max_depth": 0,
        "auto_index": False,
        "indexable_types": [],
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    os.environ["BIBIT_CONFIG"] = str(cfg_path)
    os.environ["BIBIT_FILESEARCH_DATA_ROOT"] = str(tmp)

    with patch("bibit_filesearch.scanner.ScanService.start"), \
         patch("bibit_filesearch.recovery.EmbedRecovery.start"), \
         patch("bibit_filesearch.retriever.Retriever.warmup"), \
         patch("bibit_filesearch.model_client.EmbedClient.probe_dim", return_value=1024), \
         patch("bibit_filesearch.model_client.EmbedClient.embed_batch",
               side_effect=lambda ts: [[1.0] + [0.0] * 1023 for _ in ts]):
        app = create_app(str(cfg_path))

    try:
        # 走池子的正常入口：放进去 → 取活 → 处理
        app.state.pool.offer(str(victim))
        row = app.state.pool.take()
        if row is not None:
            app.state.scanner.worker._run_one(row)
        rec = app.state.db.get_file_by_path(str(victim))
        assert rec is not None, "文件至少要落元数据（仅登记，不进内容索引）"
        assert rec["vectorize_status"] != "completed", (
            f"全部格式取消后 txt 仍被索引为 {rec['vectorize_status']}——"
            "indexable_types=[] 必须在启动时就生效"
        )
        chunks = app.state.db.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_id=?", (rec["file_id"],)).fetchone()[0]
        assert chunks == 0, f"不应产生内容块，实际 {chunks} 块"
    finally:
        app.state.watcher.stop()
        app.state.scanner._status = "idle"
        app.state.embed.close()
