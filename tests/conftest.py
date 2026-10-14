"""全仓测试夹具：隔离**真实数据目录**（2026-09-23）。

## 为什么要有这一条

`create_app()` 的数据目录来自 `main._data_root()`：

    BIBIT_FILESEARCH_DATA_ROOT  env  >  ~/.local/share/bibit-filesearch

注意它**不看 config 里的 storage.db_path** —— 优先级是 env 最高。于是所有用
`create_app` 的测试（t7 / t12 / 健康检查 / 搜索错误路径…）在没设这个 env 时，
会去读写**开发机上真实用户的** `~/.local/share/bibit-filesearch/`：

  · 写入的持久化设置（index_dirs / exclude_patterns / 索引格式 / max_depth）
    会被后来的测试读到 → 「上一个测试的知识库目录」变成「这一个测试的扫描目标」；
  · 反过来，测试也会污染真实用户的服务状态。

## 实测（这不是理论推测）

同一份**未改动的基线代码**（git worktree 到 tag 前）连跑 20 次 `tests/test_t12_settings.py`，
2 次失败，且是两个不同用例：

    [base16] FAILED test_settings_reconcile_excludes_filename_glob
             E assert {...'app.log'...} is None          ← 该回收的没回收
    [base17] FAILED test_settings_reconcile_max_depth_reclaims_deep_files
             E assert None is not None                   ← base 内文件被误删

同一个文件单独跑（默认顺序）19/19 通过。失败只在**与其它测试共处一个共享数据目录**时出现。

改法：给每个测试一个独立的数据目录。用 `monkeypatch.setenv` 而非直接写 `os.environ`，
测试结束后自动还原，测试内部仍可按需覆盖/删除。

## 不隔离 `BIBIT_CONFIG`

`BIBIT_CONFIG` 是**配置**路径、`BIBIT_FILESEARCH_DATA_ROOT` 是**数据**目录，
两者语义独立。`test_config_paths.py` 专门验前者（内含 `monkeypatch.delenv`），
在这里统一改写会与它冲突。故本夹具只碰数据目录——那正是全部实测偶发的来源。
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_data_root(tmp_path_factory, monkeypatch):
    """每个测试一个独立数据目录（真实用户目录概不外泄）。"""
    root = tmp_path_factory.mktemp("bibit_data")
    monkeypatch.setenv("BIBIT_FILESEARCH_DATA_ROOT", str(root))
    return root
