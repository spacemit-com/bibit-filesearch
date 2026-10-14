"""配置升级语义回归测试（2026-09-09，方案 B+C）。

背景：旧版首启把整份出厂模板复制成 ~/.config 副本，把当时的默认值冻结成
「用户选择」。此后出厂默认改进（如 query_instruction 清空）永远触达不了老用户
——用户实测 haha.md 检索 0% 时，包里改了默认但用户文件里还留着旧值，修复失效。

新语义：
1. 用户文件只保存「用户显式改过的键」，其余回落代码 DEFAULTS（单一权威）。
2. 首启不再复制整份模板；首次保存设置时才落盘（含 config_version）。
3. 老文件启动时一次性迁移：删「值 == 出厂默认」的化石键 + 补版本号；
   用户真改过的键原样保留。

覆盖：
- 首启无文件：不创建文件，GET 返回完整默认（含 cors_origins）
- 迁移：化石键删除、用户改动保留、版本号写入
- 迁移幂等：重复构造不重复改写
- 稀疏落盘：保存「值 == 默认」的键 → 不落盘
- 历史默认残留（query_instruction 旧值）→ 删除，新默认生效
- 出厂模板与代码默认值一致（防再次漂移）
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from bibit_filesearch.settings import (
    CONFIG_VERSION,
    SettingsService,
    _factory_defaults,
)


def _cfg(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_first_run_does_not_create_file(tmp_path, monkeypatch):
    """首启不写文件；GET 仍返回完整默认（含 CORS 白名单）。"""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path / "nohome")
    p = tmp_path / "config.yaml"
    svc = SettingsService(config_path=p)
    assert not p.exists(), "首启不应复制出厂模板（复制会把默认值冻结成用户选择）"
    d = svc.get_all()
    assert d["top_k"] == 5
    assert d["auto_index"] is True
    assert "md" in d["indexable_types"]
    # 出厂 CORS 白名单必须来自代码默认，否则删配置后 GUI 跨域挂
    from bibit_filesearch.config import load_config

    monkeypatch.setenv("BIBIT_CONFIG", str(p))
    assert load_config(str(p)).server.cors_origins, "无用户文件时 CORS 白名单不得为空"


def test_migration_removes_fossils_keeps_user_values(tmp_path):
    """老文件迁移：删化石键、保留用户改动、写版本号。"""
    p = _cfg(tmp_path, yaml.safe_dump({
        "index_dirs": ["/home/u/Documents"],      # 用户改过 → 保留
        "indexable_types": ["md", "png"],         # 用户改过 → 保留
        "auto_index": True,                       # == 默认 → 删
        "max_depth": 3,                           # == 默认 → 删
        "top_k": 5,                               # == 默认 → 删
        "exclude_patterns": [".git", "node_modules", "__pycache__"],  # == 默认 → 删
        "server": {"host": "127.0.0.1",
                   "cors_origins": ["http://localhost:5173", "http://127.0.0.1:5173",
                                    "tauri://localhost", "http://tauri.localhost",
                                    "http://localhost"]},  # 全 == 默认 → 删
        "models": {"embed": {"url": "http://127.0.0.1:8082", "dim": 1024,
                             "model": "qwen3-embedding-0.6b", "query_instruction": ""}},
    }, allow_unicode=True))

    svc = SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))

    assert after["index_dirs"] == ["/home/u/Documents"], "用户改过的 index_dirs 必须保留"
    assert after["indexable_types"] == ["md", "png"], "用户改过的 indexable_types 必须保留"
    assert after["config_version"] == CONFIG_VERSION
    for k in ("auto_index", "max_depth", "top_k", "exclude_patterns", "server", "models"):
        assert k not in after, f"化石键 {k} 应被删除（值等于出厂默认）"
    # 生效值不变：删掉的键回落默认
    d = svc.get_all()
    assert d["auto_index"] is True and d["top_k"] == 5 and d["max_depth"] == 3


def test_migration_is_idempotent(tmp_path):
    """已是当前版本 → 不重写文件（避免每次启动改动 mtime / 触发 watcher）。"""
    p = _cfg(tmp_path, yaml.safe_dump(
        {"index_dirs": ["/home/u/Docs"], "config_version": CONFIG_VERSION}, allow_unicode=True))
    before = p.read_text(encoding="utf-8")
    SettingsService(config_path=p)
    assert p.read_text(encoding="utf-8") == before, "已迁移过的文件不得被再次改写"


def test_migration_removes_stale_embed_path_default(tmp_path):
    """v4 → v5：embed 接口定为 /v1/embed/embeddings，旧默认路径必须删除。

    v4 那版出厂默认是 /v1/embed/local/embeddings（早期草案）。用户保存过设置
    就会把它冻进文件，升级后仍打这条已被上游移除的路径 → 一直 404，表现为
    「向量化服务未启动」。只删「值等于旧默认」的键：用户自己填过的路径保留。
    """
    p = _cfg(tmp_path, yaml.safe_dump({
        "index_dirs": ["/home/u/Docs"],
        "config_version": 4,
        "models": {"embed": {"path": "/v1/embed/local/embeddings"}},
    }, allow_unicode=True))
    SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after["config_version"] == CONFIG_VERSION
    assert "models" not in after, f"旧路径默认应被删除，实际残留 {after.get('models')}"


def test_migration_keeps_user_chosen_embed_path(tmp_path):
    """用户自己指向过的路径（不等于任何历史默认）不得被误删。"""
    p = _cfg(tmp_path, yaml.safe_dump({
        "index_dirs": ["/home/u/Docs"],
        "config_version": 4,
        "models": {"embed": {"path": "/my/own/embedding/endpoint"}},
    }, allow_unicode=True))
    SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after["models"]["embed"]["path"] == "/my/own/embedding/endpoint", \
        f"用户自选路径被误删：{after}"


def test_migration_removes_stale_historical_default(tmp_path):
    """历史默认残留：query_instruction 旧值必须删除，让新默认（空）生效。

    这是用户实测 haha.md 检索 0% 的直接成因：包里改了默认，老用户文件里的
    旧值把新默认盖住了。
    """
    p = _cfg(tmp_path, yaml.safe_dump({
        "index_dirs": ["/home/u/Docs"],
        "models": {"embed": {"query_instruction": "为这个句子生成表示以用于检索相关文章"}},
    }, allow_unicode=True))
    SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "models" not in after, "旧默认值残留应被删除（models.embed 清空后不留空父层）"

    from bibit_filesearch.config import load_config

    cfg = load_config(str(p))
    assert cfg.models.embed.get("query_instruction", "") == "", "迁移后应回落到新默认（空）"


def test_migration_keeps_custom_instruction(tmp_path):
    """用户自定义的 instruction 不是化石，必须保留（对照组，防误删）。"""
    p = _cfg(tmp_path, yaml.safe_dump({
        "models": {"embed": {"query_instruction": "我的自定义任务描述"}},
    }, allow_unicode=True))
    SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after["models"]["embed"]["query_instruction"] == "我的自定义任务描述"


def test_migration_upgrades_legacy_indexable_types(tmp_path):
    """老版出厂模板冻结的 18 类列表（不含 png/jpg）应被识别为化石删除。

    2026-09-09：png/jpg 纳入出厂默认后，老用户文件里的旧 18 类列表会盖住
    新默认 → 图片文字永远搜不到。该值登记在 _CHANGED_DEFAULTS。
    """
    legacy = ["txt", "md", "docx", "pdf", "py", "js", "java", "cpp", "c", "cs",
              "go", "rs", "ts", "sql", "sh", "json", "xlsx", "pptx"]
    p = _cfg(tmp_path, yaml.safe_dump(
        {"index_dirs": ["/home/u/Docs"], "indexable_types": legacy}, allow_unicode=True))
    svc = SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "indexable_types" not in after, "旧 18 类列表是化石，应删除让新默认生效"
    assert "png" in svc.get("indexable_types") and "jpg" in svc.get("indexable_types"), \
        "迁移后应回落新默认（含 png/jpg）"


def test_migration_keeps_custom_indexable_types(tmp_path):
    """用户真改过的格式列表（不等于旧默认）必须保留——对照组，防误删。"""
    p = _cfg(tmp_path, yaml.safe_dump(
        {"indexable_types": ["md", "png"]}, allow_unicode=True))
    svc = SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after["indexable_types"] == ["md", "png"], "用户自定义格式列表不得被删"


def test_update_writes_sparsely(tmp_path):
    """保存设置时「值 == 出厂默认」的键不落盘，只写用户改过的。"""
    p = _cfg(tmp_path, "{}\n")
    svc = SettingsService(config_path=p)
    svc.update({
        "index_dirs": ["/home/u/Docs"],   # 非默认 → 落盘
        "top_k": 5,                       # == 默认 → 不落盘
        "auto_index": True,               # == 默认 → 不落盘
    })
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after["index_dirs"] == ["/home/u/Docs"]
    assert "top_k" not in after and "auto_index" not in after
    assert after["config_version"] == CONFIG_VERSION


def test_update_resets_to_default_removes_key(tmp_path):
    """用户把某键改回默认值 → 键从文件里消失（等价于「恢复默认」）。"""
    p = _cfg(tmp_path, yaml.safe_dump(
        {"top_k": 50, "config_version": CONFIG_VERSION}, allow_unicode=True))
    svc = SettingsService(config_path=p)
    svc.update({"top_k": 5})
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "top_k" not in after, "改回默认值应删键（稀疏存储）"
    assert svc.get("top_k") == 5, "生效值仍是默认 5"


def test_migration_v6_removes_fossil_top_k(tmp_path):
    """v6 → v7：top_k 出厂默认 10 → 5，用户文件里冻着的 10 是化石必须删除。

    与 max_depth/indexable_types 同理：不删的话老用户永远停在 10，
    包里改的新默认对他们不生效。
    """
    p = _cfg(tmp_path, yaml.safe_dump(
        {"top_k": 10, "index_dirs": ["/home/u/Docs"],
         "config_version": 6}, allow_unicode=True))
    svc = SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "top_k" not in after, "旧默认 10 是化石，应删除让新默认 5 生效"
    assert svc.get("top_k") == 5, "迁移后生效值应为新默认 5"


def test_migration_keeps_custom_top_k(tmp_path):
    """用户真调过的 top_k（不等于旧默认 10）必须保留——对照组，防误删。"""
    p = _cfg(tmp_path, yaml.safe_dump(
        {"top_k": 33, "config_version": 6}, allow_unicode=True))
    svc = SettingsService(config_path=p)
    after = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert after["top_k"] == 33, f"用户设定 33 不得被迁移删掉：{after}"
    assert svc.get("top_k") == 33


def test_defaults_have_single_source():
    """出厂默认只有一处权威：/opt 模板必须与代码默认值一致（防再次漂移）。"""
    defaults = _factory_defaults()
    tpl_path = pathlib.Path("/opt/bibit-filesearch/config.yaml")
    if not tpl_path.exists():
        pytest.skip("未安装 deb（无 /opt 出厂模板）")
    tpl = yaml.safe_load(tpl_path.read_text(encoding="utf-8"))

    # 模板是「人类可读参考」，只需覆盖 settings 的 6 个用户键 + server/models
    for k in ("index_dirs", "auto_index", "indexable_types", "exclude_patterns",
              "max_depth", "top_k"):
        assert tpl[k] == defaults[k], f"模板 {k} 与代码默认不一致"
    assert tpl["server"]["cors_origins"] == defaults["server"]["cors_origins"], \
        "CORS 白名单漂移会导致删配置后 GUI 跨域挂（2026-09-07 踩过）"
    assert tpl["models"]["embed"]["dim"] == defaults["models"]["embed"]["dim"]
    assert tpl["models"]["embed"]["query_instruction"] == \
        defaults["models"]["embed"]["query_instruction"]
