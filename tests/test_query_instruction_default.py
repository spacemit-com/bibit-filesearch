"""query_instruction 默认必须为空（2026-09-09 用户实测：haha.md 检索显示 0%）。

现象：向 haha.md 写入 abcdef，检索 abcdef 时 haha.md conf=0.0 排最后。
根因：查询侧 EmbedClient.embed_one 加 "Instruct: <描述>\nQuery: <query>" 前缀，
索引侧 embed_batch 从不加 → 两侧不对称；而 :8082 以 --pooling mean 启动，
前缀 token 远多于 query，mean 后向量被指令文本主导（cos(带前缀, 纯指令)=0.93），
短查询命中块余弦从 0.98 掉到 0.35 → 跌出 top_candidates → embed 分为 0，
仅剩 BM25 分，min-max 归一化后显示 0%。

断言：出厂默认与 DEFAULTS 的 query_instruction 均为空；EmbedClient 空值时
发出的 payload 与索引侧完全一致（裸文本）。
"""
import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def test_defaults_query_instruction_empty():
    """DEFAULTS 里 query_instruction 必须为空串。"""
    from bibit_filesearch.config import DEFAULTS

    instr = DEFAULTS["models"]["embed"]["query_instruction"]
    assert instr == "", (
        f"DEFAULTS query_instruction={instr!r} 应为空串——非空时查询侧加前缀、"
        "索引侧不加，mean pooling 下短查询会被指令文本淹没（haha.md 0% 实测）"
    )


def test_packaging_config_query_instruction_empty():
    """出厂模板 packaging/config.yaml 必须与 DEFAULTS 一致（留空）。"""
    p = REPO / "packaging" / "config.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    instr = cfg["models"]["embed"].get("query_instruction")
    assert instr in (None, ""), (
        f"packaging/config.yaml query_instruction={instr!r} 应为空——"
        "该文件是 deb 出厂模板，用户首次启动即复制成 ~/.config 副本"
    )


def test_embed_client_payload_symmetric_when_empty(monkeypatch):
    """query_instruction 为空时，embed_one 发出的 payload == 裸文本（与索引侧对称）。"""
    from bibit_filesearch.model_client import EmbedClient

    sent: list[list[str]] = []
    c = EmbedClient("http://127.0.0.1:9", dim=4, model="m", query_instruction="")

    def fake_batch(texts):
        sent.append(list(texts))
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(c, "embed_batch", fake_batch)
    c.embed_one("abcdef")
    assert sent == [["abcdef"]], (
        f"embed_one 实际发出 {sent}——query_instruction 为空时不得加任何前缀，"
        "否则查询向量与索引向量（embed_batch 裸文本）不在同一语义空间"
    )


def test_embed_client_adds_prefix_when_configured(monkeypatch):
    """对照组：显式配置了前缀时仍按官方模板拼接（防「永不生效」假修复）。"""
    from bibit_filesearch.model_client import EmbedClient

    sent: list[list[str]] = []
    c = EmbedClient("http://127.0.0.1:9", dim=4, model="m", query_instruction="任务描述")

    def fake_batch(texts):
        sent.append(list(texts))
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(c, "embed_batch", fake_batch)
    c.embed_one("abcdef")
    assert sent == [["Instruct: 任务描述\nQuery: abcdef"]], (
        "显式配置前缀时行为应保持不变（该能力仍保留，供 last-token pooling 部署使用）"
    )


def test_loaded_config_default_empty(monkeypatch, tmp_path):
    """无用户配置时 load_config 的 embed 指令前缀为空。"""
    monkeypatch.delenv("BIBIT_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    from bibit_filesearch.config import load_config

    cfg = load_config()
    assert cfg.models.embed.get("query_instruction", "") == ""
