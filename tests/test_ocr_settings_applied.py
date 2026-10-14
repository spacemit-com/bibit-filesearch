"""OCR 配置必须在引擎构造**之前**生效（2026-10-08，审查 R03）。

现象：设置 `ocr.device: cpu` 或自定义 `ocr.max_side_len` 后重启服务，实际仍跑
NPU / 1400。原因在生产启动顺序：

    main()
      → ocr_selfcheck()      构造并缓存 RapidOCR（_PPOCR_TRIED = True）
      → uvicorn → create_app()  这时才写 OCR_DEVICE / MAX_SIDE_LEN

引擎已经建好了，后写的全局量只影响「下次构造」——而下次永远不会来（进程内只建
一次）。`/api/meta` 会如实报出那个陈旧值，所以现象是「配置看着生效、行为没变」。

修法两条：
  ① `ocr_engine.apply_settings(ocr_cfg)` 是唯一的赋值入口，create_app 与 main()
     共用；main() 在自检**之前**调用。
  ② `_PPOCR_BUILT` 记住构造时的 (设备, 长边)；apply_settings 发现与当前配置不符
     就丢弃缓存实例，下次按新参数重建（否则「改配置重启」也无效）。
"""
from __future__ import annotations

import sys
import types

import pytest

from bibit_filesearch import ocr_engine as oe


@pytest.fixture(autouse=True)
def _clean_engine():
    """每个用例前后把模块级引擎状态复位（这些是全局量，会互相污染）。"""
    def reset():
        oe._PPOCR = None
        oe._PPOCR_TRIED = False
        oe._PPOCR_ERROR = ""
        oe._PPOCR_DEVICE = ""
        oe._PPOCR_BUILT = None
        oe.OCR_DEVICE = "auto"
        oe.MAX_SIDE_LEN = 1400
    reset()
    yield
    reset()


@pytest.fixture()
def fake_rapidocr(monkeypatch):
    """替身 RapidOCR：记录每次构造的 params。NPU 装载固定成功（设备路径可控）。"""
    built: list[dict] = []

    class _RapidOCR:
        def __init__(self, **kw):
            built.append(kw)

    monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=_RapidOCR))
    monkeypatch.setattr(oe, "_LOCK", oe._LOCK)          # 复用真锁，无副作用
    import bibit_filesearch.ocr_ep as ep
    monkeypatch.setattr(ep, "patch_rapidocr", lambda: (True, "mock"))
    return built


# ── ① apply_settings 是赋值入口 ────────────────────────────────────────────────

def test_apply_settings_sets_globals():
    oe.apply_settings({"device": "cpu", "max_side_len": 777})
    assert oe.OCR_DEVICE == "cpu"
    assert oe.MAX_SIDE_LEN == 777


def test_apply_settings_defaults_and_bad_values():
    """缺省 / 非法值都回落默认，且不抛（配置坏掉不能让服务起不来）。"""
    oe.apply_settings({})
    assert (oe.OCR_DEVICE, oe.MAX_SIDE_LEN) == ("auto", 1400)

    oe.apply_settings({"device": "gpu", "max_side_len": "不合法"})
    assert oe.OCR_DEVICE == "auto", "非法设备名应回落 auto"
    assert oe.MAX_SIDE_LEN == 1400, "非法长边应回落 1400"


def test_apply_settings_pushes_ep_and_parser_knobs():
    """同一份配置还要送达 ocr_ep / parser 的那几个上限（原先散在 create_app）。"""
    import bibit_filesearch.ocr_ep as ep
    import bibit_filesearch.parser as parser

    oe.apply_settings({"pdf_max_ocr_pages": 3, "worker_max_runs": 5,
                       "worker_max_rss_mb": 900, "idle_release_secs": 7})
    assert parser.PDF_MAX_OCR_PAGES == 3
    assert ep.WORKER_MAX_RUNS == 5
    assert ep.WORKER_MAX_RSS_MB == 900
    assert ep.WORKER_IDLE_RELEASE_SECS == 7


# ── ② 配置变了要重建引擎 ──────────────────────────────────────────────────────

def test_engine_is_rebuilt_when_settings_change(fake_rapidocr):
    """先按默认构造，再改配置 → 下一个 _try_ppocr 必须按新参数重建。"""
    oe.apply_settings({"device": "auto", "max_side_len": 1400})
    first = oe._try_ppocr()
    assert first is not None and fake_rapidocr == [{"params": {"Global.max_side_len": 1400}}]

    oe.apply_settings({"device": "cpu", "max_side_len": 777})
    assert oe._PPOCR is None, "配置变了必须丢弃已构造的引擎，否则新参数永远不生效"

    second = oe._try_ppocr()
    assert second is not first, "应重新构造，而不是复用旧实例"
    assert fake_rapidocr[-1] == {"params": {"Global.max_side_len": 777}}, \
        f"重建时未用新长边：{fake_rapidocr}"


def test_engine_is_not_rebuilt_when_settings_unchanged(fake_rapidocr):
    """配置没变就不重建（幂等）——否则每次 create_app 都要重跑一次构造。"""
    oe.apply_settings({"device": "auto", "max_side_len": 1400})
    eng = oe._try_ppocr()
    n = len(fake_rapidocr)

    oe.apply_settings({"device": "auto", "max_side_len": 1400})

    assert oe._PPOCR is eng, "同样的配置不应丢缓存"
    assert len(fake_rapidocr) == n, "同样的配置不应重新构造"


def test_apply_settings_after_engine_keeps_effective_device(fake_rapidocr):
    """重建设备要看新配置：cpu 不能再打 NPU 补丁。"""
    oe.apply_settings({"device": "auto", "max_side_len": 1400})
    oe._try_ppocr()
    assert oe._PPOCR_DEVICE == "npu"

    oe.apply_settings({"device": "cpu", "max_side_len": 1400})
    oe._try_ppocr()
    assert oe._PPOCR_DEVICE == "cpu", "ocr.device=cpu 重新构造后不应再走 NPU"


# ── ③ 启动顺序：main() 必须在自检前应用配置 ───────────────────────────────────

def test_main_applies_ocr_settings_before_selfcheck(tmp_path, monkeypatch):
    """生产路径的核心断言：自检被调用时，OCR 全局量已经是用户配置值。

    这一条直接抓原 bug——修复前 OcrSelfcheck 看到的是模块默认（auto/1400），
    用户配置（cpu/777）要等 create_app 才写，而那时引擎早已构造完毕。
    """
    import os
    from unittest.mock import patch

    import yaml

    from bibit_filesearch import main as main_mod
    from bibit_filesearch.settings import CONFIG_VERSION

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "config_version": CONFIG_VERSION,
        "ocr": {"device": "cpu", "max_side_len": 777},
    }, allow_unicode=True), encoding="utf-8")

    seen: list[tuple] = []

    def _spy_selfcheck():
        seen.append((oe.OCR_DEVICE, oe.MAX_SIDE_LEN))

    monkeypatch.setenv("BIBIT_FILESEARCH_DATA_ROOT", str(tmp_path / "data"))
    # 拦下 uvicorn：本用例只关心「自检被调用时 OCR 配置处于什么状态」，不起服务。
    # 同时兜住两种启动形式（`uvicorn.run` 与 Config/Server），免得本用例绑死在
    # 某一版的启动接线上。
    with patch.object(sys, "argv", ["x", "serve", "--config", str(cfg)]), \
         patch.object(main_mod, "ocr_selfcheck", _spy_selfcheck), \
         patch.object(main_mod, "create_app", lambda *a, **k: object()), \
         patch.object(main_mod.uvicorn, "run", lambda *a, **k: None), \
         patch.object(main_mod.uvicorn, "Config", lambda app, **kw: app), \
         patch.object(main_mod.uvicorn, "Server",
                      lambda cfg: type("S", (), {"run": lambda self: None})()):
        main_mod.main()

    assert seen, "自检未被调用——测试前提不成立"
    assert seen[0] == ("cpu", 777), (
        f"自检开始时 OCR 配置为 {seen[0]}，应为 ('cpu', 777)——"
        "配置必须在自检（首次构造引擎）之前生效，否则整轮跑默认值"
    )
