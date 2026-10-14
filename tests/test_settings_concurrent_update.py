"""并发保存设置不得丢更新（2026-10-08，审查 R09）。

原实现：`update()` 在锁外 `_load()`，只在 `_write_all()` 里加锁。两个并发保存各自
读到同一份旧文件、各写各的 → 后写的那次把先写的字段覆盖掉。

实测（两线程分别改 top_k 与 max_depth）：期望两者都落盘，实际只剩 top_k。

修法：锁覆盖整个「读 → 合并 → 写」事务，并把 Lock 换成 RLock（update/reset 持
外层锁时仍要调 `_write_all`，普通 Lock 会自锁）。
"""
from __future__ import annotations

import threading

import pytest
import yaml

from bibit_filesearch.settings import CONFIG_VERSION, SettingsService


def _svc(tmp_path, text: str | None = None) -> SettingsService:
    p = tmp_path / "config.yaml"
    p.write_text(text if text is not None else f"config_version: {CONFIG_VERSION}\n",
                 encoding="utf-8")
    return SettingsService(config_path=p)


def _saved(svc: SettingsService) -> dict:
    """落盘内容（去掉版本号）。"""
    data = yaml.safe_load(svc.config_path().read_text(encoding="utf-8")) or {}
    return {k: v for k, v in data.items() if k != "config_version"}


def test_sequential_updates_keep_both_fields(tmp_path):
    """对照组：串行保存本来就没问题。"""
    svc = _svc(tmp_path)
    svc.update({"top_k": 7})
    svc.update({"max_depth": 6})
    assert _saved(svc) == {"top_k": 7, "max_depth": 6}


def test_concurrent_updates_keep_both_fields(tmp_path, monkeypatch):
    """两个线程各改一个字段 → 两个字段都必须落盘。

    用 Barrier 卡在两次 `_load()` 之间强制交错：修复前两次读都拿到旧内容，
    后写的把先写的覆盖掉。修复后第二次 `_load()` 在锁内、必然看到第一次的结果。
    """
    svc = _svc(tmp_path, f"config_version: {CONFIG_VERSION}\ntop_k: 5\n")

    original_load = svc._load
    # 超时短一点：修复后两次 `_load()` 被锁串行化、barrier 永远集不满，
    # 这里会白等一个超时（修复前则瞬间放行）。3s 足够区分两种情形又不太慢。
    barrier = threading.Barrier(2, timeout=3)
    first_call = {"n": 0}
    lock = threading.Lock()

    def sync_load():
        data = original_load()
        with lock:
            first_call["n"] += 1
            n = first_call["n"]
        if n <= 2:
            # 只卡前两次（即两个线程各自的首次读）——修复后它们不会并发到这里，
            # barrier 会超时并被吃掉；修复前则会真正交错。
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return data

    monkeypatch.setattr(svc, "_load", sync_load)

    errors: list = []

    def worker(key, value):
        try:
            svc.update({key: value})
        except Exception as e:                       # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=("top_k", 7)),
          threading.Thread(target=worker, args=("max_depth", 6))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert not any(t.is_alive() for t in ts), "线程未退出（疑似死锁）"
    assert not errors, f"保存抛异常：{errors}"

    saved = _saved(svc)
    assert saved.get("top_k") == 7 and saved.get("max_depth") == 6, (
        f"并发保存丢更新：落盘 {saved}，应同时含 top_k=7 与 max_depth=6"
    )


def test_concurrent_many_updates_leave_a_consistent_file(tmp_path):
    """多线程各改不同键：文件必须是**完整**的（不缺字段、不出半文件）。"""
    svc = _svc(tmp_path)
    keys = {"top_k": 3, "max_depth": 2, "auto_index": False}
    errors: list = []

    def worker(k, v):
        try:
            svc.update({k: v})
        except Exception as e:                       # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k, v)) for k, v in keys.items()]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert not errors
    saved = _saved(svc)
    for k, v in keys.items():
        assert saved.get(k) == v, f"{k} 丢失：落盘 {saved}"


def test_reset_and_update_do_not_deadlock(tmp_path):
    """锁改成 RLock 之后 reset 仍可嵌套 _write_all（普通 Lock 会自锁）。"""
    svc = _svc(tmp_path, f"config_version: {CONFIG_VERSION}\ntop_k: 9\n")

    done = threading.Event()

    def do_reset():
        svc.reset()
        done.set()

    t = threading.Thread(target=do_reset, daemon=True)
    t.start()
    assert done.wait(timeout=10), "reset() 卡死——锁不可重入？"
    assert _saved(svc) == {}


def test_update_still_validates_and_normalises(tmp_path):
    """并发改动别把既有语义弄丢：类型校验、max_depth 负数归一、未知键忽略。"""
    svc = _svc(tmp_path)
    svc.update({"top_k": "不是数字", "unknown_key": 1, "max_depth": -5})
    saved = _saved(svc)
    assert "top_k" not in saved, "类型不符应被拒绝"
    assert "unknown_key" not in saved, "未知键应被忽略"
    assert saved.get("max_depth") == 0, "负数应归一为 0（无限深）"
