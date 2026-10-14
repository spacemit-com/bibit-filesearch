"""未纳入索引：明细（数 + 明细同源）+ 深度剪枝也记账 的回归测试（2026-09-23 第二版）。

守三件事：
  ① 「未纳入索引」不只给数，还给**明细**（用户能追查「是哪几个」）——此前只有计数，
     界面无法提供「查看未纳入索引文件」入口。
  ② 计数与明细同源：total 是全量真值；details 是样本，超限时 detailsTruncated=true
     （不许出现「格上写 800、展开 200 条」却不说一声）。
  ③ 超出索引深度被剪掉的文件**也要计入**（提示文案早就把「超出索引深度」列为原因，
     此前一条都不记 → 理由给了、账对不上）。
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
    """确定性 embed 替身。"""

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
    """ScanService 的最小配置替身（`scan` 读 .get，故用 dict）。"""

    def __init__(self, dirs, excludes=None, max_depth=0, indexable_types=None):
        self.index_dirs = dirs
        self.exclude_patterns = list(excludes or [])
        self.indexable_types = list(indexable_types or ["txt", "md", "pdf"])
        self.scan = {"io_throttle_ms": 0, "max_depth": max_depth, "max_attempts": 3}


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp(prefix="bibit_notidx_"))
    db = Database(str(tmp / "test.db"))
    embed = _Embed()
    yield tmp, db, embed
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


def _scanner(db, embed, dirs, excludes=None, max_depth=0, types=None):
    idx = Indexer(db, embed, indexable_types=set(types or ["txt", "md", "pdf"]))
    return ScanService(_Cfg(dirs, excludes, max_depth, types), db, idx)


def _phase(s):
    return s.progress()["notIndexedPhase"]


def test_excluded_details_list_which_files(env):
    """命中排除规则的文件：计数与明细都要有，且明细能指到具体路径。"""
    tmp, db, embed = env
    kb = tmp / "kb"
    (kb / "#recycle").mkdir(parents=True)
    (kb / "#recycle" / "junk1.txt").write_text("a\n", encoding="utf-8")
    (kb / "#recycle" / "junk2.txt").write_text("b\n", encoding="utf-8")
    (kb / "keep.md").write_text("kept\n", encoding="utf-8")

    s = _scanner(db, embed, [str(kb)], excludes=["#recycle"])
    s.start()
    try:
        assert _wait(lambda: _phase(s)["total"] >= 2), \
            "被剪掉的目录里的文件必须计入未纳入索引"
        ph = _phase(s)
        paths = [d["path"] for d in ph["details"]]
        assert any("#recycle" in p for p in paths), \
            "明细必须能指到被剪掉的目录/文件，否则用户无从追查；实际: %s" % paths
        assert ph["detailsTruncated"] is False, "两条明细不该被标成截断"
        assert ph["byKind"].get("rule", 0) >= 2, ph["byKind"]
        # 被排除的文件不该有库行（连池行都没有）
        assert db.get_file_by_path(str(kb / "#recycle" / "junk1.txt")) is None
    finally:
        s.stop()


def test_depth_cut_counts_toward_not_indexed(env):
    """超出 max_depth 的文件也要计入未纳入索引（kind='depth'）。"""
    tmp, db, embed = env
    kb = tmp / "kb"
    (kb / "lvl1" / "lvl2").mkdir(parents=True)
    (kb / "root.txt").write_text("root\n", encoding="utf-8")
    (kb / "lvl1" / "one.txt").write_text("one\n", encoding="utf-8")
    (kb / "lvl1" / "lvl2" / "two.txt").write_text("two\n", encoding="utf-8")

    # max_depth=2 → base + 下 1 层；lvl2 及其文件越界
    s = _scanner(db, embed, [str(kb)], max_depth=2)
    s.start()
    try:
        assert _wait(lambda: _phase(s)["byKind"].get("depth", 0) >= 1, timeout=15), \
            "超出索引深度的文件必须被计数，且原因标为 depth；实际 %s" % _phase(s)["byKind"]
        ph = _phase(s)
        assert any("two.txt" in d["path"] for d in ph["details"]), \
            "明细里应能查到被深度挡下的那个文件；实际 %s" % ph["details"]
        # 深度内的文件照常入库
        assert _wait(lambda: db.get_file_by_path(str(kb / "lvl1" / "one.txt")) is not None)
    finally:
        s.stop()


def test_not_indexed_total_matches_details_when_small(env):
    """明细充足时：total 必须等于明细条数（数 + 明细同源）。"""
    tmp, db, embed = env
    kb = tmp / "kb"
    (kb / "skip").mkdir(parents=True)
    (kb / "skip" / "a.txt").write_text("a\n", encoding="utf-8")
    (kb / "ok.md").write_text("ok\n", encoding="utf-8")

    s = _scanner(db, embed, [str(kb)], excludes=["skip"])
    s.start()
    try:
        assert _wait(lambda: _phase(s)["total"] == 1)
        ph = _phase(s)
        assert ph["detailsTruncated"] is False, ph
        assert len(ph["details"]) == ph["total"], ph
    finally:
        s.stop()


def test_unsupported_files_appear_in_details(env):
    """库里 unsupported 的文件（格式没勾）也要出现在「未纳入索引」明细里。"""
    tmp, db, embed = env
    kb = tmp / "kb"
    kb.mkdir(parents=True)
    (kb / "note.md").write_text("hello\n", encoding="utf-8")
    (kb / "data.xyz").write_text("???\n", encoding="utf-8")

    s = _scanner(db, embed, [str(kb)])
    s.start()
    try:
        assert _wait(lambda: _phase(s)["total"] >= 1, timeout=20), \
            "格式未勾选的文件必须出现在未纳入索引里"
        ph = _phase(s)
        assert ph["byKind"].get("type", 0) >= 1, ph["byKind"]
        assert any("data.xyz" in d["path"] for d in ph["details"]), ph["details"]
    finally:
        s.stop()


def test_reset_clears_details_between_rounds(env):
    """新一轮扫描必须清掉上一轮的明细（否则界面挂着上一轮的条目）。"""
    tmp, db, embed = env
    kb = tmp / "kb"
    (kb / "skip").mkdir(parents=True)
    (kb / "skip" / "a.txt").write_text("a\n", encoding="utf-8")

    s = _scanner(db, embed, [str(kb)], excludes=["skip"])
    s.start()
    assert _wait(lambda: _phase(s)["total"] >= 1)
    s.stop()

    # 去掉排除规则后重启：那个文件应被正常索引，且上一轮明细不再挂着
    s.excludes = []
    s.restart()
    try:
        assert _wait(lambda: db.get_file_by_path(str(kb / "skip" / "a.txt")) is not None,
                     timeout=15), "去掉排除规则后，该文件应被正常索引"
        assert _phase(s)["total"] == 0, \
            "上一轮的未纳入索引明细必须被清掉；实际 %s" % _phase(s)
    finally:
        s.stop()


def test_no_text_counts_toward_total(env):
    """★ 2026-10-09 回归：库侧「解析了但没文本」（noText）必须计入 notIndexedPhase.total。

    用户实测（取消勾选 md 后）：格上写 2、点开明细 3 条 —— 差值恰好是那 1 个 noText
    （一个只含内嵌图、无文字的 docx）。根因是 total 写作 `excluded + not_indexable`，
    漏了 no_text，而**明细**是三者相加（excluded_details + unsupported_details，
    后者含 noText 与 type）。同文件的 `test_not_indexed_total_matches_details_when_small`
    钉的正是「total == 明细条数」这条契约，但它用排除规则造样本、只盖了 excluded
    分支，没盖到 noText，才让这个漏项溜过。

    本测试同时造出 noText 与 type 两类，直接钉住 total == len(details)。
    """
    tmp, db, embed = env
    kb = tmp / "kb"
    kb.mkdir(parents=True)
    # noText：类型勾了（txt）、解析成功、但读不出文字 → unsupported(noText)
    (kb / "blank.txt").write_text("   \n", encoding="utf-8")
    # type：类型没勾（md）→ unsupported(type)
    (kb / "note.md").write_text("hello\n", encoding="utf-8")

    s = _scanner(db, embed, [str(kb)], types=["txt", "pdf"])
    s.start()
    try:
        assert _wait(lambda: _phase(s)["byKind"].get("noText", 0) >= 1, timeout=20), \
            "解析成功但无文本的文件必须被记为 noText；实际 %s" % _phase(s)["byKind"]
        ph = _phase(s)
        assert ph["byKind"].get("type", 0) >= 1, ph["byKind"]
        assert ph["total"] == len(ph["details"]), \
            "total 必须等于明细条数（含 noText）；实际 total=%s details=%s byKind=%s" % (
                ph["total"], len(ph["details"]), ph["byKind"])
        assert ph["total"] == 2, ph
    finally:
        s.stop()


def test_excluded_tree_counts_nested_files(env):
    """★ 2026-10-09 回归：被剪掉的目录，计数要含**其子目录里**的文件。

    用户实测：`test/excluded/` 下 12 个文件，界面「未纳入索引」只报 7 ——
    差值正好是 `excluded/picture/` 里的 5 个。根因是 `_note_excluded_tree` 当初
    只 scandir 一层（旧注释自认「已知下界：更深的子树不遍历」）。既然 `_walk`
    剪掉的是**整棵**子树，计数就必须是整棵子树的文件数，否则「未纳入索引」这个
    用户最常问的数永远小于磁盘真实。
    """
    tmp, db, embed = env
    kb = tmp / "kb"
    # 被剪掉的目录里放：2 个本层文件 + 一层子目录 2 个 + 两层子目录 1 个
    (kb / "#recycle" / "pic").mkdir(parents=True)
    (kb / "#recycle" / "pic" / "deep").mkdir(parents=True)
    (kb / "#recycle" / "junk1.txt").write_text("a\n", encoding="utf-8")
    (kb / "#recycle" / "junk2.txt").write_text("b\n", encoding="utf-8")
    (kb / "#recycle" / "pic" / "shot1.png").write_text("c\n", encoding="utf-8")
    (kb / "#recycle" / "pic" / "shot2.png").write_text("d\n", encoding="utf-8")
    (kb / "#recycle" / "pic" / "deep" / "far.txt").write_text("e\n", encoding="utf-8")
    (kb / "keep.md").write_text("kept\n", encoding="utf-8")

    s = _scanner(db, embed, [str(kb)], excludes=["#recycle"])
    s.start()
    try:
        assert _wait(lambda: _phase(s)["total"] >= 5, timeout=15), \
            "被剪目录的计数必须含其子目录里的文件（本层 2 + pic 2 + pic/deep 1 = 5）；" \
            "实际 %s" % _phase(s)["total"]
        ph = _phase(s)
        assert ph["total"] == 5, "实际 %s" % ph["total"]
        paths = [d["path"] for d in ph["details"]]
        assert any("deep" in p and "far.txt" in p for p in paths), \
            "两层深的文件也要出现在明细里；实际 %s" % paths
        assert ph["detailsTruncated"] is False, "5 条明细不该被标成截断"
        # keep.md 照常入库（没被排除规则误伤）
        assert db.get_file_by_path(str(kb / "keep.md")) is not None
    finally:
        s.stop()
