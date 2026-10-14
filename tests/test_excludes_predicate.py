"""排除规则判据单测（is_excluded 唯一实现）。

2026-09-11 回归：判据曾在三处各写一份且语义不一致——遍历时按「条目名」比对（规则
命中父目录名即可剪枝），索引回收却按「完整路径 + 文件名」比对（规则写父目录名时
两个候选都对不上）→ 加排除规则保存后，已在库的文件永不回收、仍可被检索到。
"""
import pytest

from bibit_filesearch.excludes import is_excluded


def test_default_patterns_match_dir_component():
    """默认规则命中任意层级的目录分量（等价遍历时的目录剪枝）。"""
    pats = [".git", "node_modules", "__pycache__"]
    assert is_excluded(".git", pats)
    assert is_excluded("/home/u/kb/.git/config", pats)
    assert is_excluded("/home/u/kb/node_modules/foo/index.js", pats)
    assert is_excluded("/home/u/kb/a/b/__pycache__/x.pyc", pats)
    assert not is_excluded("/home/u/kb/notes.md", pats)


def test_parent_dir_pattern_matches_full_stored_path():
    """回归 B5：规则写父目录名时，已在库文件的完整路径必须判为排除。

    修复前 `fnmatch(path, "skipme")` 与 `fnmatch("no-index.txt", "skipme")` 均为
    False → 回收漏判 → 文件留在库里可搜到。
    """
    path = "/home/user/Desktop/fs-testdata/skipme/no-index.txt"
    assert is_excluded(path, ["skipme"]), "规则命中父目录分量即应排除"
    assert is_excluded("/a/b/skipme", ["skipme"]), "目录本身也排除"


def test_trailing_glob_suffix_normalized():
    """`dir/*`、`dir/**` 归一为 `dir`——否则按分量比对永远匹配不上（规则静默失效）。"""
    for pat in ("skipme/*", "skipme/**", "/skipme/*", "skipme/"):
        assert is_excluded("/a/b/skipme/f.txt", [pat]), f"{pat} 应等价于 skipme"
        assert is_excluded("skipme", [pat])


def test_filename_glob_still_works():
    """文件名 glob 不受影响（`*.log` 排除任意层级的 .log）。"""
    pats = ["*.log", "*.tmp"]
    assert is_excluded("/a/b/app.log", pats)
    assert is_excluded("app.log", pats)
    assert not is_excluded("/a/b/app.txt", pats)


def test_full_path_glob_supported():
    """允许写完整路径 glob（整路径匹配分支）。"""
    assert is_excluded("/home/u/kb/secret/a.md", ["/home/u/kb/secret/*"])
    assert not is_excluded("/home/u/other/a.md", ["/home/u/kb/secret/*"])


@pytest.mark.parametrize("patterns", [None, [], ["", "   ", "/"]])
def test_empty_patterns_exclude_nothing(patterns):
    """空规则/空串不排除任何东西（不得因空串变成「全部排除」）。"""
    assert not is_excluded("/a/b/c.md", patterns)
