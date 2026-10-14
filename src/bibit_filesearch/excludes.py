"""排除规则判定的唯一定义（scanner / watcher / 索引回收共用）。

历史问题（2026-09-11 修）：三处各写一份判据，语义不一致——
  · `scanner._excluded(name)` / `watcher._excluded(name)` 逐条目判定（遍历到
    `skipme/` 目录时拿目录名比对）→ 规则命中父目录即可剪枝，新文件不会入库；
  · `api/settings._reconcile_index` 事后核对全库，却只拿「完整路径 + 文件名」
    比对 → 规则写 `skipme`（父目录名）时两个候选都对不上 → **已在库的文件
    永不回收**。
表现为：把目录加进排除规则并保存后，该目录下的文件仍能被检索到（B5）。
本模块把判据收敛成一处：**路径的任一分量命中规则即排除**，与遍历时的
目录剪枝天然同语义；同时保留「整路径 glob」写法。
"""
from __future__ import annotations

import fnmatch
from typing import Iterable


def _pattern_variants(raw: str) -> list[str]:
    """一条用户规则 → 若干等价候选写法（去重、去空）。

    用户在界面上按直觉写规则，以下写法应等价：
      `dir`、`dir/`、`/dir`、`dir/*`、`dir/**`   → 均表示「排除 dir」
      `*.log`                                    → 排除任意层级的 .log
      `/home/u/kb/secret/*`                      → 完整路径 glob 仍按整路径匹配
    归一在判定侧做（而非改写用户配置），保证落盘内容与用户输入一致。
    """
    pat = str(raw or "").strip()
    out: list[str] = []

    def _add(p: str) -> None:
        if p and p not in out and p.strip("/"):
            out.append(p)

    _add(pat)
    _add(pat.rstrip("/"))   # "dir/" → "dir"
    _add(pat.strip("/"))    # "/dir" → "dir"
    # 尾部 `/*`、`/**` 逐层剥掉：按分量比对时分量不含斜杠，带后缀的规则否则永不命中
    stripped = pat.rstrip("/")
    while stripped.endswith("/**") or stripped.endswith("/*"):
        stripped = stripped[:-3] if stripped.endswith("/**") else stripped[:-2]
        stripped = stripped.rstrip("/")
        _add(stripped)
        _add(stripped.strip("/"))
    return out


def is_excluded(path_or_name: str, patterns: Iterable[str] | None) -> bool:
    """路径/条目名是否命中排除规则。

    命中条件（任一成立）：
      1. 完整路径匹配某条 glob（支持用户写完整路径 glob，如 `/home/u/kb/*.log`）；
      2. 路径的**任一分量**（目录名或文件名）匹配某条 glob——这是遍历时
         「目录剪枝」的等价事后判据，也是本次修复的关键。
    """
    if not patterns:
        return False
    text = str(path_or_name or "")
    parts = [p for p in text.split("/") if p]
    for raw in patterns:
        for pat in _pattern_variants(raw):
            if fnmatch.fnmatch(text, pat):
                return True
            for part in parts:
                if fnmatch.fnmatch(part, pat):
                    return True
    return False
