"""路径的两种呈现：文件系统用原始字节，接口与数据库用可编码的字符串。

**为什么需要这个模块**（2026-09-21 实测）：文件名不一定是 UTF-8。Windows/GBK 环境
拷进来的文件可以是 GBK 编码的名字（实测 `cuda/docs/` 下两个：按 GBK 读是
「深度拷问清单.md」与「CUDA_GPU_硬件架构深度解析.md」，字节为 `\\xc9\\xee\\xb6\\xc8…`）。

Python 的文件系统接口（`os.walk` / `os.listdir`）对这类名字不会报错，而是走
`os.fsdecode` 的 **surrogateescape**：字节 `\\xc9` 变成 `\\udcc9`。这个字符串
· `os.path.exists()` / `open()` 可用 —— 它记得原始字节，能原样还原；
· `sqlite3` 与 `json.dumps(..., ensure_ascii=False).encode("utf-8")` **不可用** ——
  两者都要求合法 UTF-8，含 surrogate 的 str 编不出来。

于是同一个路径在两类消费方之间必须分开呈现：内部留着原始字节去碰文件系统，
对外（SQL、JSON）必须是可编码的。**别把后者当"清洗"随手丢掉原始字节** ——
丢掉的写法（`os.path.exists(safe)`）会让「文件明明在」判成「文件不在」，
反而让失败明细从界面上消失。

含义与用法：
    display_path(p)  p 里的非法字节替换为 U+FFFD，只为能编码；可能不再是真实路径
    判存在/读写文件时**一律用原路径**，只有写库和出接口才换 display_path
"""
from __future__ import annotations

import os


def display_path(path: str) -> str:
    """把可能含 surrogate 的路径转成一定能 `encode("utf-8")` 的字符串。

    已经合法的路径原样返回（绝大多数情况，零改动）；只有含非法字节时才替换。
    非法字节 → U+FFFD，保留路径长度与其余部分，便于人眼定位是哪个文件。
    """
    if not path:
        return path
    try:
        path.encode("utf-8")
        return path
    except UnicodeEncodeError:
        return os.fsencode(path).decode("utf-8", "replace")


def is_utf8_safe(path: str) -> bool:
    """路径能否直接进 SQL / JSON（含 surrogate 则不能）。"""
    try:
        (path or "").encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


# 常见的中文文件名编码，按序尝试——只为把文件名「读给人看」，不改变磁盘上的字节。
_FILENAME_ENCODINGS = ("gb18030", "big5", "shift_jis", "cp1252")


def describe_unencodable_filename(path: str) -> dict:
    """含非法字节的文件名 → **结构化**原因（供界面按语言出词，不写死中文）。

    为什么改结构（2026-09-23 用户报）：此前这里返回一句拼好的中文，界面把
    `errorDetails[].reason` 原样打在「失败文件」面板右侧。用户把界面切到英文后，
    这一句仍然是中文（而且很长），既跟语言设置不符、又占掉大半个面板宽度。

    改法：后端只给**事实**（按哪个编码读、读出来是什么、原始字节是什么），文案在
    locales 里按语言拼。与 `kind` 同一套做法：后端给分类，前端给词。

    返回（键都在，值可为空；界面按 `kind == "unencodableFilename"` 认这条）：
        {"kind": "unencodableFilename",
         "encoding": "gb18030",   # 试出来的编码；试不出则 ""
         "guess": "深度拷问清单.md",  # 按该编码读出来的文件名（含目录）
         "base": b"\\xc9\\xee...md"}   # 原始字节的 repr，便于排查
    """
    raw = os.fsencode(path)
    enc_used, guess = "", ""
    for enc in _FILENAME_ENCODINGS:
        try:
            guess = raw.decode(enc)
            enc_used = enc
            break
        except Exception:  # noqa: BLE001 —— 试编码失败是预期的，继续试下一个
            continue
    return {
        "kind": "unencodableFilename",
        "encoding": enc_used,
        "guess": guess,
        "base": repr(os.path.basename(raw)),
    }
