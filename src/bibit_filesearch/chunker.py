"""分块器：文本/代码按固定长度重叠分块（分块策略由实现确定）。

- 文本：按字符数 500/100 重叠（中文场景字符 ≈ token）
- 代码：按字符预算分块（600 字符 ≈ 400 token，留余量到 embed 模型 512 上限），
  优先在空行/函数边界断开（2026-08-26：原 60 行分块实测 1000-2100 token 超限，
  embed 服务 500 报错且前端无提示——改为字符预算保证不超）
- 每个块携带来源文件与块序号，供向量化/检索用
"""
from __future__ import annotations

from bibit_filesearch.parser import CODE_TYPES


def _is_word_char(c: str) -> bool:
    """分块边界处的\"词内字符\"判断：字母/数字/下划线/中日韩文字等。

    2026-09-04：英文支持——原 chunk_text 按字符硬切，边界落在英文单词中间时
    会把词拦腰截断（proce|ssing），BM25 分词与向量语义都受影响。
    """
    return c.isalnum() or c == "_"


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """按字符数重叠分块。空文本返回 []。

    2026-09-04：边界若落在单词中间，向前回退到最近空白（不超过 overlap 距离），
    避免英文/代码词被切断；纯中文段落无空白时保持原硬切（行为不变）。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    step = size - overlap
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text) and _is_word_char(text[end - 1]) and _is_word_char(text[end]):
            # 边界在词中间：向前找最近空白（限于 overlap 窗口内），找不到则原样硬切
            for i in range(end - 1, max(start, end - overlap) - 1, -1):
                if text[i].isspace():
                    end = i
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_code(text: str, size_lines: int = 60, overlap_lines: int = 10) -> list[str]:
    """代码按行分块：优先在空行断开（近似函数边界），无空行则硬切。

    注意：2026-08-26 起生产路径（split_by_type）改用 chunk_code_by_chars（字符预算，
    防超 embed token 上限）；本函数保留为行数版 API（测试用/显式行数控制）。
    """
    lines = text.split("\n")
    if not lines:
        return []
    if len(lines) <= size_lines:
        return [text]
    chunks: list[str] = []
    step = size_lines - overlap_lines
    start = 0
    while start < len(lines):
        end = min(start + size_lines, len(lines))
        block = lines[start:end]
        # 尝试在空行处微调（向 block 尾部找空行，保持块不超 size_lines+10）
        if end < len(lines):
            for i in range(len(block) - 1, max(0, len(block) - 11), -1):
                if block[i].strip() == "":
                    block = block[: i + 1]
                    end = start + i + 1
                    break
        chunks.append("\n".join(block))
        if end >= len(lines):
            break
        start = end - overlap_lines
    return chunks


def chunk_code_by_chars(text: str, size_chars: int = 600, overlap_lines: int = 5) -> list[str]:
    """代码按字符预算分块（2026-08-26 生产路径）。

    600 字符 ≈ 400 token（中文 1 token≈1.5 字符，留 20% 余量到 embed 512 上限）。
    尽量在行边界 + 空行处断开（函数边界语义）；单行超预算时硬切该行（防 minified 超长行）。
    """
    lines = text.split("\n")
    if not lines:
        return []
    if len(text) <= size_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(lines):
        # 累计行字符到预算
        chars = 0
        end = start
        while end < len(lines) and chars + len(lines[end]) + 1 <= size_chars:
            chars += len(lines[end]) + 1
            end += 1
        if end == start:
            # 单行超预算：按字符硬切该行（防 minified 超长行撑爆 token）。
            # 注意不 start+=1——lines[start] 已替换为剩余部分，下一轮继续切（防丢内容）
            line = lines[start]
            chunks.append(line[:size_chars])
            rest = line[size_chars:]
            if rest:
                lines[start] = rest
            else:
                start += 1
            continue
        # 块尾附近找空行断开（不超预算 1.3 倍，保函数边界语义）
        if end < len(lines):
            cur_chars = sum(len(l) + 1 for l in lines[start:end])
            for i in range(end - 1, start, -1):
                if lines[i].strip() == "":
                    cand = sum(len(l) + 1 for l in lines[start : i + 1])
                    if cand <= size_chars * 1.3:
                        end = i + 1
                    break
        chunks.append("\n".join(lines[start:end]))
        start = max(end - overlap_lines, start + 1)
    return chunks


def need_chars_for_chunks(size: int, overlap: int, n_chunks: int) -> int:
    """产出 **n_chunks** 块所需的**最小**输入字符数（0 = 不限 / 无需求）。

    2026-09-29（PDF 早停）：这是"预算 → 输入量"的唯一权威换算。调用方是
    `Indexer.index_file`（它知道自己要留几块），消费方是 `parser._parse_pdf`
    （它只把它当"抽到这么多字就不用再往后看了"的界）。**parser 因此不必知道
    "块"这个概念** —— 换算留在这边，与 `chunk_text` 的切块规则同文件、同步演进。

    推导（与 chunk_text 逐字对齐）：
        · n == 1：`len(text) <= size` 走短路返回 1 块 ⇒ 非空即可，界 = 1。
        · n >= 2：必须**越过**短路分支（len > size），且循环要能走到第 n-1 块。
          `start_k = k*(size-overlap)`；第 k 块在 `start_k < len` 时产出。
          取最小满足值 ⇒ len = (n-1)*(size-overlap) + overlap + 1。
          （因为 step + overlap == size，等价于 (n-2)*step + size + 1。）

    ★ 这两个数不是推出来的，是**实测标定**的：对硬切文本做二分，最小长度
      n=2 → 501、n=3 → 901、n=4 → 1301、n=64 → 25301、n=65 → 25701，
      每多一块恰好 +400（= step）。上面的公式逐点吻合；漏掉 "+overlap"
      会低估 100 字符（写成 (n-1)*step+1 时 n=2 给 401，而 401 字仍走短路只出 1 块）。
      测试 `test_pdf_twopass.py::test_need_chars_is_exact` 对 n=1..70 逐点验：
      用 need(n) 恰好产出 n 块、用 need(n)-1 只产出 n-1 块。
    """
    n = int(n_chunks or 0)
    if n <= 0:
        return 0
    if n == 1:
        return 1
    step = max(int(size) - int(overlap), 1)
    return (n - 1) * step + max(int(overlap), 0) + 1


def split_by_type(text: str, file_type: str, size: int = 500, overlap: int = 100) -> list[str]:
    """按文件类型分块：代码走字符预算分块，其余走字符分块。

    2026-08-26：代码类型用 parser.CODE_TYPES（原硬编码集合与 parser 重复）；
    代码分块改 chunk_code_by_chars（600 字符预算，原 60 行实测超 embed 512 token 上限
    → embed 500 报错且前端无提示）。
    """
    if file_type in CODE_TYPES:
        # 600 字符预算固定（实测 0 超限；不随文本 size 变——代码行长短差异大）
        return chunk_code_by_chars(text, size_chars=600)
    return chunk_text(text, size, overlap)
