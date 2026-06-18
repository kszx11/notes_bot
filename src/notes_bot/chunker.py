from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class Chunk:
    text: str
    start_line: int
    end_line: int
    chunk_index: int
    content_type: str
    title: str
    section_path: str


@dataclass
class _Block:
    text: str
    start_line: int
    end_line: int
    content_type: str
    title: str
    section_path: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_CONFIG_RE = re.compile(r"^\s*[A-Za-z0-9_.-]+\s*[:=]\s*\S")
_LOG_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[ T]|\[\w+]|(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE)\b|[A-Z][a-z]{2}\s+\d{1,2}\s+\d\d:\d\d:\d\d)"
)


def _classify_text_block(lines: list[str]) -> str:
    non_blank = [ln for ln in lines if ln.strip()]
    if not non_blank:
        return "prose"

    if all(_CONFIG_RE.match(ln) for ln in non_blank[: min(6, len(non_blank))]):
        return "config"

    log_like = sum(1 for ln in non_blank if _LOG_RE.match(ln.strip()))
    if log_like >= max(2, len(non_blank) // 2):
        return "log"

    return "prose"


def _split_lines_to_chunks(
    lines: list[str],
    start_line: int,
    *,
    chunk_chars: int,
    overlap: int,
    content_type: str,
    title: str,
    section_path: str,
) -> list[Chunk]:
    if not lines:
        return []

    chunks: list[Chunk] = []
    idx = 0
    start_idx = 0
    overlap = max(0, overlap)

    while start_idx < len(lines):
        total = 0
        end_idx = start_idx

        while end_idx < len(lines):
            next_len = len(lines[end_idx]) + (1 if end_idx > start_idx else 0)
            if end_idx > start_idx and total + next_len > chunk_chars:
                break
            total += next_len
            end_idx += 1

        if end_idx == start_idx:
            end_idx = start_idx + 1

        chunk_lines = lines[start_idx:end_idx]
        chunk_text = "\n".join(chunk_lines).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_line=start_line + start_idx,
                    end_line=start_line + end_idx - 1,
                    chunk_index=idx,
                    content_type=content_type,
                    title=title,
                    section_path=section_path,
                )
            )
            idx += 1

        if end_idx >= len(lines):
            break

        if overlap <= 0:
            start_idx = end_idx
            continue

        rewind = 0
        back = end_idx - 1
        while back > start_idx:
            rewind += len(lines[back]) + 1
            if rewind >= overlap:
                break
            back -= 1
        start_idx = max(start_idx + 1, back)

    return chunks


def chunk_with_line_ranges(text: str, chunk_chars: int, overlap: int) -> list[Chunk]:
    if not text or chunk_chars <= 0:
        return []

    lines = text.splitlines()
    blocks: list[_Block] = []
    heading_stack: list[tuple[int, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_no = i + 1

        if not stripped:
            i += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            i += 1
            continue

        section_titles = [title for _, title in heading_stack]
        title = section_titles[-1] if section_titles else ""
        section_path = " / ".join(section_titles)

        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            block_lines = [line]
            start = i
            i += 1
            while i < len(lines):
                block_lines.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            blocks.append(
                _Block(
                    text="\n".join(block_lines).strip(),
                    start_line=start + 1,
                    end_line=start + len(block_lines),
                    content_type="code",
                    title=title,
                    section_path=section_path,
                )
            )
            continue

        if _BULLET_RE.match(line):
            block_lines = [line]
            start = i
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    block_lines.append(nxt)
                    i += 1
                    continue
                if _BULLET_RE.match(nxt) or nxt.startswith((" ", "\t")):
                    block_lines.append(nxt)
                    i += 1
                    continue
                break
            blocks.append(
                _Block(
                    text="\n".join(block_lines).strip(),
                    start_line=start + 1,
                    end_line=start + len(block_lines),
                    content_type="list",
                    title=title,
                    section_path=section_path,
                )
            )
            continue

        block_lines = [line]
        start = i
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if _HEADING_RE.match(nxt):
                break
            if nxt.strip().startswith(("```", "~~~")):
                break
            if _BULLET_RE.match(nxt):
                break
            block_lines.append(nxt)
            i += 1

        blocks.append(
            _Block(
                text="\n".join(block_lines).strip(),
                start_line=start + 1,
                end_line=start + len(block_lines),
                content_type=_classify_text_block(block_lines),
                title=title,
                section_path=section_path,
            )
        )

    chunks: list[Chunk] = []
    chunk_index = 0
    for block in blocks:
        block_lines = block.text.splitlines()
        for chunk in _split_lines_to_chunks(
            block_lines,
            block.start_line,
            chunk_chars=chunk_chars,
            overlap=overlap,
            content_type=block.content_type,
            title=block.title,
            section_path=block.section_path,
        ):
            chunk.chunk_index = chunk_index
            chunk_index += 1
            chunks.append(chunk)

    return chunks
