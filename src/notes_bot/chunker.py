from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Chunk:
    text: str
    start_line: int
    end_line: int
    chunk_index: int

def _line_start_offsets(text: str) -> list[int]:
    # offsets[i] = char offset where line i (1-based) starts
    offsets = [0]
    off = 0
    for ch in text:
        off += 1
        if ch == "\n":
            offsets.append(off)
    return offsets  # len = number_of_lines

def _offset_to_line(offsets: list[int], char_offset: int) -> int:
    # Find rightmost line start <= char_offset
    # Simple linear could be ok; use binary search for speed.
    import bisect
    i = bisect.bisect_right(offsets, char_offset) - 1
    return max(1, i + 1)

def chunk_with_line_ranges(text: str, chunk_chars: int, overlap: int) -> List[Chunk]:
    if not text:
        return []
    if chunk_chars <= 0:
        return []
    overlap = max(0, overlap)

    offsets = _line_start_offsets(text)
    n = len(text)

    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < n:
        end = min(n, start + chunk_chars)
        # Try not to cut mid-paragraph: expand end to next blank line if close
        window = text[start:end]
        cut = window.rfind("\n\n")
        if cut != -1 and cut > int(0.6 * len(window)):
            candidate_end = start + cut + 2  # include the blank line
            # Never choose a cut that would prevent forward progress with overlap.
            if (candidate_end - start) > overlap:
                end = candidate_end

        start_line = _offset_to_line(offsets, start)
        end_line = _offset_to_line(offsets, max(start, end - 1))

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, start_line=start_line, end_line=end_line, chunk_index=idx))
            idx += 1

        if end >= n:
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            # Hard guard against pathological overlap/cut combinations.
            next_start = min(n, start + 1)
        start = next_start

    return chunks
    
