import re

CITATION_RE = re.compile(r"\(([^:()]+):\s*(\d+)-(\d+)\)\s*$")

def _ranges_substantially_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
    threshold: float = 0.6,
) -> bool:
    overlap = min(end_a, end_b) - max(start_a, start_b) + 1
    if overlap <= 0:
        return False
    span_a = max(1, end_a - start_a + 1)
    span_b = max(1, end_b - start_b + 1)
    return (overlap / min(span_a, span_b)) >= threshold


def build_sources_block(results, max_chars: int, source_loader=None) -> tuple[str, list[dict], list[str]]:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[None]*len(docs)])[0]

    blocks = []
    used = 0
    used_sources = []
    source_texts = []
    seen_ranges: dict[str, list[tuple[int, int]]] = {}
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        loaded_meta = dict(meta)
        body = doc.strip()
        if source_loader is not None:
            loaded_meta, loaded_body = source_loader(dict(meta), doc)
            loaded_meta = dict(loaded_meta or meta)
            body = str(loaded_body or doc).strip()
        if not body:
            continue

        rel_path = str(loaded_meta.get("rel_path", meta.get("rel_path", "")))
        start_line = int(loaded_meta.get("start_line", meta.get("start_line", 1)))
        end_line = int(loaded_meta.get("end_line", meta.get("end_line", start_line)))
        overlap_ranges = seen_ranges.setdefault(rel_path, [])
        if any(
            _ranges_substantially_overlap(start_line, end_line, seen_start, seen_end)
            for seen_start, seen_end in overlap_ranges
        ):
            continue
        overlap_ranges.append((start_line, end_line))

        title = loaded_meta.get("title", "").strip()
        section_path = loaded_meta.get("section_path", "").strip()
        content_type = loaded_meta.get("content_type", "").strip()
        context_bits = [bit for bit in [section_path, content_type] if bit]
        context = f" [{ ' | '.join(context_bits) }]" if context_bits else ""
        if title and title != section_path:
            context = f"{context} title={title}"
        focus_start = loaded_meta.get("focus_start_line")
        focus_end = loaded_meta.get("focus_end_line")
        focus = ""
        if focus_start and focus_end and (focus_start != start_line or focus_end != end_line):
            focus = f" focus={focus_start}-{focus_end}"
        header = f"[SOURCE {len(used_sources) + 1}] {rel_path}:{start_line}-{end_line}{focus}{context}\n"
        block = header + body + "\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
        used_sources.append({**loaded_meta, "rank": len(used_sources) + 1, "distance": dist})
        source_texts.append(body)

    return ("\n".join(blocks).strip(), used_sources, source_texts)

def allowed_citation_set(used_sources: list[dict]) -> set[str]:
    """
    Create a set of allowed citation strings like:
      "path/to/file.md:12-34"
    """
    out = set()
    for m in used_sources:
        out.add(f"{m['rel_path']}:{m['start_line']}-{m['end_line']}")
    return out
    
