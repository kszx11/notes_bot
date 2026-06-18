from __future__ import annotations

from pathlib import Path

from .doc_roots import DocRoot, resolve_doc_path


def safe_doc_path(doc_roots: tuple[DocRoot, ...], rel_path: str) -> Path:
    _, _, candidate = resolve_doc_path(doc_roots, rel_path)
    return candidate


def get_note_excerpt(
    *,
    doc_roots: tuple[DocRoot, ...],
    rel_path: str,
    start_line: int | None,
    end_line: int | None,
    context_before: int = 0,
    context_after: int = 0,
    max_chars: int = 120000,
) -> dict:
    root, inner_rel_path, abs_path = resolve_doc_path(doc_roots, rel_path)
    if not abs_path.exists() or not abs_path.is_file():
        raise FileNotFoundError(rel_path)

    lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    s = max(1, int(start_line or 1))
    e = min(n, int(end_line or n))
    if e < s:
        s, e = e, s

    display_start = max(1, s - max(0, int(context_before)))
    display_end = min(n, e + max(0, int(context_after)))
    excerpt = "\n".join(lines[display_start - 1 : display_end])
    truncated = False
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
        truncated = True

    return {
        "rel_path": rel_path,
        "root_name": root.name,
        "root_rel_path": inner_rel_path,
        "abs_path": str(abs_path),
        "start_line": s,
        "end_line": e,
        "display_start_line": display_start,
        "display_end_line": display_end,
        "total_lines": n,
        "text": excerpt,
        "truncated": truncated,
    }
