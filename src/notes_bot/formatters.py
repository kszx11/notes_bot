from __future__ import annotations

from .search import SearchResults


def format_search_results(results: SearchResults) -> str:
    if not results.hits:
        return f"No indexed notes matched: {results.query}"

    lines = [f"Top matches for: {results.query}"]
    for idx, hit in enumerate(results.hits, start=1):
        if hit.start_line is not None and hit.end_line is not None:
            location = f"{hit.rel_path}:{hit.start_line}-{hit.end_line}"
        else:
            location = hit.rel_path
        reasons = ", ".join(hit.reasons)
        lines.append(f"{idx}. {location}")
        lines.append(f"   reasons: {reasons}")
        if hit.heading:
            lines.append(f"   heading: {hit.heading}")
        if hit.snippet:
            lines.append(f"   snippet: {hit.snippet}")
    return "\n".join(lines)
