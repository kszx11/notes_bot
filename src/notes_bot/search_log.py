from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .search import SearchResults


@dataclass
class LoggedSearch:
    ts: float
    query: str
    query_type: str
    top_hits: list[dict[str, object]]


def append_search_log(path: Path, *, query: str, results: SearchResults, max_hits: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = LoggedSearch(
        ts=time.time(),
        query=query,
        query_type=results.query_type,
        top_hits=[
            {
                "rel_path": hit.rel_path,
                "start_line": hit.start_line,
                "end_line": hit.end_line,
                "snippet": hit.snippet,
                "score": round(hit.score, 6),
                "reasons": hit.reasons,
            }
            for hit in results.hits[:max_hits]
        ],
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(payload), ensure_ascii=True) + "\n")


def load_search_log(path: Path) -> list[LoggedSearch]:
    if not path.exists():
        return []

    rows: list[LoggedSearch] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        query = str(obj.get("query", "")).strip()
        query_type = str(obj.get("query_type", "")).strip() or "mixed"
        if not query:
            continue
        top_hits_raw = obj.get("top_hits", [])
        top_hits: list[dict[str, object]] = []
        if isinstance(top_hits_raw, list):
            for item in top_hits_raw:
                if isinstance(item, dict):
                    top_hits.append(item)
        rows.append(
            LoggedSearch(
                ts=float(obj.get("ts", 0.0) or 0.0),
                query=query,
                query_type=query_type,
                top_hits=top_hits,
            )
        )
    return rows
