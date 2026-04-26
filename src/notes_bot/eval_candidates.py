from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .eval_runner import load_cases
from .search_log import load_search_log


def load_eval_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Candidate eval file must contain a JSON array.")
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def write_eval_candidate_file(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidates, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_candidate_selections(args: list[str], total: int) -> list[int] | None:
    if not args or any(arg.lower() == "all" for arg in args):
        return None

    selections: list[int] = []
    for raw in args:
        value = raw.strip()
        if not value:
            continue
        if "-" in value:
            start_raw, end_raw = value.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            lo = min(start, end)
            hi = max(start, end)
            for idx in range(lo, hi + 1):
                if 1 <= idx <= total and idx not in selections:
                    selections.append(idx)
            continue
        idx = int(value)
        if 1 <= idx <= total and idx not in selections:
            selections.append(idx)
    return selections


def build_eval_candidates(
    *,
    log_path: Path,
    existing_eval_path: Path | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    rows = load_search_log(log_path)
    existing_queries: set[str] = set()
    if existing_eval_path is not None and existing_eval_path.exists():
        existing_queries = {case.query for case in load_cases(existing_eval_path)}

    seen_queries: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda item: item.ts, reverse=True):
        query = row.query.strip()
        if not query or query in existing_queries or query in seen_queries:
            continue
        seen_queries.add(query)

        expected_paths: list[str] = []
        top_snippet = ""
        for hit in row.top_hits:
            rel_path = str(hit.get("rel_path", "")).strip()
            if rel_path and rel_path not in expected_paths:
                expected_paths.append(rel_path)
            if not top_snippet:
                top_snippet = str(hit.get("snippet", "")).strip()

        if not expected_paths:
            continue

        candidate = {
            "query": query,
            "expected_paths": expected_paths[:1],
            "expected_query_type": row.query_type if row.query_type in ("filename_focus", "snippet_focus", "mixed") else "mixed",
            "expected_snippet_terms": [],
            "notes": "Generated from search log. Review expected path/query type before promoting to eval set.",
        }
        if top_snippet:
            candidate["top_observed_snippet"] = top_snippet
        candidates.append(candidate)
        if len(candidates) >= limit:
            break

    return list(reversed(candidates))


def write_eval_candidates(
    *,
    log_path: Path,
    output_path: Path,
    existing_eval_path: Path | None = None,
    limit: int = 25,
) -> int:
    candidates = build_eval_candidates(
        log_path=log_path,
        existing_eval_path=existing_eval_path,
        limit=limit,
    )
    write_eval_candidate_file(output_path, candidates)
    return len(candidates)


def promote_eval_candidates(
    *,
    candidate_path: Path,
    eval_path: Path,
    selections: list[int] | None = None,
) -> tuple[int, int]:
    candidates = load_eval_candidates(candidate_path)
    if not candidates:
        return 0, 0

    if selections is None:
        selected_indexes = list(range(len(candidates)))
    else:
        selected_indexes = []
        for idx in selections:
            if 1 <= idx <= len(candidates):
                selected_indexes.append(idx - 1)

    if not selected_indexes:
        return 0, len(candidates)

    existing = load_eval_candidates(eval_path) if eval_path.exists() else []
    existing_queries = {
        str(item.get("query", "")).strip()
        for item in existing
        if isinstance(item, dict)
    }

    selected_set = set(selected_indexes)
    kept_candidates: list[dict[str, Any]] = []
    promoted = 0

    for idx, item in enumerate(candidates):
        if idx not in selected_set:
            kept_candidates.append(item)
            continue

        query = str(item.get("query", "")).strip()
        if not query or query in existing_queries:
            continue

        promoted_item = dict(item)
        promoted_item.pop("top_observed_snippet", None)
        existing.append(promoted_item)
        existing_queries.add(query)
        promoted += 1

    write_eval_candidate_file(eval_path, existing)
    write_eval_candidate_file(candidate_path, kept_candidates)
    return promoted, len(kept_candidates)
