from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI

from .config import load_config
from .manifest import Manifest
from .search import search_notes
from .store import VectorStore

EvalQueryType = Literal["filename_focus", "snippet_focus", "mixed"]


@dataclass
class EvalCase:
    query: str
    expected_paths: list[str]
    expected_snippet_terms: list[str]
    expected_query_type: EvalQueryType | None = None
    notes: str = ""


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval file must contain a JSON array.")
    cases: list[EvalCase] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each eval case must be a JSON object.")
        query = str(item.get("query", "")).strip()
        expected_paths = [str(p) for p in item.get("expected_paths", []) if str(p).strip()]
        expected_snippet_terms = [str(t).strip().lower() for t in item.get("expected_snippet_terms", []) if str(t).strip()]
        raw_query_type = item.get("expected_query_type")
        expected_query_type = None
        if raw_query_type is not None:
            expected_query_type = str(raw_query_type).strip() or None
            if expected_query_type not in ("filename_focus", "snippet_focus", "mixed"):
                raise ValueError("expected_query_type must be filename_focus, snippet_focus, or mixed.")
        notes = str(item.get("notes", "")).strip()
        if not query or not expected_paths:
            raise ValueError("Each eval case requires query and expected_paths.")
        cases.append(
            EvalCase(
                query=query,
                expected_paths=expected_paths,
                expected_snippet_terms=expected_snippet_terms,
                expected_query_type=expected_query_type,
                notes=notes,
            )
        )
    return cases


def _first_match_rank(results: list[str], expected_paths: list[str]) -> int | None:
    expected = set(expected_paths)
    for idx, rel_path in enumerate(results, start=1):
        if rel_path in expected:
            return idx
    return None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _snippet_terms_rank(results, expected_paths: list[str], expected_terms: list[str]) -> int | None:
    if not expected_terms:
        return None
    expected = set(expected_paths)
    for idx, hit in enumerate(results.hits, start=1):
        if hit.rel_path not in expected:
            continue
        snippet = _normalize_text(hit.snippet)
        if all(term in snippet for term in expected_terms):
            return idx
    return None


def run_eval(config_path: str | Path, eval_path: str | Path, limit: int = 8) -> str:
    cfg = load_config(config_path)
    manifest = Manifest(cfg.manifest_path)
    store = VectorStore(cfg.index_dir, collection_name="notes")
    client = OpenAI()
    cases = load_cases(Path(eval_path))

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_limit = 0
    query_type_hits = 0
    snippet_cases = 0
    snippet_hit_at_1 = 0
    snippet_hit_at_3 = 0
    snippet_hit_at_limit = 0
    lines = [f"Running {len(cases)} search eval case(s) with limit={limit}."]

    for idx, case in enumerate(cases, start=1):
        results = search_notes(
            query=case.query,
            client=client,
            cfg=cfg,
            manifest=manifest,
            store=store,
            limit=limit,
        )
        paths = [hit.rel_path for hit in results.hits]
        rank = _first_match_rank(paths, case.expected_paths)
        snippet_rank = _snippet_terms_rank(results, case.expected_paths, case.expected_snippet_terms)
        if rank == 1:
            hit_at_1 += 1
        if rank is not None and rank <= 3:
            hit_at_3 += 1
        if rank is not None and rank <= limit:
            hit_at_limit += 1
        if case.expected_query_type is not None and results.query_type == case.expected_query_type:
            query_type_hits += 1
        if case.expected_snippet_terms:
            snippet_cases += 1
            if snippet_rank == 1:
                snippet_hit_at_1 += 1
            if snippet_rank is not None and snippet_rank <= 3:
                snippet_hit_at_3 += 1
            if snippet_rank is not None and snippet_rank <= limit:
                snippet_hit_at_limit += 1

        top = paths[0] if paths else "<no result>"
        outcome = f"rank={rank}" if rank is not None else "miss"
        lines.append(f"{idx}. {case.query} -> {outcome}; top={top}; query_type={results.query_type}")
        if rank is None:
            lines.append(f"   expected: {', '.join(case.expected_paths)}")
        if case.expected_query_type is not None and results.query_type != case.expected_query_type:
            lines.append(f"   expected_query_type: {case.expected_query_type}")
        if case.expected_snippet_terms:
            snippet_outcome = f"snippet_rank={snippet_rank}" if snippet_rank is not None else "snippet_miss"
            lines.append(f"   {snippet_outcome}; terms={', '.join(case.expected_snippet_terms)}")
        if results.hits:
            lines.append(f"   top_snippet: {results.hits[0].snippet}")
        if case.notes:
            lines.append(f"   notes: {case.notes}")

    total = max(1, len(cases))
    lines.append("")
    lines.append(
        "Summary: "
        f"hit@1={hit_at_1}/{len(cases)} ({(100*hit_at_1/total):.1f}%), "
        f"hit@3={hit_at_3}/{len(cases)} ({(100*hit_at_3/total):.1f}%), "
        f"hit@{limit}={hit_at_limit}/{len(cases)} ({(100*hit_at_limit/total):.1f}%)"
    )
    typed_cases = sum(1 for case in cases if case.expected_query_type is not None)
    if typed_cases:
        lines.append(
            "Query Type: "
            f"{query_type_hits}/{typed_cases} ({(100*query_type_hits/max(1, typed_cases)):.1f}%)"
        )
    if snippet_cases:
        lines.append(
            "Snippet Terms: "
            f"hit@1={snippet_hit_at_1}/{snippet_cases} ({(100*snippet_hit_at_1/max(1, snippet_cases)):.1f}%), "
            f"hit@3={snippet_hit_at_3}/{snippet_cases} ({(100*snippet_hit_at_3/max(1, snippet_cases)):.1f}%), "
            f"hit@{limit}={snippet_hit_at_limit}/{snippet_cases} ({(100*snippet_hit_at_limit/max(1, snippet_cases)):.1f}%)"
        )
    return "\n".join(lines)


def stream_eval(config_path: str | Path, eval_path: str | Path, limit: int = 8) -> str:
    cfg = load_config(config_path)
    manifest = Manifest(cfg.manifest_path)
    store = VectorStore(cfg.index_dir, collection_name="notes")
    client = OpenAI()
    cases = load_cases(Path(eval_path))

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_limit = 0
    query_type_hits = 0
    snippet_cases = 0
    snippet_hit_at_1 = 0
    snippet_hit_at_3 = 0
    snippet_hit_at_limit = 0
    lines = [f"Running {len(cases)} search eval case(s) with limit={limit}."]
    print(lines[0], flush=True)

    for idx, case in enumerate(cases, start=1):
        print(f"[eval] case {idx}/{len(cases)}: {case.query}", flush=True)
        results = search_notes(
            query=case.query,
            client=client,
            cfg=cfg,
            manifest=manifest,
            store=store,
            limit=limit,
        )
        paths = [hit.rel_path for hit in results.hits]
        rank = _first_match_rank(paths, case.expected_paths)
        snippet_rank = _snippet_terms_rank(results, case.expected_paths, case.expected_snippet_terms)
        if rank == 1:
            hit_at_1 += 1
        if rank is not None and rank <= 3:
            hit_at_3 += 1
        if rank is not None and rank <= limit:
            hit_at_limit += 1
        if case.expected_query_type is not None and results.query_type == case.expected_query_type:
            query_type_hits += 1
        if case.expected_snippet_terms:
            snippet_cases += 1
            if snippet_rank == 1:
                snippet_hit_at_1 += 1
            if snippet_rank is not None and snippet_rank <= 3:
                snippet_hit_at_3 += 1
            if snippet_rank is not None and snippet_rank <= limit:
                snippet_hit_at_limit += 1

        top = paths[0] if paths else "<no result>"
        outcome = f"rank={rank}" if rank is not None else "miss"
        case_lines = [f"{idx}. {case.query} -> {outcome}; top={top}; query_type={results.query_type}"]
        if rank is None:
            case_lines.append(f"   expected: {', '.join(case.expected_paths)}")
        if case.expected_query_type is not None and results.query_type != case.expected_query_type:
            case_lines.append(f"   expected_query_type: {case.expected_query_type}")
        if case.expected_snippet_terms:
            snippet_outcome = f"snippet_rank={snippet_rank}" if snippet_rank is not None else "snippet_miss"
            case_lines.append(f"   {snippet_outcome}; terms={', '.join(case.expected_snippet_terms)}")
        if results.hits:
            case_lines.append(f"   top_snippet: {results.hits[0].snippet}")
        if case.notes:
            case_lines.append(f"   notes: {case.notes}")
        for line in case_lines:
            print(line, flush=True)
        lines.extend(case_lines)

    total = max(1, len(cases))
    summary = (
        "Summary: "
        f"hit@1={hit_at_1}/{len(cases)} ({(100*hit_at_1/total):.1f}%), "
        f"hit@3={hit_at_3}/{len(cases)} ({(100*hit_at_3/total):.1f}%), "
        f"hit@{limit}={hit_at_limit}/{len(cases)} ({(100*hit_at_limit/total):.1f}%)"
    )
    print("", flush=True)
    print(summary, flush=True)
    lines.append("")
    lines.append(summary)

    typed_cases = sum(1 for case in cases if case.expected_query_type is not None)
    if typed_cases:
        query_type_summary = (
            "Query Type: "
            f"{query_type_hits}/{typed_cases} ({(100*query_type_hits/max(1, typed_cases)):.1f}%)"
        )
        print(query_type_summary, flush=True)
        lines.append(query_type_summary)
    if snippet_cases:
        snippet_summary = (
            "Snippet Terms: "
            f"hit@1={snippet_hit_at_1}/{snippet_cases} ({(100*snippet_hit_at_1/max(1, snippet_cases)):.1f}%), "
            f"hit@3={snippet_hit_at_3}/{snippet_cases} ({(100*snippet_hit_at_3/max(1, snippet_cases)):.1f}%), "
            f"hit@{limit}={snippet_hit_at_limit}/{snippet_cases} ({(100*snippet_hit_at_limit/max(1, snippet_cases)):.1f}%)"
        )
        print(snippet_summary, flush=True)
        lines.append(snippet_summary)

    return "\n".join(lines)
