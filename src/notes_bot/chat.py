from __future__ import annotations
from dataclasses import dataclass
import re
import threading
import time
from pathlib import Path

from openai import OpenAI

from .config import load_config
from .eval_candidates import build_eval_candidates, load_eval_candidates, parse_candidate_selections, promote_eval_candidates, write_eval_candidate_file
from .formatters import format_search_results
from .intent import route_user_input
from .manifest import Manifest
from .store import VectorStore
from .indexer import run_index_once
from .history import ChatHistory, ChatTurn
from .prompt import GENERAL_CHAT_PROMPT
from .search import search_notes
from .search_log import append_search_log

HELP_TEXT = (
    "Ask naturally:\n"
    "- where did I mention docker auth?\n"
    "- find my note about backups\n"
    "- notes with incident in the title\n"
    "- what is the difference between TCP and UDP?\n"
    "\n"
    "Commands:\n"
    "- /help show this help\n"
    "- /clear clear chat context/history\n"
    "- /reindex run incremental indexing now\n"
    "- /reindex-force rebuild the entire index now\n"
    "- /reindex-status show current indexer status\n"
    "- /eval-candidates refresh and show draft eval cases from search logs\n"
    "- /eval-promote <n|a-b|all> move reviewed eval candidates into the main eval set\n"
    "- /indexed list indexed files from manifest\n"
    "- /exit quit\n"
)

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s<>'\"`]+")
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SQL_BLOCK_RE = re.compile(
    r"(?is)(?:^|[;\n])\s*((?:select|insert|update|delete|create|alter|drop|truncate|merge|with)\b[\s\S]{1,1200}?;)"
)
_SQL_LINE_RE = re.compile(
    r"(?im)^\s*(select|insert|update|delete|create|alter|drop|truncate|merge|with)\b[^\n;]{6,}$"
)
_API_KEY_RE_LIST = [
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),               # OpenAI-style
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                  # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),            # Google API key
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),      # Slack tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),        # GitHub tokens
]


@dataclass
class ReindexStatus:
    running: bool = False
    owner: str = ""
    mode: str = "incremental"
    phase: str = "idle"
    status: str = "idle"
    current_file: str = ""
    current_index: int = 0
    total_files: int = 0
    updated: int = 0
    deleted: int = 0
    errors: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0


def _read_note_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _fmt_seconds(total_seconds: float) -> str:
    if total_seconds < 0 or total_seconds == float("inf"):
        return "--:--"
    s = int(total_seconds)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _set_reindex_status(state: ReindexStatus, state_lock: threading.Lock, **kwargs) -> None:
    with state_lock:
        for key, value in kwargs.items():
            setattr(state, key, value)
        state.updated_at = time.time()


def _get_reindex_status(state: ReindexStatus, state_lock: threading.Lock) -> ReindexStatus:
    with state_lock:
        return ReindexStatus(**state.__dict__)


def _format_reindex_status(state: ReindexStatus) -> str:
    if state.running:
        runtime = _fmt_seconds(max(0.0, time.time() - state.started_at)) if state.started_at else "--:--"
        progress = f"{state.current_index}/{state.total_files}" if state.total_files > 0 else "--/--"
        location = state.current_file or "<starting>"
        return (
            f"Indexer is running.\n"
            f"- owner: {state.owner or 'unknown'}\n"
            f"- mode: {state.mode}\n"
            f"- phase: {state.phase}\n"
            f"- status: {state.status}\n"
            f"- progress: {progress}\n"
            f"- elapsed: {runtime}\n"
            f"- updated: {state.updated}\n"
            f"- deleted: {state.deleted}\n"
            f"- errors: {state.errors}\n"
            f"- file: {location}"
        )

    if state.updated_at > 0:
        age = _fmt_seconds(max(0.0, time.time() - state.updated_at))
        return (
            f"Indexer is idle.\n"
            f"- last owner: {state.owner or 'n/a'}\n"
            f"- last mode: {state.mode}\n"
            f"- last phase: {state.phase}\n"
            f"- last status: {state.status}\n"
            f"- updated: {state.updated}\n"
            f"- deleted: {state.deleted}\n"
            f"- errors: {state.errors}\n"
            f"- last update: {age} ago"
        )

    return "Indexer is idle. No reindex activity recorded in this session."


def _make_progress_callback(label: str, state: ReindexStatus | None = None, state_lock: threading.Lock | None = None):
    start = time.time()
    bar_width = 28
    last_emit = 0.0
    last_file_emit = 0.0

    def on_progress(event: dict) -> None:
        nonlocal last_emit, last_file_emit
        phase = event.get("phase")
        stats = event.get("stats")

        if phase == "schema_upgrade":
            reason = str(event.get("reason", "schema_upgrade"))
            old_version = str(event.get("old_version", "unset"))
            new_version = str(event.get("new_version", "unset"))
            total_files = int(event.get("total_files", 0))
            if state is not None and state_lock is not None:
                _set_reindex_status(
                    state,
                    state_lock,
                    running=True,
                    phase="schema_upgrade",
                    status=reason,
                    total_files=total_files,
                )
            if reason == "manual_force":
                print(f"[{label}] starting full reindex of {total_files} file(s).", flush=True)
            else:
                print(
                    f"[{label}] index schema changed {old_version} -> {new_version}; "
                    f"forcing full reindex of {total_files} file(s).",
                    flush=True,
                )
            return

        if phase == "scan":
            idx = int(event.get("index", 0))
            total = max(1, int(event.get("total", 1)))
            status = str(event.get("status", ""))
            mode = str(event.get("mode", "incremental"))
            rel_path = str(event.get("rel_path", ""))
            if state is not None and state_lock is not None:
                updated = int(getattr(stats, "updated", 0)) if stats is not None else state.updated
                deleted = int(getattr(stats, "deleted", 0)) if stats is not None else state.deleted
                errors = int(getattr(stats, "errors", 0)) if stats is not None else state.errors
                _set_reindex_status(
                    state,
                    state_lock,
                    running=True,
                    mode=mode,
                    phase="scan",
                    status=status,
                    current_file=rel_path,
                    current_index=idx,
                    total_files=total,
                    updated=updated,
                    deleted=deleted,
                    errors=errors,
                )

            elapsed = max(0.001, time.time() - start)
            rate = idx / elapsed
            eta = (total - idx) / rate if rate > 0 else float("inf")
            pct = idx / total
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)

            prefix = (
                f"[{label}] mode={mode:<11} [{bar}] {idx}/{total} {pct*100:5.1f}% "
                f"status={status:<9} eta={_fmt_seconds(eta)}"
            )
            if stats is not None:
                prefix += (
                    f" u={stats.updated} d={stats.deleted} e={stats.errors}"
                )
            line = f"{prefix} file={rel_path[:80]}"
            now = time.time()
            should_emit = (
                idx <= 1
                or idx >= total
                or status in ("updated", "error")
                or (now - last_emit) >= 0.8
            )
            if should_emit:
                print(line, flush=True)
                last_emit = now
            if status == "error":
                err = str(event.get("error", "unknown error"))
                print(f"\n[{label}] error in {rel_path}: {err}")
            elif status == "skipped_large":
                size_mb = float(event.get("size_mb", 0))
                max_mb = int(event.get("max_file_size_mb", 0))
                print(
                    f"\n[{label}] skipped large file={rel_path[:90]} "
                    f"size={size_mb:.2f}MB limit={max_mb}MB",
                    flush=True,
                )

        elif phase == "delete":
            idx = int(event.get("index", 0))
            total = max(1, int(event.get("total", 1)))
            rel_path = str(event.get("rel_path", ""))
            if state is not None and state_lock is not None:
                updated = int(getattr(stats, "updated", 0)) if stats is not None else state.updated
                deleted = int(getattr(stats, "deleted", 0)) if stats is not None else state.deleted
                errors = int(getattr(stats, "errors", 0)) if stats is not None else state.errors
                _set_reindex_status(
                    state,
                    state_lock,
                    running=True,
                    phase="delete",
                    status="deleting",
                    current_file=rel_path,
                    current_index=idx,
                    total_files=total,
                    updated=updated,
                    deleted=deleted,
                    errors=errors,
                )
            pct = idx / total
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"[{label}] deleting [{bar}] {idx}/{total} {pct*100:5.1f}% file={rel_path[:80]}", flush=True)

        elif phase == "file":
            idx = int(event.get("index", 0))
            total = max(1, int(event.get("total", 1)))
            status = str(event.get("status", ""))
            rel_path = str(event.get("rel_path", ""))
            stats_suffix = ""
            if stats is not None:
                stats_suffix = f" u={stats.updated} d={stats.deleted} e={stats.errors}"
            if state is not None and state_lock is not None:
                updated = int(getattr(stats, "updated", 0)) if stats is not None else state.updated
                deleted = int(getattr(stats, "deleted", 0)) if stats is not None else state.deleted
                errors = int(getattr(stats, "errors", 0)) if stats is not None else state.errors
                _set_reindex_status(
                    state,
                    state_lock,
                    running=True,
                    phase="file",
                    status=status,
                    current_file=rel_path,
                    current_index=idx,
                    total_files=total,
                    updated=updated,
                    deleted=deleted,
                    errors=errors,
                )

            line = None
            if status == "read_start":
                line = f"[{label}] file {idx}/{total} reading {rel_path[:80]}{stats_suffix}"
            elif status == "read_done":
                chars = int(event.get("chars", 0))
                line = f"[{label}] file {idx}/{total} read_done chars={chars} file={rel_path[:80]}{stats_suffix}"
            elif status == "chunk_start":
                line = f"[{label}] file {idx}/{total} chunking {rel_path[:80]}{stats_suffix}"
            elif status == "chunk_done":
                chunks = int(event.get("chunks", 0))
                line = f"[{label}] file {idx}/{total} chunk_done chunks={chunks} file={rel_path[:80]}{stats_suffix}"
            elif status == "delete_start":
                line = f"[{label}] file {idx}/{total} replacing old chunks for {rel_path[:80]}{stats_suffix}"
            elif status == "chunked":
                chunks = int(event.get("chunks", 0))
                batches = int(event.get("batches", 0))
                line = (
                    f"[{label}] file {idx}/{total} ready_to_embed chunks={chunks} "
                    f"batches={batches} file={rel_path[:80]}{stats_suffix}"
                )
            elif status == "embedding_batch":
                batch = int(event.get("batch", 0))
                batches = int(event.get("batches", 0))
                batch_size = int(event.get("batch_size", 0))
                line = (
                    f"[{label}] file {idx}/{total} embedding batch {batch}/{batches} "
                    f"size={batch_size} file={rel_path[:80]}{stats_suffix}"
                )

            now = time.time()
            if line and (status == "embedding_batch" or (now - last_file_emit) >= 0.5):
                print(line, flush=True)
                last_file_emit = now

        elif phase == "done":
            elapsed = time.time() - start
            if state is not None and state_lock is not None:
                updated = int(getattr(stats, "updated", 0)) if stats is not None else state.updated
                deleted = int(getattr(stats, "deleted", 0)) if stats is not None else state.deleted
                errors = int(getattr(stats, "errors", 0)) if stats is not None else state.errors
                _set_reindex_status(
                    state,
                    state_lock,
                    running=False,
                    phase="done",
                    status="done",
                    updated=updated,
                    deleted=deleted,
                    errors=errors,
                )
            if stats is not None:
                print(
                    f"[{label}] complete in {_fmt_seconds(elapsed)}. "
                    f"scanned={stats.scanned} updated={stats.updated} "
                    f"deleted={stats.deleted} errors={stats.errors}"
                )
            else:
                print(f"[{label}] complete in {_fmt_seconds(elapsed)}.")

    return on_progress


def _make_background_progress_callback(label: str, state: ReindexStatus, state_lock: threading.Lock):
    last_scan_emit = 0.0
    last_file_emit = 0.0
    saw_notable_event = False

    def on_progress(event: dict) -> None:
        nonlocal last_scan_emit, last_file_emit, saw_notable_event
        phase = event.get("phase")
        stats = event.get("stats")

        if phase == "schema_upgrade":
            reason = str(event.get("reason", "schema_upgrade"))
            old_version = str(event.get("old_version", "unset"))
            new_version = str(event.get("new_version", "unset"))
            total_files = int(event.get("total_files", 0))
            _set_reindex_status(
                state,
                state_lock,
                running=True,
                owner="background",
                mode="incremental",
                phase="schema_upgrade",
                status=reason,
                total_files=total_files,
            )
            if reason == "manual_force":
                print(f"\n[{label}] starting full reindex of {total_files} file(s).\n", flush=True)
            else:
                print(
                    f"\n[{label}] index schema changed {old_version} -> {new_version}; "
                    f"forcing full reindex of {total_files} file(s).\n",
                    flush=True,
                )
            return

        if phase == "scan":
            idx = int(event.get("index", 0))
            total = max(1, int(event.get("total", 1)))
            status = str(event.get("status", ""))
            mode = str(event.get("mode", "incremental"))
            rel_path = str(event.get("rel_path", ""))
            updated = int(getattr(stats, "updated", 0)) if stats is not None else state.updated
            deleted = int(getattr(stats, "deleted", 0)) if stats is not None else state.deleted
            errors = int(getattr(stats, "errors", 0)) if stats is not None else state.errors
            _set_reindex_status(
                state,
                state_lock,
                running=True,
                owner="background",
                mode=mode,
                phase="scan",
                status=status,
                current_file=rel_path,
                current_index=idx,
                total_files=total,
                updated=updated,
                deleted=deleted,
                errors=errors,
            )

            if status == "error":
                saw_notable_event = True
                err = str(event.get("error", "unknown error"))
                print(f"\n[{label}] file error: {rel_path} -> {err}\n")
                return

            notable_statuses = {"updating", "updated", "skipped_large", "reindex_all"}
            if status not in notable_statuses:
                return

            saw_notable_event = True
            now = time.time()
            if status in ("updated", "skipped_large", "error") or (now - last_scan_emit) >= 2.0:
                print(
                    f"[{label}] mode={mode:<11} {idx}/{total} status={status:<12} "
                    f"u={updated} d={deleted} e={errors} file={rel_path[:80]}",
                    flush=True,
                )
                last_scan_emit = now
            return

        if phase == "file":
            idx = int(event.get("index", 0))
            total = max(1, int(event.get("total", 1)))
            status = str(event.get("status", ""))
            rel_path = str(event.get("rel_path", ""))
            updated = int(getattr(stats, "updated", 0)) if stats is not None else state.updated
            deleted = int(getattr(stats, "deleted", 0)) if stats is not None else state.deleted
            errors = int(getattr(stats, "errors", 0)) if stats is not None else state.errors
            _set_reindex_status(
                state,
                state_lock,
                running=True,
                owner="background",
                phase="file",
                status=status,
                current_file=rel_path,
                current_index=idx,
                total_files=total,
                updated=updated,
                deleted=deleted,
                errors=errors,
            )
            now = time.time()
            if status == "embedding_batch":
                saw_notable_event = True
                batch = int(event.get("batch", 0))
                batches = int(event.get("batches", 0))
                print(
                    f"[{label}] file {idx}/{total} embedding batch {batch}/{batches} file={rel_path[:80]}",
                    flush=True,
                )
                last_file_emit = now
            elif (now - last_file_emit) >= 2.0 and status in ("read_start", "chunk_start", "chunk_done", "chunked"):
                saw_notable_event = True
                print(f"[{label}] file {idx}/{total} status={status} file={rel_path[:80]}", flush=True)
                last_file_emit = now
            return

        if phase == "done":
            updated = int(getattr(stats, "updated", 0)) if stats is not None else state.updated
            deleted = int(getattr(stats, "deleted", 0)) if stats is not None else state.deleted
            errors = int(getattr(stats, "errors", 0)) if stats is not None else state.errors
            _set_reindex_status(
                state,
                state_lock,
                running=False,
                owner="",
                phase="done",
                status="done",
                updated=updated,
                deleted=deleted,
                errors=errors,
                current_file="",
                current_index=0,
                total_files=0,
            )
            if saw_notable_event or updated > 0 or deleted > 0 or errors > 0:
                total_files = int(event.get("total_files", 0))
                print(
                    f"[{label}] complete scanned={total_files} updated={updated} "
                    f"deleted={deleted} errors={errors}",
                    flush=True,
                )

    return on_progress


def _embed_query(client: OpenAI, model: str, text: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def _list_indexed_files(manifest: Manifest) -> list[str]:
    return sorted(manifest.all_paths())


def _format_indexed_files(files: list[str], max_items: int = 100) -> str:
    if not files:
        return "No files are indexed yet."
    shown = files[:max_items]
    lines = [f"Indexed files: {len(files)} total"]
    lines.extend(f"- {p}" for p in shown)
    if len(files) > max_items:
        lines.append(f"... ({len(files) - max_items} more)")
    return "\n".join(lines)


def _format_eval_candidates(candidates: list[dict], max_items: int = 20) -> str:
    if not candidates:
        return (
            "No eval candidates yet.\n"
            "- Restart the bot so search logging is active.\n"
            "- Use natural-language note searches for a while.\n"
            "- Run /eval-candidates again."
        )

    shown = candidates[:max_items]
    lines = [f"Eval candidates: {len(candidates)}"]
    for idx, item in enumerate(shown, start=1):
        query = str(item.get("query", "")).strip()
        expected_paths = item.get("expected_paths", [])
        top_path = expected_paths[0] if isinstance(expected_paths, list) and expected_paths else "<unknown>"
        query_type = str(item.get("expected_query_type", "mixed"))
        lines.append(f"{idx}. {query}")
        lines.append(f"   top_path: {top_path}")
        lines.append(f"   query_type: {query_type}")
        top_snippet = str(item.get("top_observed_snippet", "")).strip()
        if top_snippet:
            lines.append(f"   observed: {top_snippet}")
    if len(candidates) > max_items:
        lines.append(f"... ({len(candidates) - max_items} more)")
    lines.append("Use /eval-promote 1 2 5-8 or /eval-promote all after review.")
    return "\n".join(lines)


def _eval_file_paths(cfg) -> tuple[Path, Path]:
    return cfg.data_dir / "search_queries.jsonl", cfg.data_dir / "eval_candidates.json"


def _refresh_eval_candidates(cfg) -> list[dict]:
    search_log_path, candidate_path = _eval_file_paths(cfg)
    candidates = build_eval_candidates(
        log_path=search_log_path,
        existing_eval_path=cfg.data_dir / "eval_queries.json",
        limit=50,
    )
    write_eval_candidate_file(candidate_path, candidates)
    return candidates


def _promote_eval_candidates_from_text(user_text: str, cfg) -> str:
    _, candidate_path = _eval_file_paths(cfg)
    eval_path = cfg.data_dir / "eval_queries.json"
    candidates = load_eval_candidates(candidate_path)
    if not candidates:
        return "No eval candidates are available to promote."

    normalized = user_text.strip()
    if normalized.lower().startswith("/eval-promote"):
        suffix = normalized[len("/eval-promote"):].strip()
    elif normalized.lower().startswith("promote eval candidates"):
        suffix = normalized[len("promote eval candidates"):].strip()
    else:
        suffix = ""
    args = suffix.split() if suffix else []
    try:
        selections = parse_candidate_selections(args, len(candidates))
    except ValueError:
        return "Could not parse candidate selection. Use /eval-promote 1 2 5-8 or /eval-promote all."

    promoted, remaining = promote_eval_candidates(
        candidate_path=candidate_path,
        eval_path=eval_path,
        selections=selections,
    )
    if promoted == 0:
        return "No candidates were promoted."
    return (
        f"Promoted {promoted} eval candidate(s) into {eval_path.name}.\n"
        f"Remaining candidates: {remaining}"
    )


def _count_ipv4_addresses(text: str) -> tuple[int, int]:
    total = 0
    unique: set[str] = set()
    for m in _IPV4_RE.finditer(text):
        ip = m.group(0)
        parts = ip.split(".")
        if len(parts) != 4:
            continue
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            continue
        if any(n < 0 or n > 255 for n in nums):
            continue
        total += 1
        unique.add(ip)
    return total, len(unique)


def _count_emails(text: str) -> tuple[int, int]:
    matches = [m.group(0) for m in _EMAIL_RE.finditer(text)]
    unique = {m.lower() for m in matches}
    return len(matches), len(unique)


def _count_urls(text: str) -> tuple[int, int]:
    matches = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?)]}")
        if url:
            matches.append(url)
    unique = set(matches)
    return len(matches), len(unique)


def _count_api_key_like(text: str) -> tuple[int, int]:
    matches: list[str] = []
    for rx in _API_KEY_RE_LIST:
        matches.extend(m.group(0) for m in rx.finditer(text))
    unique = set(matches)
    return len(matches), len(unique)


def _luhn_ok(digits: str) -> bool:
    total = 0
    alt = False
    for ch in reversed(digits):
        d = ord(ch) - ord("0")
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return (total % 10) == 0


def _count_credit_card_like(text: str) -> tuple[int, int]:
    normalized: list[str] = []
    for m in _CARD_CANDIDATE_RE.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 13 or len(digits) > 19:
            continue
        if _luhn_ok(digits):
            normalized.append(digits)
    unique = set(normalized)
    return len(normalized), len(unique)


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).lower()


def _count_sql_statements(text: str) -> tuple[int, int]:
    matches: list[str] = []

    # Multi-line / block SQL ending in semicolon.
    for m in _SQL_BLOCK_RE.finditer(text):
        stmt = m.group(1).strip()
        low = stmt.lower()
        if low.startswith(("select", "with")) and " from " not in low:
            continue
        matches.append(stmt)

    # One-line SQL commands without semicolon.
    for m in _SQL_LINE_RE.finditer(text):
        stmt = m.group(0).strip()
        low = stmt.lower()
        if low.startswith(("select", "with")) and " from " not in low:
            continue
        matches.append(stmt)

    normalized = [_normalize_sql(s) for s in matches if s]
    unique = set(normalized)
    return len(normalized), len(unique)


def _is_density_question(low: str) -> bool:
    asks_files = "which file" in low or "what file" in low or "which files" in low or "what files" in low
    asks_density = any(p in low for p in ("a lot", "lots", "many", "most", "highest", "top"))
    return asks_files and asks_density


def _detect_analytic_target(user_text: str) -> str | None:
    low = user_text.lower()
    if not _is_density_question(low):
        return None
    if "ip address" in low or "ip addresses" in low or re.search(r"\bips?\b", low):
        return "ip"
    if "email address" in low or "email addresses" in low or re.search(r"\bemails?\b", low):
        return "email"
    if re.search(r"\burls?\b", low) or re.search(r"\blinks?\b", low) or re.search(r"\bwebsites?\b", low):
        return "url"
    if "api key" in low or "api keys" in low or "access key" in low or "access keys" in low:
        return "api_key"
    if "credit card" in low or "card number" in low or "card numbers" in low:
        return "credit_card"
    if "sql" in low or "sql query" in low or "sql queries" in low or "sql statement" in low or "sql statements" in low:
        return "sql"
    if "query" in low or "queries" in low or "statement" in low or "statements" in low:
        return "sql"
    return None


def _format_analytic_density_results(cfg, manifest: Manifest, target: str, max_items: int = 10) -> str:
    analyzers = {
        "ip": ("IPv4 address", _count_ipv4_addresses),
        "email": ("email address", _count_emails),
        "url": ("URL", _count_urls),
        "api_key": ("API-key-like string", _count_api_key_like),
        "credit_card": ("credit-card-like number", _count_credit_card_like),
        "sql": ("SQL statement", _count_sql_statements),
    }
    label, counter = analyzers[target]
    rows: list[tuple[str, int, int]] = []
    for rel_path in _list_indexed_files(manifest):
        abs_path = cfg.doc_root / rel_path
        if not abs_path.exists() or not abs_path.is_file():
            continue
        total, unique = counter(_read_note_text(abs_path))
        if total > 0:
            rows.append((rel_path, total, unique))

    if not rows:
        return f"No indexed files contain {label}s."

    rows.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    shown = rows[:max_items]
    lines = [f"Top files by {label} count ({len(rows)} file(s) with at least one match):"]
    lines.extend(f"- {rel_path}: {total} match(es), {unique} unique" for rel_path, total, unique in shown)
    if len(rows) > max_items:
        lines.append(f"... ({len(rows) - max_items} more)")
    return "\n".join(lines)


def _handle_meta_query(user_text: str, cfg, manifest: Manifest) -> str | None:
    text = user_text.strip()
    low = text.lower()

    analytic_target = _detect_analytic_target(text)
    if analytic_target:
        return _format_analytic_density_results(cfg, manifest, analytic_target)

    return None


def _run_general_chat(user_text: str, turns: list[ChatTurn], client: OpenAI, cfg) -> str:
    messages = [{"role": "system", "content": GENERAL_CHAT_PROMPT}]
    for tr in turns:
        messages.append({"role": tr.role, "content": tr.content})
    messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model=cfg.chat_model,
        messages=messages,
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def _background_index_loop(
    stop_event: threading.Event,
    index_lock: threading.Lock,
    cfg,
    client,
    manifest,
    store,
    reindex_state: ReindexStatus,
    reindex_state_lock: threading.Lock,
):
    interval = max(1, int(cfg.scan_interval_minutes)) * 60
    while not stop_event.is_set():
        try:
            with index_lock:
                _set_reindex_status(
                    reindex_state,
                    reindex_state_lock,
                    running=True,
                    owner="background",
                    mode="incremental",
                    phase="starting",
                    status="starting",
                    started_at=time.time(),
                )
                stats = run_index_once(
                    client=client,
                    doc_root=cfg.doc_root,
                    include_ext=cfg.include_ext,
                    manifest=manifest,
                    store=store,
                    embedding_model=cfg.embedding_model,
                    chunk_chars=cfg.chunk_chars,
                    chunk_overlap=cfg.chunk_overlap,
                    max_file_size_mb=cfg.max_file_size_mb,
                    max_chunks_per_file=cfg.max_chunks_per_file,
                    progress_callback=_make_background_progress_callback("index-bg", reindex_state, reindex_state_lock),
                )
                _set_reindex_status(
                    reindex_state,
                    reindex_state_lock,
                    running=False,
                    owner="",
                    mode="incremental",
                    phase="idle",
                    status="idle",
                    current_file="",
                    current_index=0,
                    total_files=0,
                )
        except Exception as e:
            _set_reindex_status(
                reindex_state,
                reindex_state_lock,
                running=False,
                owner="",
                phase="error",
                status="error",
                errors=reindex_state.errors + 1,
            )
            print(f"\n[index] error: {e}\n")

        slept = 0
        while slept < interval and not stop_event.is_set():
            time.sleep(1)
            slept += 1


def main(config_path: str | Path = "config.yaml"):
    cfg = load_config(config_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    manifest = Manifest(cfg.manifest_path)
    store = VectorStore(cfg.index_dir, collection_name="notes")

    history_store = ChatHistory(cfg.chat_history_path)
    turns = history_store.load()
    search_log_path = cfg.data_dir / "search_queries.jsonl"

    stop_event = threading.Event()
    index_lock = threading.Lock()
    reindex_state = ReindexStatus()
    reindex_state_lock = threading.Lock()

    t = threading.Thread(
        target=_background_index_loop,
        args=(stop_event, index_lock, cfg, client, manifest, store, reindex_state, reindex_state_lock),
        daemon=True
    )
    t.start()

    print("Notes bot ready.")
    print("Ask naturally about your notes or ask general questions.")
    print("Commands: /help, /clear, /reindex, /reindex-force, /reindex-status, /eval-candidates, /eval-promote, /indexed, /exit")
    print("Indexing runs in the background periodically.\n")

    def recent_turns():
        return turns[-2 * cfg.max_history_turns:] if cfg.max_history_turns > 0 else []

    try:
        while True:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue

            decision = route_user_input(user)

            if decision.mode == "command":
                if decision.command_name == "exit":
                    break

                if decision.command_name == "help":
                    print("\n" + HELP_TEXT)
                    continue

                if decision.command_name == "clear":
                    turns.clear()
                    history_store.clear()
                    print("Cleared chat context.\n")
                    continue

                if decision.command_name == "indexed":
                    indexed_text = _format_indexed_files(_list_indexed_files(manifest))
                    print("\n" + indexed_text + "\n")
                    turns.append(ChatTurn(role="user", content=user, ts=time.time()))
                    turns.append(ChatTurn(role="assistant", content=indexed_text, ts=time.time()))
                    history_store.append("user", user)
                    history_store.append("assistant", indexed_text)
                    continue

                if decision.command_name == "reindex_status":
                    status_text = _format_reindex_status(_get_reindex_status(reindex_state, reindex_state_lock))
                    print("\n" + status_text + "\n")
                    turns.append(ChatTurn(role="user", content=user, ts=time.time()))
                    turns.append(ChatTurn(role="assistant", content=status_text, ts=time.time()))
                    history_store.append("user", user)
                    history_store.append("assistant", status_text)
                    continue

                if decision.command_name == "eval_candidates":
                    candidate_text = _format_eval_candidates(_refresh_eval_candidates(cfg))
                    print("\n" + candidate_text + "\n")
                    turns.append(ChatTurn(role="user", content=user, ts=time.time()))
                    turns.append(ChatTurn(role="assistant", content=candidate_text, ts=time.time()))
                    history_store.append("user", user)
                    history_store.append("assistant", candidate_text)
                    continue

                if decision.command_name == "eval_promote":
                    promote_text = _promote_eval_candidates_from_text(user, cfg)
                    print("\n" + promote_text + "\n")
                    turns.append(ChatTurn(role="user", content=user, ts=time.time()))
                    turns.append(ChatTurn(role="assistant", content=promote_text, ts=time.time()))
                    history_store.append("user", user)
                    history_store.append("assistant", promote_text)
                    continue

                if decision.command_name not in ("reindex", "reindex_force"):
                    print("\nUnknown command.\n")
                    continue

                is_force = decision.command_name == "reindex_force"
                if is_force:
                    print("Reindexing (full rebuild) ...")
                else:
                    print("Reindexing (incremental) ...")
                locked = index_lock.acquire(blocking=False)
                if not locked:
                    print("Indexer is busy in background; waiting for current pass to finish ...")
                    wait_start = time.time()
                    last_wait_emit = 0.0
                    while True:
                        locked = index_lock.acquire(timeout=1.0)
                        if locked:
                            break
                        now = time.time()
                        if (now - last_wait_emit) >= 2.0:
                            st = _get_reindex_status(reindex_state, reindex_state_lock)
                            waited = _fmt_seconds(now - wait_start)
                            location = st.current_file[:80] if st.current_file else "<starting>"
                            progress = f"{st.current_index}/{st.total_files}" if st.total_files > 0 else "--/--"
                            print(
                                f"[wait] owner={st.owner or 'unknown'} mode={st.mode} phase={st.phase} "
                                f"status={st.status} progress={progress} waited={waited} "
                                f"u={st.updated} d={st.deleted} e={st.errors} file={location}",
                                flush=True,
                            )
                            last_wait_emit = now
                try:
                    _set_reindex_status(
                        reindex_state,
                        reindex_state_lock,
                        running=True,
                        owner="manual",
                        mode="full" if is_force else "incremental",
                        phase="starting",
                        status="starting",
                        started_at=time.time(),
                    )
                    stats = run_index_once(
                        client=client,
                        doc_root=cfg.doc_root,
                        include_ext=cfg.include_ext,
                        manifest=manifest,
                        store=store,
                        embedding_model=cfg.embedding_model,
                        chunk_chars=cfg.chunk_chars,
                        chunk_overlap=cfg.chunk_overlap,
                        max_file_size_mb=cfg.max_file_size_mb,
                        max_chunks_per_file=cfg.max_chunks_per_file,
                        force_reindex=is_force,
                        progress_callback=_make_progress_callback("reindex", reindex_state, reindex_state_lock),
                    )
                finally:
                    _set_reindex_status(
                        reindex_state,
                        reindex_state_lock,
                        running=False,
                        owner="",
                        mode="incremental",
                        phase="idle",
                        status="idle",
                        current_file="",
                        current_index=0,
                        total_files=0,
                    )
                    index_lock.release()
                print(f"Done. scanned={stats.scanned} updated={stats.updated} deleted={stats.deleted} errors={stats.errors}\n")
                continue

            meta_answer = _handle_meta_query(user, cfg, manifest)
            if meta_answer is not None:
                print("\n" + meta_answer + "\n")
                turns.append(ChatTurn(role="user", content=user, ts=time.time()))
                turns.append(ChatTurn(role="assistant", content=meta_answer, ts=time.time()))
                history_store.append("user", user)
                history_store.append("assistant", meta_answer)
                continue

            if decision.mode == "notes_search":
                results = search_notes(
                    query=user,
                    client=client,
                    cfg=cfg,
                    manifest=manifest,
                    store=store,
                    limit=cfg.top_k,
                )
                try:
                    append_search_log(search_log_path, query=user, results=results)
                except OSError:
                    pass
                answer = format_search_results(results)
            else:
                answer = _run_general_chat(user, recent_turns(), client, cfg)

            print("\n" + answer + "\n")

            # Persist turns
            turns.append(ChatTurn(role="user", content=user, ts=time.time()))
            turns.append(ChatTurn(role="assistant", content=answer, ts=time.time()))
            history_store.append("user", user)
            history_store.append("assistant", answer)

    finally:
        stop_event.set()
        
