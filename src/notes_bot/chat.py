from __future__ import annotations
import re
import threading
import time
from pathlib import Path

from openai import OpenAI

from .config import load_config
from .manifest import Manifest
from .store import VectorStore
from .indexer import run_index_once
from .history import ChatHistory, ChatTurn
from .runtime import validate_runtime_config
from .retrieval import RetrievalService, format_passage_results
from .note_view import get_note_excerpt, safe_doc_path
from .intent import detect_query_intent
from .doc_roots import split_virtual_rel_path

UNSUPPORTED_ANSWER = "I can't find that in your notes."

HELP_TEXT = (
    "Commands:\n"
    "- /help show this help\n"
    "- /clear clear chat context/history\n"
    "- /reindex run incremental indexing now\n"
    "- /indexed list indexed files from manifest\n"
    "- /roots show indexed file counts by configured doc root\n"
    "- /status show manifest status summary and recent non-indexed files\n"
    "- /find <term> search indexed files by filename or text\n"
    "- /findfresh <term> search indexed files but exclude stale entries\n"
    "- /findname <term> search filename only\n"
    "- /findtext <term> search text content only\n"
    "- /search <term> search note passages and show excerpts\n"
    "- /searchfresh <term> search passages from fresh indexed content only\n"
    "- /open <n> open surrounding context for a result from the last /search\n"
    "- /open <path[:start-end]> open a note or line range with context\n"
    "- /exit quit\n"
    "\n"
    "Scope prompts/results to a configured doc root with a leading @rootname.\n"
    "- example: @personal1 what does this say about prayer?\n"
    "- example: @personal2 /search authentication failure\n"
    "\n"
    "Auto routing:\n"
    "- short keyword-style queries default to passage search\n"
    "- question-style queries default to grounded answering\n"
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


def _make_progress_callback(label: str):
    start = time.time()
    bar_width = 28
    last_emit = 0.0

    def on_progress(event: dict) -> None:
        nonlocal last_emit
        phase = event.get("phase")
        stats = event.get("stats")

        if phase == "scan":
            idx = int(event.get("index", 0))
            total = max(1, int(event.get("total", 1)))
            status = str(event.get("status", ""))
            rel_path = str(event.get("rel_path", ""))

            elapsed = max(0.001, time.time() - start)
            rate = idx / elapsed
            eta = (total - idx) / rate if rate > 0 else float("inf")
            pct = idx / total
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)

            prefix = (
                f"[{label}] [{bar}] {idx}/{total} {pct*100:5.1f}% "
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
            pct = idx / total
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"[{label}] deleting [{bar}] {idx}/{total} {pct*100:5.1f}% file={rel_path[:80]}", flush=True)

        elif phase == "done":
            elapsed = time.time() - start
            if stats is not None:
                print(
                    f"[{label}] complete in {_fmt_seconds(elapsed)}. "
                    f"scanned={stats.scanned} updated={stats.updated} "
                    f"deleted={stats.deleted} errors={stats.errors}"
                )
            else:
                print(f"[{label}] complete in {_fmt_seconds(elapsed)}.")

    return on_progress


def _make_background_progress_callback(label: str):
    def on_progress(event: dict) -> None:
        phase = event.get("phase")

        if phase == "scan":
            status = str(event.get("status", ""))
            rel_path = str(event.get("rel_path", ""))

            if status == "error":
                err = str(event.get("error", "unknown error"))
                print(f"\n[{label}] file error: {rel_path} -> {err}\n")
                return

    return on_progress


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


def _format_manifest_status(manifest: Manifest, max_items: int = 20) -> str:
    counts = manifest.counts_by_status()
    if not counts:
        return "No files are tracked in the manifest yet."

    order = ["indexed", "stale", "skipped_large", "error"]
    summary = ", ".join(f"{status}={counts.get(status, 0)}" for status in order if status in counts)
    lines = [f"Manifest status: {summary}"]

    non_indexed = [
        st for st in sorted(manifest.iter_all(), key=lambda x: (x.status, x.rel_path))
        if st.status != "indexed"
    ]
    for st in non_indexed[:max_items]:
        detail = f"- {st.rel_path} [{st.status}]"
        if st.chunk_count:
            detail += f" chunks={st.chunk_count}"
        if st.last_success_at:
            age_seconds = max(0, int(time.time() - st.last_success_at))
            detail += f" last_success={_fmt_seconds(age_seconds)} ago"
        if st.last_error:
            detail += f" -> {st.last_error}"
        lines.append(detail)
    if len(non_indexed) > max_items:
        lines.append(f"... ({len(non_indexed) - max_items} more)")
    return "\n".join(lines)


def _format_root_counts(cfg, manifest: Manifest) -> str:
    root_names = [root.name for root in cfg.doc_roots]
    counts: dict[str, dict[str, int]] = {
        root_name: {"total": 0, "indexed": 0, "stale": 0, "skipped_large": 0, "error": 0}
        for root_name in root_names
    }
    unknown = {"total": 0, "indexed": 0, "stale": 0, "skipped_large": 0, "error": 0}

    for st in manifest.iter_all():
        try:
            root, _ = split_virtual_rel_path(cfg.doc_roots, st.rel_path)
            bucket = counts.setdefault(
                root.name,
                {"total": 0, "indexed": 0, "stale": 0, "skipped_large": 0, "error": 0},
            )
        except ValueError:
            bucket = unknown
        bucket["total"] += 1
        bucket[st.status] = bucket.get(st.status, 0) + 1

    lines = ["Doc roots:"]
    for root_name in root_names:
        bucket = counts[root_name]
        lines.append(
            f"- {root_name}: total={bucket['total']} indexed={bucket['indexed']} "
            f"stale={bucket['stale']} skipped_large={bucket['skipped_large']} error={bucket['error']}"
        )
    if unknown["total"]:
        lines.append(
            f"- unknown: total={unknown['total']} indexed={unknown['indexed']} "
            f"stale={unknown['stale']} skipped_large={unknown['skipped_large']} error={unknown['error']}"
        )
    return "\n".join(lines)


def _format_single_root_count(cfg, manifest: Manifest, root_name: str) -> str | None:
    target = root_name.strip().lower()
    root_names = {root.name for root in cfg.doc_roots}
    if target not in root_names:
        return None

    bucket = {"total": 0, "indexed": 0, "stale": 0, "skipped_large": 0, "error": 0}
    prefix = f"{target}/"
    for st in manifest.iter_all():
        rel_path = st.rel_path.replace("\\", "/")
        if len(cfg.doc_roots) == 1:
            matches = target == cfg.doc_roots[0].name
        else:
            matches = rel_path.startswith(prefix)
        if not matches:
            continue
        bucket["total"] += 1
        bucket[st.status] = bucket.get(st.status, 0) + 1

    return (
        f"Doc root {target}: total={bucket['total']} indexed={bucket['indexed']} "
        f"stale={bucket['stale']} skipped_large={bucket['skipped_large']} error={bucket['error']}"
    )


def _format_search_results(term: str, mode: str, matches: list[dict], max_items: int = 100) -> str:
    if not matches:
        return f"No indexed files matched '{term}'."

    mode_label = {"filename": "filename", "text": "text", "both": "filename or text"}.get(mode, mode)
    shown = matches[:max_items]
    lines = [f"Matched {len(matches)} file(s) for '{term}' in {mode_label}:"]
    for item in shown:
        rel_path = item["rel_path"]
        tags = []
        status = item.get("status")
        if item.get("filename_match"):
            tags.append("filename")
        if item.get("text_match"):
            tags.append("text")
        if item.get("section_hits"):
            tags.append(f"sections={item['section_hits']}")
        if status and status != "indexed":
            tags.append(status)
        best = item.get("best_section_path") or item.get("best_title") or ""
        suffix = f" ({', '.join(tags)})" if tags else ""
        if best:
            lines.append(f"- {rel_path}{suffix} -> {best}")
        else:
            lines.append(f"- {rel_path}{suffix}")
    if len(matches) > max_items:
        lines.append(f"... ({len(matches) - max_items} more)")
    return "\n".join(lines)


def _format_open_excerpt(excerpt: dict) -> str:
    header = (
        f"{excerpt['rel_path']} lines {excerpt['display_start_line']}-{excerpt['display_end_line']} "
        f"(focus {excerpt['start_line']}-{excerpt['end_line']})"
    )
    body_lines = []
    start = excerpt["display_start_line"]
    for offset, line in enumerate(excerpt["text"].splitlines(), start=start):
        marker = ">" if excerpt["start_line"] <= offset <= excerpt["end_line"] else " "
        body_lines.append(f"{marker} {offset:>5} | {line}")
    if excerpt["truncated"]:
        body_lines.append("... [truncated]")
    return header + "\n" + "\n".join(body_lines)


def _format_scope_banner(root_name: str | None) -> str:
    if not root_name:
        return ""
    return f"Scope: {root_name}\n\n"


def _parse_root_scope(text: str, cfg) -> tuple[str | None, str]:
    stripped = text.strip()
    if not stripped.startswith("@"):
        return None, stripped
    token, _, remainder = stripped.partition(" ")
    root_name = token[1:].strip().lower()
    valid_roots = {root.name for root in cfg.doc_roots}
    if not root_name or root_name not in valid_roots:
        raise ValueError("Unknown doc root. Use /roots to list available roots.")
    remainder = remainder.strip()
    if not remainder:
        raise ValueError(f"Missing query after @{root_name}.")
    return root_name, remainder


def _format_cli_passage_results(query: str, results: dict, root_name: str | None = None) -> str:
    formatted = format_passage_results(query, results)
    items = results.get("results", [])
    if items:
        formatted += "\n\nUse /open <n> to inspect one of these results."
    return _format_scope_banner(root_name) + formatted


def _run_auto_query(
    user: str,
    retrieval: RetrievalService,
    *,
    answer_top_k: int,
    max_sources_chars: int,
    root_name: str | None = None,
) -> tuple[str, list[dict]]:
    intent = detect_query_intent(user)
    if intent == "passage_search":
        results = retrieval.retrieve_passages(
            query=user,
            top_k=8,
            include_text=True,
            max_chars=8000,
            root_name=root_name,
        )
        return _format_cli_passage_results(user, results, root_name=root_name), results.get("results", [])

    answer_payload = retrieval.answer_question(
        question=user,
        top_k=answer_top_k,
        max_sources_chars=max_sources_chars,
        root_name=root_name,
    )
    answer = answer_payload["answer"]
    if answer.strip() == UNSUPPORTED_ANSWER:
        results = retrieval.retrieve_passages(
            query=user,
            top_k=8,
            include_text=True,
            max_chars=8000,
            root_name=root_name,
        )
        return _format_cli_passage_results(user, results, root_name=root_name), results.get("results", [])
    return _format_scope_banner(root_name) + answer, []


_OPEN_RANGE_RE = re.compile(r"^(?P<path>.+?)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$")


def _parse_open_target(text: str) -> tuple[str, int | None, int | None] | None:
    spec = text.strip()
    if not spec:
        return None
    m = _OPEN_RANGE_RE.match(spec)
    if not m:
        return None
    rel_path = m.group("path").strip()
    start = int(m.group("start")) if m.group("start") else None
    end = int(m.group("end")) if m.group("end") else start
    return rel_path, start, end


def _extract_mention_term(user_text: str) -> tuple[str, str] | None:
    t = user_text.strip()
    low = t.lower()

    patterns = [
        (r"^(?:what|which)\s+files\s+mention\s+(.+?)\s+in\s+(?:the\s+)?filename\??$", "filename"),
        (r"^(?:what|which)\s+files\s+mention\s+(.+?)\s+in\s+text\??$", "text"),
        (r"^(?:what|which)\s+files\s+mention\s+(.+?)\??$", "both"),
        (r"^find\s+files\s+mentioning\s+(.+?)\s+in\s+(?:the\s+)?filename\??$", "filename"),
        (r"^find\s+files\s+mentioning\s+(.+?)\s+in\s+text\??$", "text"),
        (r"^find\s+files\s+mentioning\s+(.+?)\??$", "both"),
    ]
    for pat, mode in patterns:
        m = re.match(pat, low, flags=re.IGNORECASE)
        if m:
            term = m.group(1).strip().strip("'\"")
            if term:
                return term, mode
    return None


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
        try:
            abs_path = safe_doc_path(cfg.doc_roots, rel_path)
        except ValueError:
            continue
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


def _handle_meta_query(
    user_text: str,
    cfg,
    manifest: Manifest,
    retrieval: RetrievalService,
    root_name: str | None = None,
) -> str | None:
    text = user_text.strip()
    low = text.lower()

    indexed_phrases = (
        "what files have been indexed",
        "which files have been indexed",
        "what files are indexed",
        "which files are indexed",
        "show indexed files",
        "list indexed files",
    )
    if any(p in low for p in indexed_phrases):
        files = _list_indexed_files(manifest)
        if root_name and len(cfg.doc_roots) > 1:
            files = [p for p in files if p.startswith(f"{root_name}/")]
        return _format_scope_banner(root_name) + _format_indexed_files(files)

    if low in ("/status", "show index status", "show manifest status", "what failed to index", "which files failed to index"):
        return _format_manifest_status(manifest)

    if low in ("/roots", "show roots", "show doc roots", "show root counts", "list roots", "list doc roots"):
        return _format_root_counts(cfg, manifest)

    for root in cfg.doc_roots:
        root_name = root.name.lower()
        if low in (
            f"how many docs in {root_name}",
            f"how many documents in {root_name}",
            f"how many files in {root_name}",
            f"how many docs are in {root_name}",
            f"how many documents are in {root_name}",
            f"how many files are in {root_name}",
            f"how many docs are indexed for {root_name}",
            f"how many documents are indexed for {root_name}",
            f"how many files are indexed for {root_name}",
            f"how many docs were indexed for {root_name}",
            f"how many documents were indexed for {root_name}",
            f"how many files were indexed for {root_name}",
        ):
            return _format_single_root_count(cfg, manifest, root_name)

    analytic_target = _detect_analytic_target(text)
    if analytic_target:
        return _format_analytic_density_results(cfg, manifest, analytic_target)

    mention = _extract_mention_term(text)
    if mention:
        term, mode = mention
        matches = retrieval.search_files(term, mode=mode, limit=100, root_name=root_name)
        return _format_scope_banner(root_name) + _format_search_results(term, mode, matches)

    if low.startswith("/indexed"):
        files = _list_indexed_files(manifest)
        if root_name and len(cfg.doc_roots) > 1:
            files = [p for p in files if p.startswith(f"{root_name}/")]
        return _format_scope_banner(root_name) + _format_indexed_files(files)

    if low.startswith("/findfresh "):
        term = text[11:].strip()
        mode = "both"
        matches = retrieval.search_files(term, mode=mode, limit=100, freshness_mode="indexed_only", root_name=root_name)
        return _format_scope_banner(root_name) + _format_search_results(term, mode, matches)

    if low.startswith("/find "):
        term = text[6:].strip()
        mode = "both"
        matches = retrieval.search_files(term, mode=mode, limit=100, root_name=root_name)
        return _format_scope_banner(root_name) + _format_search_results(term, mode, matches)

    if low.startswith("/findname "):
        term = text[10:].strip()
        matches = retrieval.search_files(term, mode="filename", limit=100, root_name=root_name)
        return _format_scope_banner(root_name) + _format_search_results(term, "filename", matches)

    if low.startswith("/findtext "):
        term = text[10:].strip()
        matches = retrieval.search_files(term, mode="text", limit=100, root_name=root_name)
        return _format_scope_banner(root_name) + _format_search_results(term, "text", matches)

    return None


def _background_index_loop(stop_event: threading.Event, index_lock: threading.Lock, cfg, client, manifest, store):
    interval = max(1, int(cfg.scan_interval_minutes)) * 60
    initial_delay = max(0, int(cfg.background_index_start_delay_seconds))
    slept = 0
    while slept < initial_delay and not stop_event.is_set():
        time.sleep(1)
        slept += 1

    while not stop_event.is_set():
        try:
            with index_lock:
                stats = run_index_once(
                    client=client,
                    doc_roots=cfg.doc_roots,
                    include_ext=cfg.include_ext,
                    manifest=manifest,
                    store=store,
                    embedding_model=cfg.embedding_model,
                    chunk_chars=cfg.chunk_chars,
                    chunk_overlap=cfg.chunk_overlap,
                    max_file_size_mb=cfg.max_file_size_mb,
                    max_chunks_per_file=cfg.max_chunks_per_file,
                    retry_error_after_minutes=cfg.retry_error_after_minutes,
                    progress_callback=_make_background_progress_callback("index-bg"),
                )
        except Exception as e:
            print(f"\n[index] error: {e}\n")

        slept = 0
        while slept < interval and not stop_event.is_set():
            time.sleep(1)
            slept += 1


def main(config_path: str | Path = "config.yaml", enable_background: bool = True):
    cfg = load_config(config_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    validate_runtime_config(cfg)

    client = OpenAI()
    manifest = Manifest(cfg.manifest_path)
    store = VectorStore(cfg.index_dir, collection_name="notes")
    retrieval = RetrievalService(
        store=store,
        manifest=manifest,
        client=client,
        embedding_model=cfg.embedding_model,
        chat_model=cfg.chat_model,
        doc_roots=cfg.doc_roots,
        answer_context_before_lines=cfg.answer_context_before_lines,
        answer_context_after_lines=cfg.answer_context_after_lines,
        adjacent_chunk_window=cfg.adjacent_chunk_window,
    )

    history_store = ChatHistory(cfg.chat_history_path)
    turns = history_store.load()

    stop_event = threading.Event()
    index_lock = threading.Lock()

    if enable_background:
        t = threading.Thread(
            target=_background_index_loop,
            args=(stop_event, index_lock, cfg, client, manifest, store),
            daemon=True
        )
        t.start()

    print("Notes bot ready.")
    print("Commands: /clear, /reindex, /indexed, /roots, /status, /find <term>, /findfresh <term>, /findname <term>, /findtext <term>, /search <term>, /searchfresh <term>, /open <...>, /exit")
    if enable_background:
        print("Indexing runs in the background periodically.\n")
    else:
        print("Background indexing disabled for this session.\n")
    last_search_results: list[dict] = []

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

            try:
                root_name, scoped_user = _parse_root_scope(user, cfg)
            except ValueError as e:
                print(f"{e}\n")
                continue

            if scoped_user.lower() == "/exit":
                break

            if scoped_user.lower() == "/help":
                print("\n" + HELP_TEXT)
                continue

            if scoped_user.lower() == "/clear":
                turns.clear()
                history_store.clear()
                print("Cleared chat context.\n")
                continue

            if scoped_user.lower() == "/reindex":
                print("Reindexing (incremental) ...")
                locked = index_lock.acquire(blocking=False)
                if not locked:
                    print("Indexer is busy in background; waiting for current pass to finish ...")
                    index_lock.acquire()
                try:
                    try:
                        stats = run_index_once(
                            client=client,
                            doc_roots=cfg.doc_roots,
                            include_ext=cfg.include_ext,
                            manifest=manifest,
                            store=store,
                            embedding_model=cfg.embedding_model,
                            chunk_chars=cfg.chunk_chars,
                            chunk_overlap=cfg.chunk_overlap,
                            max_file_size_mb=cfg.max_file_size_mb,
                            max_chunks_per_file=cfg.max_chunks_per_file,
                            retry_error_after_minutes=0,
                            progress_callback=_make_progress_callback("reindex"),
                        )
                    except Exception as e:
                        print(f"Reindex failed: {e}\n")
                        continue
                finally:
                    index_lock.release()
                print(f"Done. scanned={stats.scanned} updated={stats.updated} deleted={stats.deleted} errors={stats.errors}\n")
                continue

            if scoped_user.lower().startswith("/search "):
                query_text = scoped_user[8:].strip()
                if not query_text:
                    print("Usage: /search <term>\n")
                    continue
                results = retrieval.retrieve_passages(
                    query=query_text,
                    top_k=8,
                    include_text=True,
                    max_chars=8000,
                    root_name=root_name,
                )
                last_search_results = results.get("results", [])
                print("\n" + _format_cli_passage_results(query_text, results, root_name=root_name) + "\n")
                continue

            if scoped_user.lower().startswith("/searchfresh "):
                query_text = scoped_user[13:].strip()
                if not query_text:
                    print("Usage: /searchfresh <term>\n")
                    continue
                results = retrieval.retrieve_passages(
                    query=query_text,
                    top_k=8,
                    include_text=True,
                    max_chars=8000,
                    freshness_mode="indexed_only",
                    root_name=root_name,
                )
                last_search_results = results.get("results", [])
                print("\n" + _format_cli_passage_results(query_text, results, root_name=root_name) + "\n")
                continue

            if scoped_user.lower().startswith("/open "):
                spec = scoped_user[6:].strip()
                if not spec:
                    print("Usage: /open <n> or /open <path[:start-end]>\n")
                    continue

                excerpt = None
                if spec.isdigit():
                    idx = int(spec)
                    if idx < 1 or idx > len(last_search_results):
                        print("No such search result. Run /search first and choose a listed result number.\n")
                        continue
                    item = last_search_results[idx - 1]
                    try:
                        excerpt = get_note_excerpt(
                            doc_roots=cfg.doc_roots,
                            rel_path=item["rel_path"],
                            start_line=item.get("start_line"),
                            end_line=item.get("end_line"),
                            context_before=12,
                            context_after=12,
                            max_chars=120000,
                        )
                    except Exception as e:
                        print(f"Failed to open result {idx}: {e}\n")
                        continue
                else:
                    parsed = _parse_open_target(spec)
                    if not parsed:
                        print("Usage: /open <n> or /open <path[:start-end]>\n")
                        continue
                    rel_path, start_line, end_line = parsed
                    try:
                        excerpt = get_note_excerpt(
                            doc_roots=cfg.doc_roots,
                            rel_path=rel_path,
                            start_line=start_line,
                            end_line=end_line,
                            context_before=12,
                            context_after=12,
                            max_chars=120000,
                        )
                    except Exception as e:
                        print(f"Failed to open note: {e}\n")
                        continue

                print("\n" + _format_open_excerpt(excerpt) + "\n")
                continue

            meta_answer = _handle_meta_query(scoped_user, cfg, manifest, retrieval, root_name=root_name)
            if meta_answer is not None:
                print("\n" + meta_answer + "\n")
                turns.append(ChatTurn(role="user", content=user, ts=time.time()))
                turns.append(ChatTurn(role="assistant", content=meta_answer, ts=time.time()))
                history_store.append("user", user)
                history_store.append("assistant", meta_answer)
                continue

            answer, last_search_results = _run_auto_query(
                scoped_user,
                retrieval,
                answer_top_k=cfg.top_k,
                max_sources_chars=cfg.max_sources_chars,
                root_name=root_name,
            )

            print("\n" + answer + "\n")

            # Persist turns
            turns.append(ChatTurn(role="user", content=user, ts=time.time()))
            turns.append(ChatTurn(role="assistant", content=answer, ts=time.time()))
            history_store.append("user", user)
            history_store.append("assistant", answer)

    finally:
        stop_event.set()
        
