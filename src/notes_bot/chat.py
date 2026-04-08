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
from .prompt import SYSTEM_PROMPT, build_sources_block, allowed_citation_set
from .validate import validate_structured_answer
from .hybrid import hybrid_rerank  # NEW

HELP_TEXT = (
    "Commands:\n"
    "- /help show this help\n"
    "- /clear clear chat context/history\n"
    "- /reindex run incremental indexing now\n"
    "- /indexed list indexed files from manifest\n"
    "- /find <term> search indexed files by filename or text\n"
    "- /findname <term> search filename only\n"
    "- /findtext <term> search text content only\n"
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


def _embed_query(client: OpenAI, model: str, text: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def _list_indexed_files(manifest: Manifest) -> list[str]:
    return sorted(manifest.all_paths())


def _search_indexed_files(cfg, manifest: Manifest, term: str, mode: str) -> list[tuple[str, bool, bool]]:
    """
    Returns tuples:
      (rel_path, filename_match, text_match)
    mode: "filename" | "text" | "both"
    """
    needle = term.strip().lower()
    if not needle:
        return []

    out: list[tuple[str, bool, bool]] = []
    for rel_path in _list_indexed_files(manifest):
        filename_match = needle in rel_path.lower()
        text_match = False

        if mode in ("text", "both"):
            abs_path = cfg.doc_root / rel_path
            if abs_path.exists() and abs_path.is_file():
                text_match = needle in _read_note_text(abs_path).lower()

        if mode == "filename" and filename_match:
            out.append((rel_path, filename_match, text_match))
        elif mode == "text" and text_match:
            out.append((rel_path, filename_match, text_match))
        elif mode == "both" and (filename_match or text_match):
            out.append((rel_path, filename_match, text_match))

    return out


def _format_indexed_files(files: list[str], max_items: int = 100) -> str:
    if not files:
        return "No files are indexed yet."
    shown = files[:max_items]
    lines = [f"Indexed files: {len(files)} total"]
    lines.extend(f"- {p}" for p in shown)
    if len(files) > max_items:
        lines.append(f"... ({len(files) - max_items} more)")
    return "\n".join(lines)


def _format_search_results(term: str, mode: str, matches: list[tuple[str, bool, bool]], max_items: int = 100) -> str:
    if not matches:
        return f"No indexed files matched '{term}'."

    mode_label = {"filename": "filename", "text": "text", "both": "filename or text"}.get(mode, mode)
    shown = matches[:max_items]
    lines = [f"Matched {len(matches)} file(s) for '{term}' in {mode_label}:"]
    for rel_path, filename_match, text_match in shown:
        tags = []
        if filename_match:
            tags.append("filename")
        if text_match:
            tags.append("text")
        suffix = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"- {rel_path}{suffix}")
    if len(matches) > max_items:
        lines.append(f"... ({len(matches) - max_items} more)")
    return "\n".join(lines)


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
        return _format_indexed_files(files)

    analytic_target = _detect_analytic_target(text)
    if analytic_target:
        return _format_analytic_density_results(cfg, manifest, analytic_target)

    mention = _extract_mention_term(text)
    if mention:
        term, mode = mention
        matches = _search_indexed_files(cfg, manifest, term, mode)
        return _format_search_results(term, mode, matches)

    if low.startswith("/indexed"):
        files = _list_indexed_files(manifest)
        return _format_indexed_files(files)

    if low.startswith("/find "):
        term = text[6:].strip()
        mode = "both"
        matches = _search_indexed_files(cfg, manifest, term, mode)
        return _format_search_results(term, mode, matches)

    if low.startswith("/findname "):
        term = text[10:].strip()
        matches = _search_indexed_files(cfg, manifest, term, "filename")
        return _format_search_results(term, "filename", matches)

    if low.startswith("/findtext "):
        term = text[10:].strip()
        matches = _search_indexed_files(cfg, manifest, term, "text")
        return _format_search_results(term, "text", matches)

    return None


def _background_index_loop(stop_event: threading.Event, index_lock: threading.Lock, cfg, client, manifest, store):
    interval = max(1, int(cfg.scan_interval_minutes)) * 60
    while not stop_event.is_set():
        try:
            with index_lock:
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
                    progress_callback=_make_background_progress_callback("index-bg"),
                )
        except Exception as e:
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

    stop_event = threading.Event()
    index_lock = threading.Lock()

    t = threading.Thread(
        target=_background_index_loop,
        args=(stop_event, index_lock, cfg, client, manifest, store),
        daemon=True
    )
    t.start()

    print("Notes bot ready.")
    print("Commands: /clear, /reindex, /indexed, /find <term>, /findname <term>, /findtext <term>, /exit")
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

            if user.lower() == "/exit":
                break

            if user.lower() == "/help":
                print("\n" + HELP_TEXT)
                continue

            if user.lower() == "/clear":
                turns.clear()
                history_store.clear()
                print("Cleared chat context.\n")
                continue

            if user.lower() == "/reindex":
                print("Reindexing (incremental) ...")
                locked = index_lock.acquire(blocking=False)
                if not locked:
                    print("Indexer is busy in background; waiting for current pass to finish ...")
                    index_lock.acquire()
                try:
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
                        progress_callback=_make_progress_callback("reindex"),
                    )
                finally:
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

            # ---- Retrieval ----
            query_text = user
            qemb = _embed_query(client, cfg.embedding_model, query_text)

            # Pull more than top_k initially so hybrid rerank has room
            initial_k = max(cfg.top_k * 3, cfg.top_k)
            results = store.query(qemb, top_k=initial_k)

            # Hybrid rerank: combine embedding rank with keyword overlap
            reranked = hybrid_rerank(query_text, results, top_k=cfg.top_k)

            sources_text, used_sources = build_sources_block(reranked, max_chars=cfg.max_sources_chars)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for tr in recent_turns():
                messages.append({"role": tr.role, "content": tr.content})

            messages.append({
                "role": "user",
                "content": f"SOURCES:\n{sources_text}\n\nQUESTION:\n{user}"
            })

            allowed = allowed_citation_set(used_sources)

            def call_model(extra_instruction: str | None = None) -> str:
                local_messages = list(messages)
                if extra_instruction:
                    local_messages.append({"role": "user", "content": extra_instruction})
                resp = client.chat.completions.create(
                    model=cfg.chat_model,
                    messages=local_messages,
                    temperature=0.1,  # tighter
                )
                return resp.choices[0].message.content.strip()

            answer = call_model()

            ok, reason = validate_structured_answer(answer, allowed)
            if not ok:
                # One retry with stricter instruction
                retry_instruction = (
                    "Your last response did not follow the required format/grounding rules. "
                    f"Problem: {reason}\n\n"
                    "Retry. You MUST:\n"
                    "- Output exactly 'Answer:' then 'Evidence:' sections.\n"
                    "- Evidence bullets must be verbatim quotes from SOURCES.\n"
                    "- Every Evidence bullet must end with a citation that matches one of the provided sources exactly.\n"
                    "- If you cannot comply, output exactly: I can't find that in your notes."
                )
                answer2 = call_model(retry_instruction)
                ok2, reason2 = validate_structured_answer(answer2, allowed)
                if ok2 or answer2.strip() == "I can't find that in your notes.":
                    answer = answer2
                else:
                    # Final fallback: refuse rather than hallucinate
                    answer = "I can't find that in your notes."

            print("\n" + answer + "\n")

            # Persist turns
            turns.append(ChatTurn(role="user", content=user, ts=time.time()))
            turns.append(ChatTurn(role="assistant", content=answer, ts=time.time()))
            history_store.append("user", user)
            history_store.append("assistant", answer)

    finally:
        stop_event.set()
        
