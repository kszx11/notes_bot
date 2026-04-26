from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from .embedding_cache import QueryEmbeddingCache, get_query_embedding

QueryType = Literal["filename_focus", "snippet_focus", "mixed"]

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_WS_RE = re.compile(r"\s+")
_LOW_SIGNAL_QUERY_TOKENS = {
    "about",
    "did",
    "describe",
    "document",
    "documents",
    "file",
    "files",
    "find",
    "in",
    "list",
    "mention",
    "mentioned",
    "mentions",
    "my",
    "named",
    "note",
    "noted",
    "notes",
    "say",
    "said",
    "search",
    "show",
    "talk",
    "talks",
    "text",
    "the",
    "title",
    "what",
    "where",
    "which",
    "with",
}
_ARTIFACT_FILENAME_TOKENS = {
    "bak",
    "cfg",
    "config",
    "crt",
    "csr",
    "env",
    "ini",
    "json",
    "jks",
    "key",
    "log",
    "pem",
    "pfx",
    "pub",
    "template",
    "xml",
    "yaml",
    "yml",
}


@dataclass
class SearchHit:
    rel_path: str
    start_line: int | None
    end_line: int | None
    snippet: str
    heading: str | None
    first_line: str | None
    score: float
    reasons: list[str]
    filename_score: float
    text_score: float
    semantic_score: float
    keyword_score: float
    metadata_score: float
    phrase_score: float
    chunk_index: int | None = None


@dataclass
class SearchResults:
    query: str
    query_type: QueryType
    hits: list[SearchHit]


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text) if len(tok) >= 3]


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def _content_tokens(text: str) -> list[str]:
    toks = _tokens(text)
    filtered = [tok for tok in toks if tok not in _LOW_SIGNAL_QUERY_TOKENS]
    return filtered or toks


def _content_query_text(query: str) -> str:
    return " ".join(_content_tokens(query))


def _literal_query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for raw in query.split():
        cleaned = raw.strip().strip("\"'`()[]{}<>.,;:!?")
        if len(cleaned) < 4:
            continue
        if any(ch in cleaned for ch in (".", "/", "_", "-")):
            terms.append(cleaned.lower())
    return terms


def infer_query_type(query: str) -> QueryType:
    low = query.lower()
    if any(
        phrase in low
        for phrase in (
            "title",
            "filename",
            "file named",
            "note named",
            "find my note",
            "find my file",
            "find the note",
            "find the file",
            "show my note",
            "show my file",
            "show the note",
            "show the file",
            "which file",
            "what file",
        )
    ):
        return "filename_focus"
    if any(phrase in low for phrase in ("where did i", "did i mention", "snippet", "passage", "text in")):
        return "snippet_focus"
    return "mixed"


def _embed_query(client: OpenAI, model: str, text: str, cache: QueryEmbeddingCache | None = None) -> list[float]:
    return get_query_embedding(client=client, model=model, text=text, cache=cache)


def _keyword_score(query: str, doc: str) -> float:
    q = _content_tokens(query)
    d = _tokens(doc)
    if not q or not d:
        return 0.0

    d_counts: dict[str, int] = {}
    for tok in d:
        d_counts[tok] = d_counts.get(tok, 0) + 1

    overlap = 0
    for tok in q:
        if d_counts.get(tok, 0) > 0:
            overlap += 1
            d_counts[tok] -= 1

    return overlap / max(1, len(q))


def _score_filename_match(query: str, rel_path: str) -> float:
    q = _content_query_text(query).strip().lower()
    p = rel_path.lower()
    if not q:
        return 0.0
    q_tokens = set(_content_tokens(query))
    p_tokens = set(_tokens(Path(rel_path).as_posix().replace("/", " ")))
    if not q_tokens or not p_tokens:
        return 0.0

    overlap = len(q_tokens & p_tokens) / len(q_tokens)
    specificity = len(q_tokens & p_tokens) / max(1, len(p_tokens))
    stem = Path(rel_path).stem
    stem_lower = stem.lower()
    stem_tokens_list = _tokens(stem_lower)
    exact_word_match = re.search(rf"(?<![a-z0-9]){re.escape(q)}(?![a-z0-9])", stem_lower) is not None
    substring_match = q in stem_lower or q in p
    exact_stem_match = stem_lower == q
    exact_token_set_match = set(stem_tokens_list) == q_tokens
    extra_tokens = max(0, len(stem_tokens_list) - len(q_tokens))
    compactness = max(0.0, 1.0 - min(extra_tokens, 6) / 6.0)
    length_bonus = max(0.0, 1.0 - min(len(stem_lower), 80) / 80.0)
    stem_tokens = set(stem_tokens_list)
    note_like_bonus = 0.05 if " " in stem_lower else 0.0
    artifact_penalty = 0.0
    if "." in stem_lower:
        artifact_penalty += 0.06
    if stem_tokens & _ARTIFACT_FILENAME_TOKENS:
        artifact_penalty += 0.08
    all_query_tokens_present = len(q_tokens & stem_tokens) == len(q_tokens)

    if exact_stem_match:
        return max(0.0, min(1.0, 0.995 - artifact_penalty))

    if exact_word_match:
        return max(
            0.0,
            min(
                0.99,
                0.74
                + (0.08 * compactness)
                + (0.08 * length_bonus)
                + note_like_bonus
                + (0.03 if exact_token_set_match else 0.0)
                + (0.02 if all_query_tokens_present else 0.0)
                - artifact_penalty,
            ),
        )

    if substring_match:
        return max(
            0.0,
            min(
                0.90,
                0.68
                + (0.10 * overlap)
                + (0.10 * specificity)
                + (0.08 * length_bonus)
                + (0.04 * compactness)
                + note_like_bonus
                - artifact_penalty,
            ),
        )

    return max(
        0.0,
        min(
            0.85,
            (0.58 * overlap)
            + (0.24 * specificity)
            + (0.18 * compactness)
            + note_like_bonus
            - artifact_penalty,
        ),
    )


def _score_text_match(query: str, text: str) -> float:
    q = _content_query_text(query).strip().lower()
    body = text.lower()
    if not q or not body:
        return 0.0
    if q in body:
        return 1.0

    q_tokens = set(_content_tokens(query))
    if not q_tokens:
        return 0.0

    body_tokens = set(_tokens(text))
    overlap = len(q_tokens & body_tokens) / len(q_tokens)
    return overlap


def _score_phrase_match(query: str, text: str) -> float:
    normalized_query = _normalize_text(_content_query_text(query))
    normalized_text = _normalize_text(text)
    if not normalized_query or not normalized_text:
        return 0.0
    if normalized_query in normalized_text:
        return 1.0

    toks = _content_tokens(query)
    if len(toks) < 2:
        return 0.0

    for size, score in ((4, 0.9), (3, 0.78), (2, 0.62)):
        if len(toks) < size:
            continue
        for idx in range(len(toks) - size + 1):
            phrase = " ".join(toks[idx : idx + size])
            if phrase in normalized_text:
                return score
    return 0.0


def _score_ordered_token_match(query: str, text: str) -> float:
    toks = _content_tokens(query)
    normalized_text = _normalize_text(text)
    if not toks or not normalized_text:
        return 0.0
    pos = 0
    matched = 0
    for tok in toks:
        idx = normalized_text.find(tok, pos)
        if idx == -1:
            break
        matched += 1
        pos = idx + len(tok)
    if matched == len(toks):
        return 1.0
    if matched == 0:
        return 0.0
    return matched / len(toks)


def _score_literal_term_match(query: str, text: str) -> float:
    terms = _literal_query_terms(query)
    normalized = text.lower()
    if not terms or not normalized:
        return 0.0
    matched = 0
    for term in terms:
        if term in normalized:
            matched += 1
    if matched == 0:
        return 0.0
    return matched / len(terms)


def _score_metadata_match(query: str, meta: dict[str, Any]) -> float:
    parts = [
        str(meta.get("basename", "") or ""),
        str(meta.get("stem", "") or ""),
        str(meta.get("heading", "") or ""),
        str(meta.get("first_line", "") or ""),
    ]
    haystack = " ".join(part for part in parts if part).strip()
    if not haystack:
        return 0.0
    return max(
        _score_phrase_match(query, haystack),
        _score_ordered_token_match(query, haystack),
        _score_text_match(query, haystack),
        _keyword_score(query, haystack),
    )


def _semantic_rank_score(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - ((rank - 1) / (total - 1))


def _weights_for(query_type: QueryType) -> dict[str, float]:
    if query_type == "filename_focus":
        return {"semantic": 0.15, "keyword": 0.10, "filename": 0.40, "text": 0.10, "metadata": 0.15, "phrase": 0.10}
    if query_type == "snippet_focus":
        return {"semantic": 0.40, "keyword": 0.20, "filename": 0.07, "text": 0.14, "metadata": 0.09, "phrase": 0.10}
    return {"semantic": 0.32, "keyword": 0.20, "filename": 0.18, "text": 0.12, "metadata": 0.10, "phrase": 0.08}


def _combine_scores(
    *,
    query_type: QueryType,
    filename_score: float,
    text_score: float,
    semantic_score: float,
    keyword_score: float,
    metadata_score: float,
    phrase_score: float,
) -> tuple[float, list[str]]:
    weights = _weights_for(query_type)
    score = (
        weights["semantic"] * semantic_score
        + weights["keyword"] * keyword_score
        + weights["filename"] * filename_score
        + weights["text"] * text_score
        + weights["metadata"] * metadata_score
        + weights["phrase"] * phrase_score
    )

    reasons: list[str] = []
    if filename_score >= 0.45:
        reasons.append("filename")
    if phrase_score >= 0.6:
        reasons.append("phrase")
    if metadata_score >= 0.45:
        reasons.append("heading")
    if text_score >= 0.45:
        reasons.append("text")
    if keyword_score >= 0.35:
        reasons.append("keyword")
    if semantic_score >= 0.45:
        reasons.append("semantic")
    if not reasons:
        reasons.append("semantic")

    return score, reasons


def _trim_snippet(text: str, max_chars: int = 220) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _line_relevance(query: str, line: str) -> float:
    if not line.strip():
        return 0.0
    return max(
        _score_phrase_match(query, line),
        _score_text_match(query, line),
        _keyword_score(query, line),
    )


def _best_snippet_for_chunk(
    query: str,
    text: str,
    start_line: int | None,
    end_line: int | None,
    max_chars: int = 220,
) -> tuple[str, int | None, int | None]:
    lines = text.splitlines()
    if not lines:
        return _trim_snippet(text, max_chars=max_chars), start_line, end_line

    best_idx = 0
    best_score = -1.0
    for idx, line in enumerate(lines):
        score = _line_relevance(query, line)
        if score > best_score:
            best_score = score
            best_idx = idx

    window_start = max(0, best_idx - 1)
    window_end = min(len(lines), best_idx + 2)
    snippet_lines = lines[window_start:window_end]
    snippet = _trim_snippet("\n".join(snippet_lines), max_chars=max_chars)

    if start_line is None:
        return snippet, start_line, end_line

    snippet_start = start_line + window_start
    snippet_end = min(
        start_line + window_end - 1,
        end_line if end_line is not None else start_line + window_end - 1,
    )
    return snippet, snippet_start, snippet_end


def _preview_for_file(abs_path: Path) -> tuple[str, int | None, int | None]:
    try:
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", None, None

    for idx, line in enumerate(lines, start=1):
        text = line.strip()
        if text:
            return text, idx, idx
    return "", None, None


def _read_file_lines(abs_path: Path) -> list[str]:
    try:
        return abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _best_snippet_for_lines(
    query: str,
    lines: list[str],
    max_chars: int = 220,
) -> tuple[str, int | None, int | None, float, float, float, float, float, float]:
    if not lines:
        return "", None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    best_idx = 0
    best_score = -1.0
    best_phrase = 0.0
    best_text = 0.0
    best_keyword = 0.0
    best_line_quality = 0.0
    best_ordered = 0.0
    best_literal = 0.0
    best_tiebreak = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

    for idx, line in enumerate(lines):
        phrase = _score_phrase_match(query, line)
        text_score = _score_text_match(query, line)
        keyword = _keyword_score(query, line)
        literal = _score_literal_term_match(query, line)
        ordered = _score_ordered_token_match(query, line)
        score = max(phrase, text_score, keyword, ordered, literal)
        normalized_line = _normalize_text(line)
        token_count = len(_tokens(line))
        compactness = max(0.0, 1.0 - min(len(normalized_line), 180) / 180.0)
        heading_like = 0.0
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("-") or stripped.endswith(":"):
            heading_like = 0.18
        elif token_count <= 6:
            heading_like = 0.10
        line_quality = min(0.30, heading_like + (0.12 * compactness))
        tiebreak = (literal, phrase, ordered, text_score, keyword, line_quality)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_phrase = phrase
            best_text = text_score
            best_keyword = keyword
            best_line_quality = line_quality
            best_ordered = ordered
            best_literal = literal
            best_tiebreak = tiebreak
        elif score == best_score and tiebreak > best_tiebreak:
            best_idx = idx
            best_phrase = phrase
            best_text = text_score
            best_keyword = keyword
            best_line_quality = line_quality
            best_ordered = ordered
            best_literal = literal
            best_tiebreak = tiebreak

    window_start = max(0, best_idx - 1)
    window_end = min(len(lines), best_idx + 2)
    snippet = _trim_snippet("\n".join(lines[window_start:window_end]), max_chars=max_chars)
    return snippet, window_start + 1, window_end, best_phrase, best_text, best_keyword, best_line_quality, best_ordered, best_literal


def _text_candidate_score(
    *,
    query_type: QueryType,
    filename_score: float,
    metadata_score: float,
    phrase_score: float,
    text_score: float,
    keyword_score: float,
    line_quality: float,
    ordered_score: float,
    literal_score: float,
) -> float:
    if literal_score >= 0.99 and ordered_score >= 0.99:
        return min(
            0.995,
            0.84
            + (0.05 * literal_score)
            + (0.04 * ordered_score)
            + (0.05 * line_quality)
            + (0.03 * filename_score)
            + (0.03 * metadata_score),
        )
    if literal_score >= 0.99 and (text_score >= 0.95 or keyword_score >= 0.95):
        return min(
            0.985,
            0.80
            + (0.05 * literal_score)
            + (0.04 * text_score)
            + (0.03 * keyword_score)
            + (0.05 * line_quality)
            + (0.02 * filename_score),
        )
    if phrase_score >= 0.99 and ordered_score >= 0.99 and line_quality >= 0.18:
        return min(
            0.992,
            0.82
            + (0.05 * phrase_score)
            + (0.04 * ordered_score)
            + (0.06 * line_quality)
            + (0.03 * filename_score)
            + (0.03 * metadata_score),
        )
    if ordered_score >= 0.99 and line_quality >= 0.18:
        return min(
            0.975,
            0.79
            + (0.05 * ordered_score)
            + (0.06 * line_quality)
            + (0.03 * metadata_score)
            + (0.02 * filename_score),
        )
    if ordered_score >= 0.99:
        return min(
            0.965,
            0.77
            + (0.05 * ordered_score)
            + (0.04 * line_quality)
            + (0.03 * metadata_score),
        )
    if phrase_score >= 0.99:
        return min(
            0.96,
            0.76
            + (0.05 * phrase_score)
            + (0.04 * line_quality)
            + (0.03 * metadata_score)
            + (0.02 * filename_score),
        )
    if text_score >= 0.99 and keyword_score >= 0.99:
        return min(
            0.93,
            0.72
            + (0.04 * text_score)
            + (0.04 * keyword_score)
            + (0.06 * line_quality)
            + (0.03 * metadata_score)
            + (0.02 * filename_score),
        )
    if phrase_score >= 0.62 and text_score >= 0.80:
        return min(0.985, (0.93 if query_type == "snippet_focus" else 0.89) + line_quality)

    if query_type == "snippet_focus":
        return min(
            0.98,
            (0.58 * phrase_score)
            + (0.24 * text_score)
            + (0.12 * keyword_score)
            + (0.06 * metadata_score)
            + (0.08 * filename_score)
            + line_quality
            + (0.04 * ordered_score),
        )
    return min(
        0.90,
        (0.48 * phrase_score)
        + (0.22 * text_score)
        + (0.10 * keyword_score)
        + (0.10 * metadata_score)
        + (0.10 * filename_score)
        + line_quality
        + (0.06 * ordered_score)
        + (0.04 * literal_score),
    )


def _collapse_hits_by_file(hits: list[SearchHit]) -> list[SearchHit]:
    best_by_file: dict[str, SearchHit] = {}
    for hit in hits:
        prev = best_by_file.get(hit.rel_path)
        if prev is None or hit.score > prev.score:
            best_by_file[hit.rel_path] = hit
    return sorted(best_by_file.values(), key=lambda h: (-h.score, h.rel_path))


def _limit_hits_per_file(hits: list[SearchHit], per_file_limit: int) -> list[SearchHit]:
    counts: dict[str, int] = {}
    kept: list[SearchHit] = []
    for hit in hits:
        current = counts.get(hit.rel_path, 0)
        if current >= per_file_limit:
            continue
        kept.append(hit)
        counts[hit.rel_path] = current + 1
    return kept


def _serialize_key(meta: dict[str, Any]) -> tuple[str, int | None, int | None, int | None]:
    return (
        str(meta.get("rel_path", "")),
        meta.get("start_line"),
        meta.get("end_line"),
        meta.get("chunk_index"),
    )


def search_notes(
    *,
    query: str,
    client: OpenAI,
    cfg: Any,
    manifest: Any,
    store: Any,
    limit: int = 8,
) -> SearchResults:
    query_type = infer_query_type(query)
    cache = QueryEmbeddingCache(cfg.data_dir / "query_embedding_cache.sqlite")
    qemb = _embed_query(client, cfg.embedding_model, query, cache=cache)
    initial_k = max(limit * 5, cfg.top_k * 3, 15)
    raw = store.query(qemb, top_k=initial_k)

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    total = len(docs)

    hits: list[SearchHit] = []
    seen_keys: set[tuple[str, int | None, int | None, int | None]] = set()

    for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
        rel_path = str(meta.get("rel_path", ""))
        filename_score = _score_filename_match(query, rel_path)
        text_score = _score_text_match(query, doc)
        semantic_score = _semantic_rank_score(rank, total)
        keyword_score = _keyword_score(query, doc)
        metadata_score = _score_metadata_match(query, meta)
        phrase_score = _score_phrase_match(query, doc)
        snippet, snippet_start_line, snippet_end_line = _best_snippet_for_chunk(
            query,
            doc,
            meta.get("start_line"),
            meta.get("end_line"),
        )
        score, reasons = _combine_scores(
            query_type=query_type,
            filename_score=filename_score,
            text_score=text_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            metadata_score=metadata_score,
            phrase_score=phrase_score,
        )
        hit = SearchHit(
            rel_path=rel_path,
            start_line=snippet_start_line,
            end_line=snippet_end_line,
            snippet=snippet,
            heading=meta.get("heading"),
            first_line=meta.get("first_line"),
            score=score,
            reasons=reasons,
            filename_score=filename_score,
            text_score=text_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            metadata_score=metadata_score,
            phrase_score=phrase_score,
            chunk_index=meta.get("chunk_index"),
        )
        hits.append(hit)
        seen_keys.add(_serialize_key(meta))

    # Add filename-only candidates so title/path searches do not depend on vector recall.
    for rel_path in sorted(manifest.all_paths()):
        filename_score = _score_filename_match(query, rel_path)
        if filename_score < 0.55:
            continue
        abs_path = cfg.doc_root / rel_path
        snippet, start_line, end_line = _preview_for_file(abs_path)
        if query_type == "filename_focus":
            candidate_score = 0.90 + (0.09 * filename_score)
        elif query_type == "mixed":
            candidate_score = 0.72 + (0.18 * filename_score)
        else:
            candidate_score = 0.62 + (0.15 * filename_score)
        hit = SearchHit(
            rel_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            snippet=_trim_snippet(snippet or rel_path),
            heading=None,
            first_line=snippet or None,
            score=candidate_score,
            reasons=["filename"],
            filename_score=filename_score,
            text_score=0.0,
            semantic_score=0.0,
            keyword_score=0.0,
            metadata_score=0.0,
            phrase_score=0.0,
            chunk_index=None,
        )
        key = (hit.rel_path, hit.start_line, hit.end_line, hit.chunk_index)
        if key in seen_keys:
            continue
        hits.append(hit)
        seen_keys.add(key)

    # Add exact-text candidates so snippet queries do not depend entirely on vector recall.
    if query_type == "snippet_focus":
        for rel_path in sorted(manifest.all_paths()):
            abs_path = cfg.doc_root / rel_path
            lines = _read_file_lines(abs_path)
            if not lines:
                continue

            snippet, start_line, end_line, phrase_score, text_score, keyword_score, line_quality, ordered_score, literal_score = _best_snippet_for_lines(query, lines)
            if phrase_score < 0.62 and not (text_score >= 0.55 and keyword_score >= 0.50):
                continue

            filename_score = _score_filename_match(query, rel_path)
            metadata = {
                "basename": Path(rel_path).name,
                "stem": Path(rel_path).stem,
                "heading": None,
                "first_line": lines[0] if lines else "",
            }
            metadata_score = _score_metadata_match(query, metadata)
            score = _text_candidate_score(
                query_type=query_type,
                filename_score=filename_score,
                metadata_score=metadata_score,
                phrase_score=phrase_score,
                text_score=text_score,
                keyword_score=keyword_score,
                line_quality=line_quality,
                ordered_score=ordered_score,
                literal_score=literal_score,
            )
            hit = SearchHit(
                rel_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                snippet=snippet,
                heading=None,
                first_line=lines[0] if lines else None,
                score=score,
                reasons=["phrase" if phrase_score >= 0.62 else "text", "keyword"],
                filename_score=filename_score,
                text_score=text_score,
                semantic_score=0.0,
                keyword_score=keyword_score,
                metadata_score=metadata_score,
                phrase_score=phrase_score,
                chunk_index=None,
            )
            key = (hit.rel_path, hit.start_line, hit.end_line, hit.chunk_index)
            if key in seen_keys:
                continue
            hits.append(hit)
            seen_keys.add(key)

    hits.sort(key=lambda h: (-h.score, h.rel_path, h.start_line or math.inf))

    hits = _collapse_hits_by_file(hits)

    return SearchResults(query=query, query_type=query_type, hits=hits[:limit])
