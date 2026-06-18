from __future__ import annotations

from collections import Counter
import re
import textwrap
from typing import Callable

from openai import OpenAI

from .answering import format_grounded_answer, synthesize_answer
from .doc_roots import DocRoot
from .manifest import Manifest
from .note_view import get_note_excerpt
from .prompt import build_sources_block
from .store import VectorStore


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_FRESHNESS_MODES = {"prefer_fresh", "indexed_only", "include_stale"}


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if tok.strip()]


def _keyword_overlap(query: str, doc: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    d = Counter(_tokens(doc))
    if not d:
        return 0.0
    total = 0
    for tok, freq in Counter(q).items():
        total += min(freq, d.get(tok, 0))
    return total / max(1, len(q))


def _indent_block(text: str, prefix: str = "  ") -> str:
    lines = text.splitlines() or [""]
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in lines)


def _wrap_excerpt(text: str, *, width: int = 92, max_lines: int = 4) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return ""
    wrapped = textwrap.wrap(
        compact,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1].rstrip(". ") + " ..."
    return "\n".join(wrapped)


def format_passage_results(query: str, results: dict, max_items: int = 8) -> str:
    if "results" in results:
        items = results.get("results", [])
        summary = results.get("retrieval", {})
        if not items:
            return f"No indexed passages matched '{query}'."
        lines = [f"Top passages for '{query}':"]
        if summary:
            lines.append(
                f"Confidence: {summary.get('confidence', 'unknown')} "
                f"({summary.get('reason', 'no rationale')})"
            )
            indexed_results = summary.get("indexed_results")
            stale_results = summary.get("stale_results")
            if indexed_results is not None and stale_results is not None:
                lines.append(
                    f"Freshness: indexed={indexed_results} stale={stale_results}"
                )
        for item in items[:max_items]:
            section = item.get("section_path") or item.get("title") or item.get("content_type") or ""
            lines.append("")
            lines.append(f"{item['rank']}. {item['rel_path']}:{item['start_line']}-{item['end_line']}")
            detail_bits = []
            if section:
                detail_bits.append(f"Section: {section}")
            status = item.get("status", "")
            if status and status != "indexed":
                detail_bits.append(f"Status: {status}")
            detail_bits.append(f"Keyword overlap: {item.get('keyword_overlap', 0.0):.2f}")
            lines.extend(f"  {bit}" for bit in detail_bits)
            excerpt = _wrap_excerpt(item.get("text", ""))
            if excerpt:
                lines.append("  Preview:")
                lines.append(_indent_block(excerpt, prefix="    "))
        return "\n".join(lines)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs or not metas:
        return f"No indexed passages matched '{query}'."

    lines = [f"Top passages for '{query}':"]
    for idx, (doc, meta) in enumerate(zip(docs[:max_items], metas[:max_items]), start=1):
        section = meta.get("section_path") or meta.get("title") or meta.get("content_type") or ""
        lines.append("")
        lines.append(f"{idx}. {meta['rel_path']}:{meta['start_line']}-{meta['end_line']}")
        if section:
            lines.append(f"  Section: {section}")
        excerpt = _wrap_excerpt(doc)
        if excerpt:
            lines.append("  Preview:")
            lines.append(_indent_block(excerpt, prefix="    "))
    return "\n".join(lines)


def _format_low_confidence_answer(question: str, results: dict, reason: str) -> str:
    passages = format_passage_results(question, results, max_items=4)
    return (
        "I can't answer that confidently from your notes.\n\n"
        f"Reason: {reason}\n\n"
        f"{passages}"
    )


def _select_evidence(
    question: str,
    answer_text: str,
    used_sources: list[dict],
    source_texts: list[str],
    *,
    max_items: int = 4,
) -> tuple[list[dict], list[str]]:
    scored: list[tuple[float, dict, str]] = []
    for meta, text in zip(used_sources, source_texts):
        question_overlap = _keyword_overlap(question, text)
        answer_overlap = _keyword_overlap(answer_text, text)
        freshness_bonus = 0.05 if meta.get("status") == "indexed" else 0.0
        score = question_overlap + (0.5 * answer_overlap) + freshness_bonus
        if score <= 0:
            continue
        scored.append((score, meta, text))

    if not scored:
        return used_sources[:max_items], source_texts[:max_items]

    fresh = [item for item in scored if item[1].get("status") == "indexed"]
    stale = [item for item in scored if item[1].get("status") == "stale"]
    other = [item for item in scored if item[1].get("status") not in ("indexed", "stale")]

    fresh.sort(key=lambda item: item[0], reverse=True)
    stale.sort(key=lambda item: item[0], reverse=True)
    other.sort(key=lambda item: item[0], reverse=True)

    picked: list[tuple[float, dict, str]]
    if len(fresh) >= 2:
        picked = fresh[:max_items]
    elif fresh and fresh[0][0] >= 0.60:
        picked = fresh[:max_items]
    else:
        picked = fresh[:max_items]
        remaining = max(0, max_items - len(picked))
        if remaining:
            picked.extend(stale[:remaining])
        remaining = max(0, max_items - len(picked))
        if remaining:
            picked.extend(other[:remaining])

    if not picked:
        scored.sort(key=lambda item: item[0], reverse=True)
        picked = scored[:max_items]

    return [item[1] for item in picked], [item[2] for item in picked]


class RetrievalService:
    def __init__(
        self,
        *,
        store: VectorStore,
        manifest: Manifest,
        client: OpenAI,
        embedding_model: str,
        chat_model: str,
        doc_roots=None,
        answer_context_before_lines: int = 0,
        answer_context_after_lines: int = 0,
        adjacent_chunk_window: int = 1,
        embed_query_fn: Callable[[str], list[float]] | None = None,
        synthesize_answer_fn: Callable[[str, str], str] | None = None,
    ):
        self.store = store
        self.manifest = manifest
        self.client = client
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.doc_roots = tuple(doc_roots or ())
        self.answer_context_before_lines = max(0, int(answer_context_before_lines))
        self.answer_context_after_lines = max(0, int(answer_context_after_lines))
        self.adjacent_chunk_window = max(0, int(adjacent_chunk_window))
        self._embed_query_fn = embed_query_fn or self._default_embed_query
        self._synthesize_answer_fn = synthesize_answer_fn or self._default_synthesize_answer

    def _resolve_root_scope(self, root_name: str | None) -> tuple[str | None, str | None]:
        if not root_name:
            return None, None
        target = root_name.strip().lower()
        if not target:
            return None, None
        roots: dict[str, DocRoot] = {root.name: root for root in self.doc_roots}
        if target not in roots:
            raise ValueError(f"Unknown doc root: {root_name}")
        if len(self.doc_roots) <= 1:
            return target, None
        return target, f"{target}/"

    @staticmethod
    def _normalize_freshness_mode(freshness_mode: str) -> str:
        mode = (freshness_mode or "prefer_fresh").strip().lower()
        if mode not in _FRESHNESS_MODES:
            return "prefer_fresh"
        return mode

    def _default_embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=text)
        return resp.data[0].embedding

    def _default_synthesize_answer(self, question: str, sources_text: str) -> str:
        return synthesize_answer(
            client=self.client,
            model=self.chat_model,
            question=question,
            sources_text=sources_text,
        )

    def search_files(
        self,
        query: str,
        mode: str = "both",
        limit: int = 100,
        freshness_mode: str = "prefer_fresh",
        root_name: str | None = None,
    ) -> list[dict]:
        freshness_mode = self._normalize_freshness_mode(freshness_mode)
        _, rel_path_prefix = self._resolve_root_scope(root_name)
        items = self.store.search_files(
            query,
            mode=mode,
            limit=max(limit * 2, limit),
            rel_path_prefix=rel_path_prefix,
        )
        for item in items:
            st = self.manifest.get(item["rel_path"])
            item["status"] = st.status if st is not None else "unknown"
        if freshness_mode == "indexed_only":
            items = [item for item in items if item.get("status") == "indexed"]
        elif freshness_mode == "prefer_fresh":
            items.sort(key=lambda item: (0 if item.get("status") == "indexed" else 1, item["rel_path"]))
        return items[:limit]

    def search_passages(
        self,
        query: str,
        top_k: int = 8,
        freshness_mode: str = "prefer_fresh",
        root_name: str | None = None,
    ) -> dict:
        freshness_mode = self._normalize_freshness_mode(freshness_mode)
        _, rel_path_prefix = self._resolve_root_scope(root_name)
        try:
            qemb = self._embed_query_fn(query)
        except Exception:
            qemb = None
        raw = self.store.query(
            query_text=query,
            query_embedding=qemb,
            top_k=max(top_k * 3, top_k),
            rel_path_prefix=rel_path_prefix,
        )
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        ids = raw.get("ids", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        ranked: list[tuple[float, str, dict, str, float | None]] = []
        for doc, meta, row_id, dist in zip(docs, metas, ids, dists):
            st = self.manifest.get(meta.get("rel_path", ""))
            status = st.status if st is not None else "unknown"
            if freshness_mode == "indexed_only" and status != "indexed":
                continue
            if freshness_mode == "prefer_fresh":
                freshness_penalty = 0.0 if status == "indexed" else 0.2 if status == "stale" else 1.0
            else:
                freshness_penalty = 0.0
            overlap = _keyword_overlap(query, doc)
            distance = float(dist if dist is not None else 1.0)
            score = overlap - distance - freshness_penalty
            if len(_tokens(query)) > 1 and query.strip().lower() in doc.lower():
                score += 0.10
            meta = {**meta, "status": status}
            ranked.append((score, doc, meta, row_id, dist))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, str, dict, str, float | None]] = []
        selected_chunks: dict[str, list[int]] = {}
        for item in ranked:
            meta = item[2]
            rel_path = str(meta.get("rel_path", ""))
            chunk_index = int(meta.get("chunk_index", -10_000))
            prior = selected_chunks.setdefault(rel_path, [])
            if self.adjacent_chunk_window > 0 and any(
                abs(chunk_index - existing) <= self.adjacent_chunk_window for existing in prior
            ):
                continue
            prior.append(chunk_index)
            selected.append(item)
            if len(selected) >= top_k:
                break
        ranked = selected
        return {
            "ids": [[item[3] for item in ranked]],
            "documents": [[item[1] for item in ranked]],
            "metadatas": [[item[2] for item in ranked]],
            "distances": [[item[4] for item in ranked]],
        }

    def retrieve_passages(
        self,
        query: str,
        top_k: int,
        include_text: bool,
        max_chars: int,
        freshness_mode: str = "prefer_fresh",
        root_name: str | None = None,
    ) -> dict:
        freshness_mode = self._normalize_freshness_mode(freshness_mode)
        scoped_root_name, _ = self._resolve_root_scope(root_name)
        results = self.search_passages(
            query,
            top_k=top_k,
            freshness_mode=freshness_mode,
            root_name=scoped_root_name,
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        confidence, reason = self._confidence(query, results)

        items: list[dict] = []
        used = 0
        for i, (doc, meta, dist, row_id) in enumerate(zip(docs, metas, dists, ids), start=1):
            expanded_meta, expanded_doc = self._expand_adjacent_chunks(meta, doc)
            snippet = expanded_doc if include_text else ""
            if include_text:
                if used + len(snippet) > max_chars:
                    snippet = snippet[: max(0, max_chars - used)]
                used += len(snippet)

            items.append(
                {
                    "rank": i,
                    "id": row_id,
                    "distance": dist,
                    "rel_path": expanded_meta.get("rel_path"),
                    "start_line": expanded_meta.get("start_line"),
                    "end_line": expanded_meta.get("end_line"),
                    "chunk_index": expanded_meta.get("chunk_index"),
                    "mtime": expanded_meta.get("mtime"),
                    "content_type": expanded_meta.get("content_type"),
                    "title": expanded_meta.get("title"),
                    "section_path": expanded_meta.get("section_path"),
                    "status": expanded_meta.get("status", "unknown"),
                    "keyword_overlap": _keyword_overlap(query, expanded_doc),
                    "focus_start_line": expanded_meta.get("focus_start_line"),
                    "focus_end_line": expanded_meta.get("focus_end_line"),
                    "window_chunk_start": expanded_meta.get("window_chunk_start"),
                    "window_chunk_end": expanded_meta.get("window_chunk_end"),
                    "text": snippet,
                }
            )
            if include_text and used >= max_chars:
                break

        return {
            "query": query,
            "top_k": top_k,
            "results": items,
            "retrieval": {
                "root_name": scoped_root_name,
                "freshness_mode": freshness_mode,
                "confidence": confidence,
                "reason": reason,
                "distinct_files": len({item["rel_path"] for item in items}),
                "result_count": len(items),
                "indexed_results": sum(1 for item in items if item.get("status") == "indexed"),
                "stale_results": sum(1 for item in items if item.get("status") == "stale"),
            },
        }

    def _expand_adjacent_chunks(self, meta: dict, doc: str) -> tuple[dict, str]:
        if self.adjacent_chunk_window <= 0:
            return meta, doc
        rel_path = str(meta.get("rel_path", "")).strip()
        chunk_index = meta.get("chunk_index")
        if not rel_path or chunk_index is None or not hasattr(self.store, "get_chunk_window"):
            return meta, doc

        try:
            chunks = self.store.get_chunk_window(
                rel_path,
                int(chunk_index),
                window_size=self.adjacent_chunk_window,
            )
        except Exception:
            return meta, doc

        if not chunks:
            return meta, doc

        section_path = str(meta.get("section_path", "") or "")
        title = str(meta.get("title", "") or "")
        filtered: list[dict] = []
        for chunk in chunks:
            if int(chunk.get("chunk_index", chunk_index)) == int(chunk_index):
                filtered.append(chunk)
                continue
            chunk_section = str(chunk.get("section_path", "") or "")
            chunk_title = str(chunk.get("title", "") or "")
            if section_path and chunk_section and chunk_section != section_path:
                continue
            if not section_path and title and chunk_title and chunk_title != title:
                continue
            filtered.append(chunk)

        if not filtered:
            return meta, doc

        filtered.sort(key=lambda item: int(item.get("chunk_index", 0)))
        combined_text = "\n\n".join(
            str(chunk.get("document", "")).strip()
            for chunk in filtered
            if str(chunk.get("document", "")).strip()
        ).strip()
        if not combined_text:
            return meta, doc

        return (
            {
                **meta,
                "start_line": min(int(chunk.get("start_line", meta.get("start_line", 1))) for chunk in filtered),
                "end_line": max(int(chunk.get("end_line", meta.get("end_line", 1))) for chunk in filtered),
                "focus_start_line": meta.get("start_line"),
                "focus_end_line": meta.get("end_line"),
                "window_chunk_start": min(int(chunk.get("chunk_index", chunk_index)) for chunk in filtered),
                "window_chunk_end": max(int(chunk.get("chunk_index", chunk_index)) for chunk in filtered),
            },
            combined_text,
        )

    def _confidence(self, question: str, results: dict) -> tuple[str, str]:
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        if not docs or not metas:
            return "low", "no passages were retrieved"

        distinct_files = len({meta.get("rel_path", "") for meta in metas})
        best_overlap = max((_keyword_overlap(question, doc) for doc in docs), default=0.0)
        best_distance = min((float(dist) for dist in dists if dist is not None), default=1.0)
        stale_count = sum(1 for meta in metas if meta.get("status") == "stale")

        if best_overlap >= 0.45:
            if stale_count:
                return "medium", "strong keyword support, but some evidence is stale"
            return "high", "strong keyword support"
        if best_overlap >= 0.25 and best_distance <= 0.55:
            return "medium", "mixed lexical and semantic support"
        if distinct_files >= 2 and best_distance <= 0.35:
            return "medium", "semantic support across multiple files"
        if best_overlap < 0.15 and best_distance > 0.65:
            return "low", "retrieved passages are only weakly related"
        if distinct_files == 1 and best_overlap < 0.2 and best_distance > 0.55:
            return "low", "only one weakly matching file was retrieved"
        return "medium", "partial support"

    def _load_answer_source(self, meta: dict, doc: str) -> tuple[dict, str]:
        meta, doc = self._expand_adjacent_chunks(meta, doc)
        if not self.doc_roots:
            return meta, doc
        if self.answer_context_before_lines <= 0 and self.answer_context_after_lines <= 0:
            return meta, doc

        try:
            excerpt = get_note_excerpt(
                doc_roots=self.doc_roots,
                rel_path=str(meta.get("rel_path", "")),
                start_line=meta.get("start_line"),
                end_line=meta.get("end_line"),
                context_before=self.answer_context_before_lines,
                context_after=self.answer_context_after_lines,
                max_chars=6000,
            )
        except Exception:
            return meta, doc

        text = excerpt.get("text", "") or doc
        if excerpt.get("truncated"):
            text = text.rstrip() + "\n... [truncated]"

        return (
            {
                **meta,
                "start_line": excerpt["display_start_line"],
                "end_line": excerpt["display_end_line"],
                "focus_start_line": meta.get("start_line"),
                "focus_end_line": meta.get("end_line"),
            },
            text,
        )

    def answer_question(
        self,
        question: str,
        top_k: int,
        max_sources_chars: int,
        freshness_mode: str = "prefer_fresh",
        root_name: str | None = None,
    ) -> dict:
        freshness_mode = self._normalize_freshness_mode(freshness_mode)
        scoped_root_name, _ = self._resolve_root_scope(root_name)
        results = self.search_passages(
            question,
            top_k=top_k,
            freshness_mode=freshness_mode,
            root_name=scoped_root_name,
        )
        sources_text, used_sources, source_texts = build_sources_block(
            results,
            max_chars=max_sources_chars,
            source_loader=self._load_answer_source,
        )
        confidence, reason = self._confidence(question, results)

        if confidence == "low":
            answer = _format_low_confidence_answer(question, results, reason)
            return {
                "question": question,
                "answer": answer,
                "sources": used_sources,
                "model": self.chat_model,
                "root_name": scoped_root_name,
                "freshness_mode": freshness_mode,
                "confidence": confidence,
                "confidence_reason": reason,
            }

        try:
            answer_text = self._synthesize_answer_fn(question, sources_text)
        except Exception:
            answer = _format_low_confidence_answer(question, results, "the language model was unavailable")
            return {
                "question": question,
                "answer": answer,
                "sources": used_sources,
                "model": self.chat_model,
                "root_name": scoped_root_name,
                "freshness_mode": freshness_mode,
                "confidence": "low",
                "confidence_reason": "the language model was unavailable",
            }
        evidence_sources, evidence_texts = _select_evidence(
            question,
            answer_text,
            used_sources,
            source_texts,
        )
        stale_evidence = any(meta.get("status") == "stale" for meta in evidence_sources)
        answer_note = "Note: some cited evidence comes from stale indexed content." if stale_evidence else ""
        answer = format_grounded_answer(answer_text, evidence_sources, evidence_texts, answer_note=answer_note)
        return {
            "question": question,
            "answer": answer,
            "sources": used_sources,
            "model": self.chat_model,
            "root_name": scoped_root_name,
            "freshness_mode": freshness_mode,
            "confidence": confidence,
            "confidence_reason": reason,
        }
