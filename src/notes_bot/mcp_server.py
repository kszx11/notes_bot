from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import load_config
from .embedding_cache import QueryEmbeddingCache, get_query_embedding
from .eval_candidates import build_eval_candidates, load_eval_candidates, promote_eval_candidates
from .intent import route_user_input
from .hybrid import hybrid_rerank
from .indexer import run_index_once
from .manifest import Manifest
from .prompt import GENERAL_CHAT_PROMPT, SYSTEM_PROMPT, allowed_citation_set, build_sources_block
from .search import search_notes
from .search_log import append_search_log
from .store import VectorStore
from .validate import validate_structured_answer

SERVER_NAME = "notes-bot-mcp"
SERVER_VERSION = "1.2.0"


class MCPError(Exception):
    def __init__(self, message: str, code: int = -32603):
        super().__init__(message)
        self.code = code


@dataclass
class ReindexState:
    running: bool = False
    owner: str = ""
    mode: str = "incremental"
    started_at: float | None = None
    ended_at: float | None = None
    current_file: str = ""
    current_index: int = 0
    total_files: int = 0
    status: str = "idle"
    last_error: str = ""
    last_stats: dict[str, int] | None = None


class NotesMCPServer:
    def __init__(self, config_path: str | Path = "config.yaml", enable_background: bool = True):
        self.cfg = load_config(config_path)
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI()
        self.manifest = Manifest(self.cfg.manifest_path)
        self.store = VectorStore(self.cfg.index_dir, collection_name="notes")
        self.query_embedding_cache = QueryEmbeddingCache(self.cfg.data_dir / "query_embedding_cache.sqlite")
        self.search_log_path = self.cfg.data_dir / "search_queries.jsonl"
        self.eval_candidates_path = self.cfg.data_dir / "eval_candidates.json"
        self.eval_queries_path = self.cfg.data_dir / "eval_queries.json"

        self._index_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reindex_state = ReindexState()

        self._tools = self._build_tools()
        self._resources = self._build_resources()
        self._prompts = self._build_prompts()

        self._background_thread: threading.Thread | None = None
        if enable_background:
            self._background_thread = threading.Thread(target=self._background_index_loop, daemon=True)
            self._background_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()

    def _set_state(self, **kwargs: Any) -> None:
        with self._state_lock:
            for k, v in kwargs.items():
                setattr(self._reindex_state, k, v)

    def _get_state(self) -> ReindexState:
        with self._state_lock:
            st = self._reindex_state
            return ReindexState(**asdict(st))

    def _progress_callback(self, owner: str):
        def cb(event: dict) -> None:
            phase = str(event.get("phase", ""))
            if phase == "scan":
                self._set_state(
                    owner=owner,
                    current_file=str(event.get("rel_path", "")),
                    current_index=int(event.get("index", 0)),
                    total_files=int(event.get("total", 0)),
                    status=str(event.get("status", "scan")),
                )
            elif phase == "file":
                self._set_state(
                    owner=owner,
                    current_file=str(event.get("rel_path", "")),
                    current_index=int(event.get("index", 0)),
                    total_files=int(event.get("total", 0)),
                    status=str(event.get("status", "file")),
                )
            elif phase == "done":
                stats = event.get("stats")
                if stats is not None:
                    self._set_state(last_stats={
                        "scanned": int(stats.scanned),
                        "updated": int(stats.updated),
                        "deleted": int(stats.deleted),
                        "errors": int(stats.errors),
                    })
        return cb

    def _run_index_pass(self, owner: str, force_reindex: bool = False) -> dict[str, Any]:
        self._set_state(
            running=True,
            owner=owner,
            mode="full" if force_reindex else "incremental",
            started_at=time.time(),
            ended_at=None,
            status="starting",
            last_error="",
        )
        try:
            stats = run_index_once(
                client=self.client,
                doc_root=self.cfg.doc_root,
                include_ext=self.cfg.include_ext,
                manifest=self.manifest,
                store=self.store,
                embedding_model=self.cfg.embedding_model,
                chunk_chars=self.cfg.chunk_chars,
                chunk_overlap=self.cfg.chunk_overlap,
                max_file_size_mb=self.cfg.max_file_size_mb,
                max_chunks_per_file=self.cfg.max_chunks_per_file,
                force_reindex=force_reindex,
                progress_callback=self._progress_callback(owner),
            )
            payload = {
                "ok": True,
                "owner": owner,
                "mode": "full" if force_reindex else "incremental",
                "stats": {
                    "scanned": stats.scanned,
                    "updated": stats.updated,
                    "deleted": stats.deleted,
                    "errors": stats.errors,
                },
            }
            self._set_state(last_stats=payload["stats"])
            return payload
        except Exception as e:
            msg = str(e)
            self._set_state(last_error=msg, status="error")
            return {"ok": False, "owner": owner, "error": msg}
        finally:
            self._set_state(
                running=False,
                owner="",
                mode="incremental",
                ended_at=time.time(),
                status="idle",
                current_file="",
                current_index=0,
                total_files=0,
            )

    def _background_index_loop(self) -> None:
        interval = max(1, int(self.cfg.scan_interval_minutes)) * 60
        while not self._stop_event.is_set():
            if self._index_lock.acquire(timeout=1):
                try:
                    self._run_index_pass(owner="background", force_reindex=False)
                finally:
                    self._index_lock.release()

            slept = 0
            while slept < interval and not self._stop_event.is_set():
                time.sleep(1)
                slept += 1

    def _embed_query(self, text: str) -> list[float]:
        return get_query_embedding(
            client=self.client,
            model=self.cfg.embedding_model,
            text=text,
            cache=self.query_embedding_cache,
        )

    def _list_indexed_files(self) -> list[str]:
        return sorted(self.manifest.all_paths())

    def _search_indexed_files(self, term: str, mode: str, limit: int) -> list[dict[str, Any]]:
        needle = term.strip().lower()
        if not needle:
            return []

        out: list[dict[str, Any]] = []
        for rel_path in self._list_indexed_files():
            filename_match = needle in rel_path.lower()
            text_match = False

            if mode in ("text", "both"):
                abs_path = self._safe_doc_path(rel_path)
                if abs_path.exists() and abs_path.is_file():
                    try:
                        body = abs_path.read_text(encoding="utf-8", errors="replace").lower()
                        text_match = needle in body
                    except Exception:
                        text_match = False

            if mode == "filename" and filename_match:
                out.append({"rel_path": rel_path, "filename_match": True, "text_match": text_match})
            elif mode == "text" and text_match:
                out.append({"rel_path": rel_path, "filename_match": filename_match, "text_match": True})
            elif mode == "both" and (filename_match or text_match):
                out.append({"rel_path": rel_path, "filename_match": filename_match, "text_match": text_match})

            if len(out) >= limit:
                break

        return out

    def _safe_doc_path(self, rel_path: str) -> Path:
        root = self.cfg.doc_root.resolve()
        candidate = (root / rel_path).resolve()
        if os.path.commonpath([str(root), str(candidate)]) != str(root):
            raise MCPError("Invalid rel_path outside doc_root", code=-32602)
        return candidate

    def _get_note_excerpt(
        self,
        rel_path: str,
        start_line: int | None,
        end_line: int | None,
        max_chars: int,
    ) -> dict[str, Any]:
        abs_path = self._safe_doc_path(rel_path)
        if not abs_path.exists() or not abs_path.is_file():
            raise MCPError(f"File not found: {rel_path}", code=-32602)

        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        n = len(lines)
        s = max(1, int(start_line or 1))
        e = min(n, int(end_line or n))
        if e < s:
            s, e = e, s

        excerpt = "\n".join(lines[s - 1 : e])
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars]

        return {
            "rel_path": rel_path,
            "abs_path": str(abs_path),
            "start_line": s,
            "end_line": e,
            "total_lines": n,
            "text": excerpt,
            "truncated": len(excerpt) >= max_chars,
        }

    def _retrieve_notes(self, query: str, top_k: int, include_text: bool, max_chars: int) -> dict[str, Any]:
        qemb = self._embed_query(query)
        initial_k = max(top_k * 3, top_k)
        results = self.store.query(qemb, top_k=initial_k)
        reranked = hybrid_rerank(query, results, top_k=top_k)

        docs = reranked.get("documents", [[]])[0]
        metas = reranked.get("metadatas", [[]])[0]
        dists = reranked.get("distances", [[]])[0]
        ids = reranked.get("ids", [[]])[0]

        items: list[dict[str, Any]] = []
        used = 0
        for i, (doc, meta, dist, row_id) in enumerate(zip(docs, metas, dists, ids), start=1):
            snippet = doc if include_text else ""
            if include_text:
                if used + len(snippet) > max_chars:
                    snippet = snippet[: max(0, max_chars - used)]
                used += len(snippet)

            items.append(
                {
                    "rank": i,
                    "id": row_id,
                    "distance": dist,
                    "rel_path": meta.get("rel_path"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "chunk_index": meta.get("chunk_index"),
                    "mtime": meta.get("mtime"),
                    "text": snippet,
                }
            )
            if include_text and used >= max_chars:
                break

        return {"query": query, "top_k": top_k, "results": items}

    def _answer_from_notes(self, question: str, top_k: int, max_sources_chars: int) -> dict[str, Any]:
        qemb = self._embed_query(question)
        initial_k = max(top_k * 3, top_k)
        results = self.store.query(qemb, top_k=initial_k)
        reranked = hybrid_rerank(question, results, top_k=top_k)

        sources_text, used_sources = build_sources_block(reranked, max_chars=max_sources_chars)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCES:\n{sources_text}\n\nQUESTION:\n{question}"},
        ]

        allowed = allowed_citation_set(used_sources)

        def call_model(extra_instruction: str | None = None) -> str:
            local_messages = list(messages)
            if extra_instruction:
                local_messages.append({"role": "user", "content": extra_instruction})
            resp = self.client.chat.completions.create(
                model=self.cfg.chat_model,
                messages=local_messages,
                temperature=0.1,
            )
            return (resp.choices[0].message.content or "").strip()

        answer = call_model()
        ok, reason = validate_structured_answer(answer, allowed)
        if not ok:
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
            ok2, _ = validate_structured_answer(answer2, allowed)
            if ok2 or answer2.strip() == "I can't find that in your notes.":
                answer = answer2
            else:
                answer = "I can't find that in your notes."

        return {
            "question": question,
            "answer": answer,
            "sources": used_sources,
            "model": self.cfg.chat_model,
        }

    def _route_query(self, query: str) -> dict[str, Any]:
        decision = route_user_input(query)
        return {
            "query": query,
            "mode": decision.mode,
            "confidence": decision.confidence,
            "reasons": decision.reasons,
            "command_name": decision.command_name,
            "note_path_hint": decision.note_path_hint,
            "note_result_index": decision.note_result_index,
        }

    def _query_notes(self, query: str, limit: int) -> dict[str, Any]:
        results = search_notes(
            query=query,
            client=self.client,
            cfg=self.cfg,
            manifest=self.manifest,
            store=self.store,
            limit=limit,
        )
        try:
            append_search_log(self.search_log_path, query=query, results=results)
        except OSError:
            pass
        payload = []
        for hit in results.hits:
            payload.append(
                {
                    "rel_path": hit.rel_path,
                    "start_line": hit.start_line,
                    "end_line": hit.end_line,
                    "snippet": hit.snippet,
                    "heading": hit.heading,
                    "first_line": hit.first_line,
                    "score": hit.score,
                    "reasons": hit.reasons,
                    "filename_score": hit.filename_score,
                    "text_score": hit.text_score,
                    "semantic_score": hit.semantic_score,
                    "keyword_score": hit.keyword_score,
                    "metadata_score": hit.metadata_score,
                    "chunk_index": hit.chunk_index,
                }
            )
        return {
            "query": query,
            "query_type": results.query_type,
            "count": len(payload),
            "results": payload,
        }

    def _eval_candidates(self, limit: int, refresh: bool) -> dict[str, Any]:
        if refresh:
            candidates = build_eval_candidates(
                log_path=self.search_log_path,
                existing_eval_path=self.eval_queries_path,
                limit=limit,
            )
            self.eval_candidates_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval_candidates_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        else:
            candidates = load_eval_candidates(self.eval_candidates_path)
            if limit > 0:
                candidates = candidates[:limit]
        return {
            "count": len(candidates),
            "results": candidates,
            "candidate_path": str(self.eval_candidates_path),
            "eval_path": str(self.eval_queries_path),
        }

    def _eval_promote(self, selections: list[int] | None, promote_all: bool) -> dict[str, Any]:
        promoted, remaining = promote_eval_candidates(
            candidate_path=self.eval_candidates_path,
            eval_path=self.eval_queries_path,
            selections=None if promote_all else selections,
        )
        return {
            "promoted": promoted,
            "remaining": remaining,
            "candidate_path": str(self.eval_candidates_path),
            "eval_path": str(self.eval_queries_path),
        }

    def _general_chat(self, query: str) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=[
                {"role": "system", "content": GENERAL_CHAT_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return {
            "query": query,
            "answer": answer,
            "model": self.cfg.chat_model,
        }

    def _build_tools(self) -> dict[str, dict[str, Any]]:
        return {
            "route_query": {
                "description": "Classify a natural-language query as command, note_open, notes_search, or general_chat.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            "query_notes": {
                "description": "Search indexed notes and return ranked file and snippet matches.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            "eval_candidates": {
                "description": "Build or read draft eval cases from logged natural-language note searches.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 25},
                        "refresh": {"type": "boolean", "default": True},
                    },
                    "additionalProperties": False,
                },
            },
            "eval_promote": {
                "description": "Promote reviewed eval candidates into the main eval set.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selections": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        },
                        "promote_all": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
            },
            "chat": {
                "description": "Handle a general world-knowledge prompt without retrieving notes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            "list_indexed_files": {
                "description": "List indexed files from the manifest.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 200},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                    "additionalProperties": False,
                },
            },
            "find_files": {
                "description": "Legacy compatibility tool. Find indexed files by term in filename and/or text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "mode": {"type": "string", "enum": ["filename", "text", "both"], "default": "both"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    },
                    "required": ["term"],
                    "additionalProperties": False,
                },
            },
            "search_notes": {
                "description": "Legacy compatibility tool. Low-level vector + hybrid keyword retrieval against indexed note chunks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                        "include_text": {"type": "boolean", "default": True},
                        "max_chars": {"type": "integer", "minimum": 500, "maximum": 200000, "default": 35000},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            "get_note": {
                "description": "Read a note or line-range excerpt by rel_path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "rel_path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": 500000, "default": 120000},
                    },
                    "required": ["rel_path"],
                    "additionalProperties": False,
                },
            },
            "reindex_status": {
                "description": "Get current and last reindex status.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            "reindex_now": {
                "description": "Run reindex immediately. Incremental by default; pass force_reindex=true to rebuild everything.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "wait": {"type": "boolean", "default": False},
                        "force_reindex": {"type": "boolean", "default": False},
                        "lock_timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 86400, "default": 300},
                    },
                    "additionalProperties": False,
                },
            },
            "answer_from_notes": {
                "description": "Legacy compatibility tool. Generate grounded answer from indexed notes using citations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
                        "max_sources_chars": {"type": "integer", "minimum": 500, "maximum": 200000, "default": 35000},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            "list_large_files": {
                "description": "List indexed files that currently exceed max_file_size_mb or a threshold.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "min_size_mb": {"type": "number", "minimum": 0.1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    },
                    "additionalProperties": False,
                },
            },
        }

    def _build_resources(self) -> dict[str, dict[str, Any]]:
        return {
            "notes://manifest": {
                "name": "Indexed Manifest",
                "description": "All indexed files from SQLite manifest.",
                "mimeType": "application/json",
            },
            "notes://config": {
                "name": "Effective Config",
                "description": "Loaded notes_bot configuration values.",
                "mimeType": "application/json",
            },
            "notes://reindex/status": {
                "name": "Reindex Status",
                "description": "Current/last reindex state.",
                "mimeType": "application/json",
            },
        }

    def _build_prompts(self) -> dict[str, dict[str, Any]]:
        return {
            "grounded_note_answer": {
                "description": "Prompt template for grounded answers from notes.",
                "arguments": [
                    {"name": "question", "required": True, "description": "User question to answer from notes."},
                    {"name": "top_k", "required": False, "description": "How many chunks to retrieve."},
                ],
            }
        }

    def _resource_value(self, uri: str) -> dict[str, Any]:
        if uri == "notes://manifest":
            items = []
            for st in sorted(self.manifest.iter_all(), key=lambda x: x.rel_path):
                items.append({"rel_path": st.rel_path, "mtime": st.mtime, "size": st.size})
            return {"count": len(items), "items": items}

        if uri == "notes://config":
            cfg = self.cfg
            return {
                "doc_root": str(cfg.doc_root),
                "include_ext": list(cfg.include_ext),
                "data_dir": str(cfg.data_dir),
                "index_dir": str(cfg.index_dir),
                "manifest_path": str(cfg.manifest_path),
                "chat_history_path": str(cfg.chat_history_path),
                "chunk_chars": cfg.chunk_chars,
                "chunk_overlap": cfg.chunk_overlap,
                "top_k": cfg.top_k,
                "scan_interval_minutes": cfg.scan_interval_minutes,
                "embedding_model": cfg.embedding_model,
                "chat_model": cfg.chat_model,
                "max_history_turns": cfg.max_history_turns,
                "max_sources_chars": cfg.max_sources_chars,
                "max_file_size_mb": cfg.max_file_size_mb,
                "max_chunks_per_file": cfg.max_chunks_per_file,
            }

        if uri == "notes://reindex/status":
            return asdict(self._get_state())

        raise MCPError(f"Unknown resource URI: {uri}", code=-32602)

    def tools_list(self) -> list[dict[str, Any]]:
        out = []
        for name, spec in self._tools.items():
            out.append({"name": name, **spec})
        return out

    def resources_list(self) -> list[dict[str, Any]]:
        out = []
        for uri, spec in self._resources.items():
            out.append({"uri": uri, **spec})
        return out

    def prompts_list(self) -> list[dict[str, Any]]:
        out = []
        for name, spec in self._prompts.items():
            out.append({"name": name, **spec})
        return out

    def prompts_get(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        if name not in self._prompts:
            raise MCPError(f"Unknown prompt: {name}", code=-32602)
        args = arguments or {}

        if name == "grounded_note_answer":
            q = str(args.get("question", "")).strip()
            if not q:
                raise MCPError("Prompt argument 'question' is required", code=-32602)
            top_k = int(args.get("top_k", self.cfg.top_k))
            prompt_text = (
                "Use the notes MCP server tools to answer this question using only indexed notes.\n"
                "1) Call query_notes with the question and a reasonable limit.\n"
                "2) If evidence is insufficient, call get_note for cited files/lines.\n"
                "3) Respond with grounded citations and no outside knowledge.\n\n"
                f"Question: {q}\n"
                f"top_k: {top_k}"
            )
            return {
                "description": "Grounded notes QA workflow",
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": prompt_text}},
                ],
            }

        raise MCPError(f"Unhandled prompt: {name}", code=-32602)

    def tool_call(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        a = args or {}

        if name == "route_query":
            query = str(a.get("query", "")).strip()
            if not query:
                raise MCPError("query is required", code=-32602)
            return _tool_ok(self._route_query(query))

        if name == "query_notes":
            query = str(a.get("query", "")).strip()
            if not query:
                raise MCPError("query is required", code=-32602)
            limit = max(1, min(30, int(a.get("limit", 8))))
            return _tool_ok(self._query_notes(query, limit))

        if name == "eval_candidates":
            limit = max(1, min(200, int(a.get("limit", 25))))
            refresh = bool(a.get("refresh", True))
            return _tool_ok(self._eval_candidates(limit, refresh))

        if name == "eval_promote":
            promote_all = bool(a.get("promote_all", False))
            raw_selections = a.get("selections", [])
            selections: list[int] | None = None
            if not promote_all:
                if raw_selections is None:
                    selections = []
                elif not isinstance(raw_selections, list):
                    raise MCPError("selections must be an array of 1-based candidate indexes", code=-32602)
                else:
                    selections = [max(1, int(item)) for item in raw_selections]
            return _tool_ok(self._eval_promote(selections, promote_all))

        if name == "chat":
            query = str(a.get("query", "")).strip()
            if not query:
                raise MCPError("query is required", code=-32602)
            return _tool_ok(self._general_chat(query))

        if name == "list_indexed_files":
            limit = max(1, min(500, int(a.get("limit", 200))))
            offset = max(0, int(a.get("offset", 0)))
            all_files = self._list_indexed_files()
            page = all_files[offset : offset + limit]
            payload = {
                "total": len(all_files),
                "offset": offset,
                "limit": limit,
                "files": page,
                "has_more": offset + len(page) < len(all_files),
            }
            return _tool_ok(payload)

        if name == "find_files":
            term = str(a.get("term", "")).strip()
            if not term:
                raise MCPError("term is required", code=-32602)
            mode = str(a.get("mode", "both"))
            if mode not in ("filename", "text", "both"):
                raise MCPError("mode must be filename|text|both", code=-32602)
            limit = max(1, min(500, int(a.get("limit", 100))))
            items = self._search_indexed_files(term, mode, limit)
            return _tool_ok({"term": term, "mode": mode, "count": len(items), "results": items})

        if name == "search_notes":
            query = str(a.get("query", "")).strip()
            if not query:
                raise MCPError("query is required", code=-32602)
            top_k = max(1, min(100, int(a.get("top_k", self.cfg.top_k))))
            include_text = bool(a.get("include_text", True))
            max_chars = max(500, min(200000, int(a.get("max_chars", self.cfg.max_sources_chars))))
            payload = self._retrieve_notes(query, top_k=top_k, include_text=include_text, max_chars=max_chars)
            return _tool_ok(payload)

        if name == "get_note":
            rel_path = str(a.get("rel_path", "")).strip()
            if not rel_path:
                raise MCPError("rel_path is required", code=-32602)
            start_line = a.get("start_line")
            end_line = a.get("end_line")
            max_chars = max(200, min(500000, int(a.get("max_chars", 120000))))
            payload = self._get_note_excerpt(rel_path, start_line, end_line, max_chars)
            return _tool_ok(payload)

        if name == "reindex_status":
            st = asdict(self._get_state())
            if st.get("started_at") and st.get("running"):
                st["running_for_seconds"] = max(0, int(time.time() - float(st["started_at"])))
            return _tool_ok(st)

        if name == "reindex_now":
            wait = bool(a.get("wait", False))
            force_reindex = bool(a.get("force_reindex", False))
            timeout = max(1, min(86400, int(a.get("lock_timeout_seconds", 300))))

            if wait:
                locked = self._index_lock.acquire(timeout=timeout)
                if not locked:
                    return _tool_ok(
                        {
                            "ok": False,
                            "started": False,
                            "reason": "timeout_waiting_for_index_lock",
                            "status": asdict(self._get_state()),
                        }
                    )
                try:
                    payload = self._run_index_pass(owner="manual", force_reindex=force_reindex)
                    payload["started"] = True
                    payload["wait"] = True
                    return _tool_ok(payload)
                finally:
                    self._index_lock.release()

            if self._index_lock.locked():
                return _tool_ok(
                    {
                        "ok": False,
                        "started": False,
                        "reason": "indexer_busy",
                        "status": asdict(self._get_state()),
                    }
                )

            def _run_async() -> None:
                if self._index_lock.acquire(blocking=False):
                    try:
                        self._run_index_pass(owner="manual_async", force_reindex=force_reindex)
                    finally:
                        self._index_lock.release()

            threading.Thread(target=_run_async, daemon=True).start()
            return _tool_ok({
                "ok": True,
                "started": True,
                "wait": False,
                "mode": "full" if force_reindex else "incremental",
            })

        if name == "answer_from_notes":
            question = str(a.get("question", "")).strip()
            if not question:
                raise MCPError("question is required", code=-32602)
            top_k = max(1, min(30, int(a.get("top_k", self.cfg.top_k))))
            max_sources_chars = max(500, min(200000, int(a.get("max_sources_chars", self.cfg.max_sources_chars))))
            payload = self._answer_from_notes(question, top_k=top_k, max_sources_chars=max_sources_chars)
            return _tool_ok(payload)

        if name == "list_large_files":
            limit = max(1, min(500, int(a.get("limit", 100))))
            threshold = float(a.get("min_size_mb", self.cfg.max_file_size_mb))
            items: list[dict[str, Any]] = []
            for rel_path in self._list_indexed_files():
                try:
                    p = self._safe_doc_path(rel_path)
                    if not p.exists() or not p.is_file():
                        continue
                    size_bytes = p.stat().st_size
                    size_mb = size_bytes / (1024 * 1024)
                    if size_mb >= threshold:
                        items.append({"rel_path": rel_path, "size_mb": round(size_mb, 3), "size_bytes": size_bytes})
                    if len(items) >= limit:
                        break
                except Exception:
                    continue
            return _tool_ok({"threshold_mb": threshold, "count": len(items), "results": items})

        raise MCPError(f"Unknown tool: {name}", code=-32601)


def _tool_ok(payload: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
        "isError": False,
    }


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _read_message(stdin: Any) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        try:
            key, value = line.decode("utf-8").split(":", 1)
        except ValueError:
            continue
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None

    body = stdin.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(stdout: Any, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8")
    stdout.write(header)
    stdout.write(data)
    stdout.flush()


def serve_stdio(config_path: str | Path = "config.yaml", enable_background: bool = True) -> None:
    server = NotesMCPServer(config_path=config_path, enable_background=enable_background)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        while True:
            msg = _read_message(stdin)
            if msg is None:
                break

            method = msg.get("method")
            req_id = msg.get("id")
            params = msg.get("params", {}) or {}

            if req_id is None:
                if method == "notifications/initialized":
                    continue
                if method == "notifications/cancelled":
                    continue
                continue

            try:
                if method == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "resources": {"subscribe": False, "listChanged": False},
                            "prompts": {"listChanged": False},
                        },
                        "instructions": (
                            "Use this server to query local indexed notes, read files, and trigger reindex runs."
                        ),
                    }
                    _write_message(stdout, _jsonrpc_result(req_id, result))
                    continue

                if method == "ping":
                    _write_message(stdout, _jsonrpc_result(req_id, {}))
                    continue

                if method == "tools/list":
                    _write_message(stdout, _jsonrpc_result(req_id, {"tools": server.tools_list()}))
                    continue

                if method == "tools/call":
                    name = str(params.get("name", "")).strip()
                    args = params.get("arguments") or {}
                    result = server.tool_call(name, args)
                    _write_message(stdout, _jsonrpc_result(req_id, result))
                    continue

                if method == "resources/list":
                    _write_message(stdout, _jsonrpc_result(req_id, {"resources": server.resources_list()}))
                    continue

                if method == "resources/read":
                    uri = str(params.get("uri", "")).strip()
                    if not uri:
                        raise MCPError("uri is required", code=-32602)
                    payload = server._resource_value(uri)
                    text = json.dumps(payload, ensure_ascii=False, indent=2)
                    result = {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": text,
                            }
                        ]
                    }
                    _write_message(stdout, _jsonrpc_result(req_id, result))
                    continue

                if method == "prompts/list":
                    _write_message(stdout, _jsonrpc_result(req_id, {"prompts": server.prompts_list()}))
                    continue

                if method == "prompts/get":
                    name = str(params.get("name", "")).strip()
                    args = params.get("arguments") or {}
                    result = server.prompts_get(name, args)
                    _write_message(stdout, _jsonrpc_result(req_id, result))
                    continue

                raise MCPError(f"Method not found: {method}", code=-32601)

            except MCPError as e:
                _write_message(stdout, _jsonrpc_error(req_id, e.code, str(e)))
            except Exception as e:
                _write_message(stdout, _jsonrpc_error(req_id, -32603, f"Internal error: {e}"))
    finally:
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="notes_bot MCP server")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--no-background-index",
        action="store_true",
        help="Disable periodic background indexing loop",
    )
    args = parser.parse_args()
    serve_stdio(config_path=args.config, enable_background=not args.no_background_index)


if __name__ == "__main__":
    main()
