from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
from typing import Callable, Iterable

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)

from .doc_roots import DocRoot
from .manifest import Manifest
from .scanner import iter_files, DiscoveredFile
from .chunker import chunk_with_line_ranges
from .store import VectorStore

@dataclass
class IndexStats:
    scanned: int = 0
    updated: int = 0
    deleted: int = 0
    errors: int = 0

def _stable_chunk_id(rel_path: str, mtime: float, chunk_index: int) -> str:
    # Deterministic enough; changes when file mtime changes.
    raw = f"{rel_path}::{mtime}::{chunk_index}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()

def _read_text(path: Path) -> str:
    # Notes are English; still be tolerant.
    return path.read_text(encoding="utf-8", errors="replace")

def _embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _is_changed_file(f: DiscoveredFile, manifest: Manifest) -> bool:
    prev = manifest.get(f.rel_path)
    return (
        (prev is None)
        or (prev.mtime != f.mtime)
        or (prev.size != f.size)
        or (prev.status in ("error", "stale"))
    )


def _should_retry_error(prev, retry_error_after_minutes: int) -> bool:
    if prev is None or prev.status != "error":
        return False
    if int(retry_error_after_minutes) <= 0:
        return True
    retry_after_seconds = max(1, int(retry_error_after_minutes)) * 60
    last_attempt_at = prev.last_indexed_at or prev.updated_at or 0.0
    if last_attempt_at <= 0:
        return True
    return (time.time() - last_attempt_at) >= retry_after_seconds


def should_reindex_file(
    f: DiscoveredFile,
    manifest: Manifest,
    *,
    retry_error_after_minutes: int,
) -> bool:
    prev = manifest.get(f.rel_path)
    if prev is None:
        return True
    if prev.mtime != f.mtime or prev.size != f.size:
        return True
    if prev.status == "stale":
        return True
    if prev.status == "error":
        return _should_retry_error(prev, retry_error_after_minutes)
    return False


def _detail_from_openai_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return ""
    low = message.lower()
    if low.startswith("error code:"):
        return ""
    return message


def _format_embedding_preflight_error(exc: Exception, model: str) -> str:
    detail = _detail_from_openai_error(exc)
    suffix = f" Detail: {detail}" if detail else ""

    if isinstance(exc, PermissionDeniedError):
        return (
            f"Embedding access denied for model '{model}'. "
            f"Check that your API key/project has access to this embedding model, "
            f"or change 'embedding_model' in config.yaml.{suffix}"
        )
    if isinstance(exc, AuthenticationError):
        return f"OpenAI authentication failed while validating embedding model '{model}'.{suffix}"
    if isinstance(exc, NotFoundError):
        return (
            f"Embedding model '{model}' was not found. "
            f"Update 'embedding_model' in config.yaml to a valid model name.{suffix}"
        )
    if isinstance(exc, BadRequestError):
        return f"Embedding request was rejected for model '{model}'. Check the configured model name.{suffix}"
    if isinstance(exc, APIConnectionError):
        return (
            f"Could not reach the OpenAI embeddings API for model '{model}'. "
            f"Check network access and try again.{suffix}"
        )
    if isinstance(exc, APIStatusError):
        return f"OpenAI returned HTTP {exc.status_code} while validating embedding model '{model}'.{suffix}"
    return f"Failed to validate embedding model '{model}'.{suffix}"


def _preflight_embedding_access(client: OpenAI, model: str) -> None:
    try:
        client.embeddings.create(model=model, input=["healthcheck"])
    except Exception as exc:
        raise RuntimeError(_format_embedding_preflight_error(exc, model)) from exc


def _empty_embeddings(count: int) -> list[list[float]]:
    return [[] for _ in range(max(0, count))]


def _mark_searchable_state(
    manifest: Manifest,
    rel_path: str,
    mtime: float,
    size: int,
    chunk_count: int,
) -> None:
    now = time.time()
    manifest.mark_status(
        rel_path=rel_path,
        mtime=mtime,
        size=size,
        status="indexed",
        last_error="",
        chunk_count=chunk_count,
        last_indexed_at=now,
        last_success_at=now,
    )

def run_index_once(
    *,
    client: OpenAI,
    doc_roots: tuple[DocRoot, ...],
    include_ext: tuple[str, ...],
    manifest: Manifest,
    store: VectorStore,
    embedding_model: str,
    chunk_chars: int,
    chunk_overlap: int,
    batch_size: int = 96,
    max_file_size_mb: int = 8,
    max_chunks_per_file: int = 2000,
    retry_error_after_minutes: int = 60,
    progress_callback: Callable[[dict], None] | None = None,
) -> IndexStats:
    stats = IndexStats()
    seen_paths: set[str] = set()
    files = list(iter_files(doc_roots, include_ext))
    total_files = len(files)
    pending_files = [
        f for f in files
        if should_reindex_file(
            f,
            manifest,
            retry_error_after_minutes=retry_error_after_minutes,
        )
    ]
    embeddings_enabled = True

    if pending_files:
        try:
            _preflight_embedding_access(client, embedding_model)
        except RuntimeError as exc:
            embeddings_enabled = False
            if progress_callback:
                progress_callback({
                    "phase": "warning",
                    "status": "embedding_unavailable",
                    "error": str(exc),
                })

    for idx, f in enumerate(files, start=1):
        stats.scanned += 1
        seen_paths.add(f.rel_path)
        inserted_ids: list[str] = []

        prev = manifest.get(f.rel_path)
        changed = should_reindex_file(
            f,
            manifest,
            retry_error_after_minutes=retry_error_after_minutes,
        )
        if progress_callback:
            progress_callback({
                "phase": "scan",
                "index": idx,
                "total": total_files,
                "rel_path": f.rel_path,
                "status": "updating" if changed else "unchanged",
                "stats": stats,
            })
        if not changed:
            continue

        try:
            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "read_start",
                    "stats": stats,
                })
            file_size_mb = f.size / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                store.delete_file(f.rel_path)
                manifest.mark_status(
                    rel_path=f.rel_path,
                    mtime=f.mtime,
                    size=f.size,
                    status="skipped_large",
                    last_error=f"file exceeds max size of {max_file_size_mb}MB",
                    chunk_count=0,
                    last_success_at=prev.last_success_at if prev is not None else 0.0,
                )
                stats.errors += 1
                if progress_callback:
                    progress_callback({
                        "phase": "scan",
                        "index": idx,
                        "total": total_files,
                        "rel_path": f.rel_path,
                        "status": "skipped_large",
                        "size_mb": round(file_size_mb, 2),
                        "max_file_size_mb": max_file_size_mb,
                        "stats": stats,
                    })
                continue

            text = _read_text(f.abs_path)
            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "read_done",
                    "chars": len(text),
                    "stats": stats,
                })
            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "chunk_start",
                    "stats": stats,
                })
            chunks = chunk_with_line_ranges(text, chunk_chars=chunk_chars, overlap=chunk_overlap)
            if len(chunks) > max_chunks_per_file:
                chunks = chunks[:max_chunks_per_file]
            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "chunk_done",
                    "chunks": len(chunks),
                    "max_chunks_per_file": max_chunks_per_file,
                    "stats": stats,
                })

            total_chunks = len(chunks)
            total_batches = max(1, (total_chunks + batch_size - 1) // batch_size) if total_chunks else 0
            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "chunked",
                    "chunks": total_chunks,
                    "batches": total_batches,
                    "stats": stats,
                })

            batch_ids: list[str] = []
            batch_texts: list[str] = []
            batch_metas: list[dict] = []
            all_ids: list[str] = []
            all_texts: list[str] = []
            all_metas: list[dict] = []
            batch_no = 0
            for ch in chunks:
                chunk_id = _stable_chunk_id(f.rel_path, f.mtime, ch.chunk_index)
                meta = {
                    "rel_path": f.rel_path,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "chunk_index": ch.chunk_index,
                    "content_type": ch.content_type,
                    "title": ch.title,
                    "section_path": ch.section_path,
                    "mtime": f.mtime,
                }
                all_ids.append(chunk_id)
                all_texts.append(ch.text)
                all_metas.append(meta)
                batch_ids.append(chunk_id)
                batch_texts.append(ch.text)
                batch_metas.append(meta)
                if len(batch_texts) < batch_size:
                    continue

                if not embeddings_enabled:
                    continue

                batch_no += 1
                if progress_callback:
                    progress_callback({
                        "phase": "file",
                        "index": idx,
                        "total": total_files,
                        "rel_path": f.rel_path,
                        "status": "embedding_batch",
                        "batch": batch_no,
                        "batches": total_batches,
                        "batch_size": len(batch_texts),
                        "stats": stats,
                    })
                try:
                    be = _embed_texts(client, embedding_model, batch_texts)
                    store.add_chunks(
                        ids=batch_ids,
                        texts=batch_texts,
                        embeddings=be,
                        metadatas=batch_metas,
                    )
                    inserted_ids.extend(batch_ids)
                    batch_ids.clear()
                    batch_texts.clear()
                    batch_metas.clear()
                except Exception:
                    embeddings_enabled = False

            if embeddings_enabled and batch_texts:
                batch_no += 1
                if progress_callback:
                    progress_callback({
                        "phase": "file",
                        "index": idx,
                        "total": total_files,
                        "rel_path": f.rel_path,
                        "status": "embedding_batch",
                        "batch": batch_no,
                        "batches": total_batches,
                        "batch_size": len(batch_texts),
                        "stats": stats,
                    })
                try:
                    be = _embed_texts(client, embedding_model, batch_texts)
                    store.add_chunks(
                        ids=batch_ids,
                        texts=batch_texts,
                        embeddings=be,
                        metadatas=batch_metas,
                    )
                    inserted_ids.extend(batch_ids)
                except Exception:
                    embeddings_enabled = False

            if not embeddings_enabled and all_ids:
                if inserted_ids:
                    store.delete_ids(inserted_ids)
                    inserted_ids.clear()
                if progress_callback:
                    progress_callback({
                        "phase": "file",
                        "index": idx,
                        "total": total_files,
                        "rel_path": f.rel_path,
                        "status": "lexical_only",
                        "chunks": total_chunks,
                        "stats": stats,
                    })
                store.add_chunks(
                    ids=all_ids,
                    texts=all_texts,
                    embeddings=_empty_embeddings(len(all_ids)),
                    metadatas=all_metas,
                )
                inserted_ids.extend(all_ids)

            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "delete_old",
                    "stats": stats,
                })
            store.delete_file_except(f.rel_path, inserted_ids)
            _mark_searchable_state(manifest, f.rel_path, f.mtime, f.size, total_chunks)
            stats.updated += 1
            if progress_callback:
                progress_callback({
                    "phase": "scan",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "updated",
                    "stats": stats,
                })

        except Exception as e:
            if inserted_ids:
                store.delete_ids(inserted_ids)
            stats.errors += 1
            if prev is not None and prev.status in ("indexed", "stale"):
                manifest.mark_status(
                    rel_path=f.rel_path,
                    mtime=prev.mtime,
                    size=prev.size,
                    status="stale",
                    last_error=str(e),
                    chunk_count=prev.chunk_count,
                    last_success_at=prev.last_success_at,
                )
            else:
                manifest.mark_status(
                    rel_path=f.rel_path,
                    mtime=f.mtime,
                    size=f.size,
                    status="error",
                    last_error=str(e),
                    chunk_count=0,
                    last_success_at=0.0,
                )
            if progress_callback:
                progress_callback({
                    "phase": "scan",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "error",
                    "error": str(e),
                    "stats": stats,
                })

    # Handle deletions
    known = manifest.all_paths()
    missing = known - seen_paths
    missing_list = sorted(missing)
    for idx, rel_path in enumerate(missing_list, start=1):
        store.delete_file(rel_path)
        manifest.delete(rel_path)
        stats.deleted += 1
        if progress_callback:
            progress_callback({
                "phase": "delete",
                "index": idx,
                "total": len(missing_list),
                "rel_path": rel_path,
                "status": "deleted",
                "stats": stats,
            })

    if progress_callback:
        progress_callback({
            "phase": "done",
            "stats": stats,
            "total_files": total_files,
            "deleted_total": len(missing_list),
        })

    return stats
    
