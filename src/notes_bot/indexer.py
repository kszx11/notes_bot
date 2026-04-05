from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Callable, Iterable

from openai import OpenAI

from .manifest import Manifest, FileState
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

def run_index_once(
    *,
    client: OpenAI,
    doc_root: Path,
    include_ext: tuple[str, ...],
    manifest: Manifest,
    store: VectorStore,
    embedding_model: str,
    chunk_chars: int,
    chunk_overlap: int,
    batch_size: int = 96,
    max_file_size_mb: int = 8,
    max_chunks_per_file: int = 2000,
    progress_callback: Callable[[dict], None] | None = None,
) -> IndexStats:
    stats = IndexStats()
    seen_paths: set[str] = set()
    files = list(iter_files(doc_root, include_ext))
    total_files = len(files)

    for idx, f in enumerate(files, start=1):
        stats.scanned += 1
        seen_paths.add(f.rel_path)

        prev = manifest.get(f.rel_path)
        changed = (prev is None) or (prev.mtime != f.mtime) or (prev.size != f.size)
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

            # Replace all chunks for this file
            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "delete_start",
                    "stats": stats,
                })
            store.delete_file(f.rel_path)
            if progress_callback:
                progress_callback({
                    "phase": "file",
                    "index": idx,
                    "total": total_files,
                    "rel_path": f.rel_path,
                    "status": "delete_done",
                    "stats": stats,
                })

            # Embed + upsert in bounded-memory batches
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
            batch_no = 0

            for ch in chunks:
                batch_ids.append(_stable_chunk_id(f.rel_path, f.mtime, ch.chunk_index))
                batch_texts.append(ch.text)
                batch_metas.append({
                    "rel_path": f.rel_path,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "chunk_index": ch.chunk_index,
                    "mtime": f.mtime,
                })
                if len(batch_texts) < batch_size:
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
                be = _embed_texts(client, embedding_model, batch_texts)
                store.add_chunks(
                    ids=batch_ids,
                    texts=batch_texts,
                    embeddings=be,
                    metadatas=batch_metas,
                )
                batch_ids.clear()
                batch_texts.clear()
                batch_metas.clear()

            if batch_texts:
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
                be = _embed_texts(client, embedding_model, batch_texts)
                store.add_chunks(
                    ids=batch_ids,
                    texts=batch_texts,
                    embeddings=be,
                    metadatas=batch_metas,
                )

            manifest.upsert(FileState(rel_path=f.rel_path, mtime=f.mtime, size=f.size))
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
            stats.errors += 1
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
    
