from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import PermissionDeniedError

from notes_bot.indexer import run_index_once


class _FakeManifest:
    def __init__(self, prior=None) -> None:
        self.prior = prior or {}
        self.marked: list[dict] = []

    def get(self, rel_path: str):
        return self.prior.get(rel_path)

    def mark_status(self, **kwargs) -> None:
        self.marked.append(kwargs)

    def all_paths(self) -> set[str]:
        return set(self.prior.keys())

    def delete(self, rel_path: str) -> None:
        self.prior.pop(rel_path, None)


class _FakeStore:
    def __init__(self) -> None:
        self.added: list[dict] = []

    def delete_file(self, rel_path: str) -> None:
        return None

    def delete_ids(self, ids: list[str]) -> None:
        return None

    def delete_file_except(self, rel_path: str, keep_ids: list[str]) -> None:
        return None

    def add_chunks(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        self.added.append(
            {
                "ids": list(ids),
                "texts": list(texts),
                "embeddings": [list(item) for item in embeddings],
                "metadatas": [dict(item) for item in metadatas],
            }
        )


class _ForbiddenEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *, model: str, input):
        self.calls += 1
        response = httpx.Response(
            403,
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        raise PermissionDeniedError(
            f"model '{model}' is not permitted for this project",
            response=response,
            body={"error": {"message": "forbidden"}},
        )


class _FakeClient:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings


class IndexerPreflightTests(unittest.TestCase):
    def test_pending_index_falls_back_to_lexical_index_on_embedding_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note_path = root / "note.md"
            note_path.write_text("hello world\n", encoding="utf-8")

            manifest = _FakeManifest()
            store = _FakeStore()
            embeddings = _ForbiddenEmbeddings()

            stats = run_index_once(
                client=_FakeClient(embeddings),
                doc_roots=(SimpleNamespace(name="notes", path=root),),
                include_ext=(".md",),
                manifest=manifest,
                store=store,
                embedding_model="text-embedding-3-small",
                chunk_chars=1000,
                chunk_overlap=0,
            )

            self.assertEqual(stats.updated, 1)
            self.assertEqual(stats.errors, 0)
            self.assertEqual(embeddings.calls, 1)
            self.assertEqual(manifest.marked[0]["status"], "indexed")
            self.assertEqual(store.added[0]["embeddings"], [[]])

    def test_unchanged_files_skip_embedding_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note_path = root / "note.md"
            note_path.write_text("hello world\n", encoding="utf-8")
            stat = note_path.stat()

            manifest = _FakeManifest(
                prior={
                    "note.md": SimpleNamespace(
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                        status="indexed",
                    )
                }
            )
            embeddings = _ForbiddenEmbeddings()

            stats = run_index_once(
                client=_FakeClient(embeddings),
                doc_roots=(SimpleNamespace(name="notes", path=root),),
                include_ext=(".md",),
                manifest=manifest,
                store=_FakeStore(),
                embedding_model="text-embedding-3-small",
                chunk_chars=1000,
                chunk_overlap=0,
            )

            self.assertEqual(stats.scanned, 1)
            self.assertEqual(stats.updated, 0)
            self.assertEqual(stats.errors, 0)
            self.assertEqual(embeddings.calls, 0)

    def test_unchanged_error_files_retry_even_without_mtime_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note_path = root / "note.md"
            note_path.write_text("hello world\n", encoding="utf-8")
            stat = note_path.stat()

            manifest = _FakeManifest(
                prior={
                    "note.md": SimpleNamespace(
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                        status="error",
                    )
                }
            )
            store = _FakeStore()
            embeddings = _ForbiddenEmbeddings()

            stats = run_index_once(
                client=_FakeClient(embeddings),
                doc_roots=(SimpleNamespace(name="notes", path=root),),
                include_ext=(".md",),
                manifest=manifest,
                store=store,
                embedding_model="text-embedding-3-small",
                chunk_chars=1000,
                chunk_overlap=0,
            )

            self.assertEqual(stats.scanned, 1)
            self.assertEqual(stats.updated, 1)
            self.assertEqual(stats.errors, 0)
            self.assertEqual(embeddings.calls, 1)
            self.assertEqual(manifest.marked[0]["status"], "indexed")
            self.assertEqual(store.added[0]["embeddings"], [[]])


if __name__ == "__main__":
    unittest.main()
