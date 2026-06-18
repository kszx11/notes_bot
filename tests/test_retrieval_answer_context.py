from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from notes_bot.answering import format_grounded_answer
from notes_bot.retrieval import RetrievalService, format_passage_results


class _FakeStore:
    def __init__(self) -> None:
        self.last_query_embedding = None
        self.last_rel_path_prefix = None

    def query(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int,
        rel_path_prefix: str | None = None,
    ) -> dict:
        self.last_query_embedding = query_embedding
        self.last_rel_path_prefix = rel_path_prefix
        return {
            "ids": [["a", "b"]],
            "documents": [[
                "target details",
                "target details repeated",
            ]],
            "metadatas": [[
                {
                    "rel_path": "note.md",
                    "start_line": 6,
                    "end_line": 7,
                    "chunk_index": 1,
                    "mtime": 1.0,
                    "content_type": "prose",
                    "title": "Test Note",
                    "section_path": "Root > Section",
                },
                {
                    "rel_path": "note.md",
                    "start_line": 7,
                    "end_line": 8,
                    "chunk_index": 2,
                    "mtime": 1.0,
                    "content_type": "prose",
                    "title": "Test Note",
                    "section_path": "Root > Section",
                },
            ]],
            "distances": [[0.1, 0.12]],
        }

    def get_chunk_window(self, rel_path: str, center_chunk_index: int, window_size: int = 1) -> list[dict]:
        return [
            {
                "rel_path": "note.md",
                "start_line": 4,
                "end_line": 5,
                "chunk_index": 0,
                "mtime": 1.0,
                "content_type": "prose",
                "title": "Test Note",
                "section_path": "Root > Section",
                "document": "surrounding context before\nmore setup",
            },
            {
                "rel_path": "note.md",
                "start_line": 6,
                "end_line": 7,
                "chunk_index": 1,
                "mtime": 1.0,
                "content_type": "prose",
                "title": "Test Note",
                "section_path": "Root > Section",
                "document": "target details\ntarget details repeated",
            },
            {
                "rel_path": "note.md",
                "start_line": 8,
                "end_line": 9,
                "chunk_index": 2,
                "mtime": 1.0,
                "content_type": "prose",
                "title": "Test Note",
                "section_path": "Root > Section",
                "document": "supporting line after\ntail",
            },
        ]


class _FakeManifest:
    def get(self, rel_path: str):
        return SimpleNamespace(status="indexed")


class RetrievalAnswerContextTests(unittest.TestCase):
    def test_passage_results_format_uses_readable_sections(self) -> None:
        formatted = format_passage_results(
            "target details",
            {
                "results": [
                    {
                        "rank": 1,
                        "rel_path": "note.md",
                        "start_line": 4,
                        "end_line": 9,
                        "section_path": "Root > Section",
                        "status": "indexed",
                        "keyword_overlap": 0.75,
                        "text": "surrounding context before target details supporting line after tail",
                    }
                ],
                "retrieval": {
                    "confidence": "high",
                    "reason": "strong overlap",
                    "indexed_results": 1,
                    "stale_results": 0,
                },
            },
        )

        self.assertIn("Confidence: high", formatted)
        self.assertIn("Freshness: indexed=1 stale=0", formatted)
        self.assertIn("1. note.md:4-9", formatted)
        self.assertIn("  Section: Root > Section", formatted)
        self.assertIn("  Preview:\n", formatted)

    def test_grounded_answer_format_uses_numbered_evidence_blocks(self) -> None:
        formatted = format_grounded_answer(
            "The note covers the target details clearly.",
            [{"rel_path": "note.md", "start_line": 2, "end_line": 9}],
            ["surrounding context before target details supporting line after tail"],
        )

        self.assertIn("Answer:\n  The note covers the target details clearly.", formatted)
        self.assertIn("Evidence:\n1. note.md:2-9", formatted)
        self.assertIn('   "surrounding context before target details', formatted)

    def test_passage_search_falls_back_to_lexical_when_query_embedding_fails(self) -> None:
        store = _FakeStore()
        retrieval = RetrievalService(
            store=store,
            manifest=_FakeManifest(),
            client=SimpleNamespace(),
            embedding_model="fake",
            chat_model="fake",
            adjacent_chunk_window=1,
            embed_query_fn=lambda text: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
            synthesize_answer_fn=lambda question, sources_text: "unused",
        )

        payload = retrieval.retrieve_passages(
            query="target details",
            top_k=4,
            include_text=True,
            max_chars=12000,
        )

        self.assertEqual(store.last_query_embedding, None)
        self.assertEqual(len(payload["results"]), 1)

    def test_passage_search_forwards_root_scope_to_store(self) -> None:
        store = _FakeStore()
        retrieval = RetrievalService(
            store=store,
            manifest=_FakeManifest(),
            client=SimpleNamespace(),
            embedding_model="fake",
            chat_model="fake",
            doc_roots=(
                SimpleNamespace(name="personal1", path=Path("/tmp/p1")),
                SimpleNamespace(name="personal2", path=Path("/tmp/p2")),
            ),
            adjacent_chunk_window=1,
            embed_query_fn=lambda text: [0.0],
            synthesize_answer_fn=lambda question, sources_text: "unused",
        )

        retrieval.retrieve_passages(
            query="target details",
            top_k=4,
            include_text=True,
            max_chars=12000,
            root_name="personal1",
        )

        self.assertEqual(store.last_rel_path_prefix, "personal1/")

    def test_passage_search_expands_adjacent_chunks_and_skips_duplicate_neighbors(self) -> None:
        retrieval = RetrievalService(
            store=_FakeStore(),
            manifest=_FakeManifest(),
            client=SimpleNamespace(),
            embedding_model="fake",
            chat_model="fake",
            adjacent_chunk_window=1,
            embed_query_fn=lambda text: [0.0],
            synthesize_answer_fn=lambda question, sources_text: "unused",
        )

        payload = retrieval.retrieve_passages(
            query="target details",
            top_k=4,
            include_text=True,
            max_chars=12000,
        )

        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["start_line"], 4)
        self.assertEqual(payload["results"][0]["end_line"], 9)
        self.assertIn("surrounding context before", payload["results"][0]["text"])
        self.assertIn("supporting line after", payload["results"][0]["text"])

    def test_answer_uses_expanded_excerpt_and_dedupes_overlapping_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc_root = Path(tmp)
            note_path = doc_root / "note.md"
            note_path.write_text(
                "\n".join(
                    [
                        "# Test Note",
                        "alpha",
                        "beta",
                        "surrounding context before",
                        "more setup",
                        "target details",
                        "target details repeated",
                        "supporting line after",
                        "tail",
                    ]
                ),
                encoding="utf-8",
            )

            retrieval = RetrievalService(
                store=_FakeStore(),
                manifest=_FakeManifest(),
                client=SimpleNamespace(),
                embedding_model="fake",
                chat_model="fake",
                doc_roots=(
                    SimpleNamespace(name="notes", path=doc_root),
                ),
                answer_context_before_lines=2,
                answer_context_after_lines=2,
                adjacent_chunk_window=1,
                embed_query_fn=lambda text: [0.0],
                synthesize_answer_fn=lambda question, sources_text: "The note covers the target details.",
            )

            payload = retrieval.answer_question(
                question="Where are the target details?",
                top_k=4,
                max_sources_chars=12000,
            )

            self.assertNotEqual(payload["confidence"], "low")
            self.assertEqual(len(payload["sources"]), 1)
            self.assertEqual(payload["sources"][0]["start_line"], 2)
            self.assertEqual(payload["sources"][0]["end_line"], 9)
            self.assertIn("surrounding context before", payload["answer"])
            self.assertIn("1. note.md:2-9", payload["answer"])


if __name__ == "__main__":
    unittest.main()
