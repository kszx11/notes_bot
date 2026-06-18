from __future__ import annotations

import unittest

from notes_bot.chat import _run_auto_query


class _FakeRetrieval:
    def __init__(self) -> None:
        self.answer_calls: list[dict] = []
        self.passage_calls: list[dict] = []

    def answer_question(self, *, question: str, top_k: int, max_sources_chars: int, root_name: str | None = None) -> dict:
        self.answer_calls.append(
            {
                "question": question,
                "top_k": top_k,
                "max_sources_chars": max_sources_chars,
                "root_name": root_name,
            }
        )
        return {"answer": "I can't find that in your notes."}

    def retrieve_passages(
        self,
        *,
        query: str,
        top_k: int,
        include_text: bool,
        max_chars: int,
        root_name: str | None = None,
    ) -> dict:
        self.passage_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "include_text": include_text,
                "max_chars": max_chars,
                "root_name": root_name,
            }
        )
        return {
            "results": [
                {
                    "rank": 1,
                    "rel_path": "note.md",
                    "start_line": 10,
                    "end_line": 12,
                    "section_path": "Runbook",
                    "status": "indexed",
                    "keyword_overlap": 0.75,
                    "text": "Authentication failure details",
                }
            ],
            "retrieval": {
                "confidence": "medium",
                "reason": "strong keyword support",
                "indexed_results": 1,
                "stale_results": 0,
            },
        }


class ChatAutoRoutingTests(unittest.TestCase):
    def test_question_falls_back_to_passage_results_when_grounded_answer_is_unsupported(self) -> None:
        retrieval = _FakeRetrieval()

        answer, search_results = _run_auto_query(
            "Why is authentication failing?",
            retrieval,
            answer_top_k=6,
            max_sources_chars=4000,
        )

        self.assertEqual(len(retrieval.answer_calls), 1)
        self.assertEqual(len(retrieval.passage_calls), 1)
        self.assertIn("Top passages for 'Why is authentication failing?':", answer)
        self.assertIn("Use /open <n> to inspect one of these results.", answer)
        self.assertEqual(search_results[0]["rel_path"], "note.md")

    def test_root_scope_is_forwarded_to_answer_and_passage_calls(self) -> None:
        retrieval = _FakeRetrieval()

        answer, _search_results = _run_auto_query(
            "Why is authentication failing?",
            retrieval,
            answer_top_k=6,
            max_sources_chars=4000,
            root_name="personal1",
        )

        self.assertEqual(retrieval.answer_calls[0]["root_name"], "personal1")
        self.assertEqual(retrieval.passage_calls[0]["root_name"], "personal1")
        self.assertIn("Scope: personal1", answer)


if __name__ == "__main__":
    unittest.main()
