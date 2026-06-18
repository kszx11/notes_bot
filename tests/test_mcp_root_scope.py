from __future__ import annotations

import unittest
from types import SimpleNamespace

from notes_bot.mcp_server import MCPError, NotesMCPServer


class _FakeRetrieval:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def search_files(self, query: str, **kwargs):
        self.calls.append(("search_files", {"query": query, **kwargs}))
        return [{"rel_path": "personal1/note.md"}]

    def retrieve_passages(self, **kwargs):
        self.calls.append(("retrieve_passages", kwargs))
        return {"results": [], "retrieval": {"root_name": kwargs.get("root_name")}}

    def answer_question(self, **kwargs):
        self.calls.append(("answer_question", kwargs))
        return {"answer": "Scoped answer", "root_name": kwargs.get("root_name")}


class MCPRootScopeTests(unittest.TestCase):
    def _make_server(self) -> NotesMCPServer:
        server = NotesMCPServer.__new__(NotesMCPServer)
        server.cfg = SimpleNamespace(
            doc_roots=(
                SimpleNamespace(name="personal1"),
                SimpleNamespace(name="personal2"),
            ),
            top_k=10,
            max_sources_chars=35000,
            max_file_size_mb=8,
        )
        server.retrieval = _FakeRetrieval()
        return server

    def test_find_files_forwards_root_name(self) -> None:
        server = self._make_server()

        result = server.tool_call(
            "find_files",
            {"term": "prayer", "root_name": "personal1", "mode": "both"},
        )

        self.assertEqual(server.retrieval.calls[0][0], "search_files")
        self.assertEqual(server.retrieval.calls[0][1]["root_name"], "personal1")
        self.assertEqual(result["structuredContent"]["root_name"], "personal1")

    def test_search_notes_forwards_root_name(self) -> None:
        server = self._make_server()

        result = server.tool_call(
            "search_notes",
            {"query": "prayer", "root_name": "personal2"},
        )

        self.assertEqual(server.retrieval.calls[0][0], "retrieve_passages")
        self.assertEqual(server.retrieval.calls[0][1]["root_name"], "personal2")
        self.assertEqual(result["structuredContent"]["retrieval"]["root_name"], "personal2")

    def test_answer_from_notes_rejects_unknown_root(self) -> None:
        server = self._make_server()

        with self.assertRaises(MCPError) as ctx:
            server.tool_call(
                "answer_from_notes",
                {"question": "What is this?", "root_name": "missing"},
            )

        self.assertEqual(ctx.exception.code, -32602)


if __name__ == "__main__":
    unittest.main()
