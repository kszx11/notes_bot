from __future__ import annotations

import re


def detect_query_intent(text: str) -> str:
    query = text.strip()
    low = query.lower()
    if not query:
        return "answer"

    file_patterns = (
        "what files",
        "which files",
        "show files",
        "list files",
        "find files",
    )
    if any(p in low for p in file_patterns):
        return "file_search"

    if low.startswith(("find ", "search ", "lookup ")):
        return "passage_search"

    question_words = (
        "what ",
        "how ",
        "why ",
        "when ",
        "where ",
        "who ",
        "which ",
        "does ",
        "do ",
        "did ",
        "is ",
        "are ",
        "can ",
        "could ",
        "should ",
        "would ",
    )
    if low.endswith("?") or any(low.startswith(p) for p in question_words):
        return "answer"

    token_count = len(re.findall(r"[A-Za-z0-9_./:-]+", query))
    if token_count <= 6:
        return "passage_search"

    if re.search(r"\b(error|exception|stacktrace|docker|compose|systemd|config|yaml|json|sql|wazuh)\b", low):
        return "passage_search"

    return "answer"


def describe_query_intent(text: str) -> dict:
    intent = detect_query_intent(text)
    if intent == "file_search":
        return {
            "intent": intent,
            "description": "Search indexed files by filename and/or text.",
            "suggested_tool": "find_files",
            "mode": "both",
            "exploratory": True,
        }
    if intent == "passage_search":
        return {
            "intent": intent,
            "description": "Search ranked note passages and inspect excerpts first.",
            "suggested_tool": "search_notes",
            "mode": "passages",
            "exploratory": True,
        }
    return {
        "intent": "answer",
        "description": "Generate a grounded answer from retrieved note passages.",
        "suggested_tool": "answer_from_notes",
        "mode": "answer",
        "exploratory": False,
    }
