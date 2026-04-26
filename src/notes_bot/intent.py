from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RouteMode = Literal["command", "notes_search", "general_chat"]

_COMMAND_ALIASES = {
    "/help": "help",
    "/clear": "clear",
    "/reindex": "reindex",
    "/reindex-force": "reindex_force",
    "/reindex-status": "reindex_status",
    "/eval-candidates": "eval_candidates",
    "/indexed": "indexed",
    "/exit": "exit",
}

_COMMAND_PHRASES = {
    "help": "help",
    "what can you do": "help",
    "show help": "help",
    "reindex now": "reindex",
    "reindex the notes": "reindex",
    "reindex notes": "reindex",
    "force reindex": "reindex_force",
    "force reindex now": "reindex_force",
    "force reindex notes": "reindex_force",
    "reindex everything": "reindex_force",
    "rebuild the index": "reindex_force",
    "reindex status": "reindex_status",
    "index status": "reindex_status",
    "show reindex status": "reindex_status",
    "show index status": "reindex_status",
    "show eval candidates": "eval_candidates",
    "list eval candidates": "eval_candidates",
    "refresh eval candidates": "eval_candidates",
    "show indexed files": "indexed",
    "list indexed files": "indexed",
    "what files have been indexed": "indexed",
    "which files have been indexed": "indexed",
    "what files are indexed": "indexed",
    "which files are indexed": "indexed",
}

_NOTES_SEARCH_PHRASES = (
    "my notes",
    "my files",
    "my file",
    "my note",
    "where did i",
    "did i mention",
    "did i write",
    "did i note",
    "find my note",
    "find my notes",
    "find my file",
    "search my notes",
    "search my files",
    "in the title",
    "in filename",
    "in the filename",
    "file named",
    "note named",
    "which file",
    "what file",
    "which files",
    "what files",
)

_NOTES_NOUN_RE = re.compile(r"\b(note|notes|file|files|document|documents|snippet|snippets|passage|passages)\b")
_NOTES_SEARCH_VERB_RE = re.compile(
    r"\b(find|search|show|list|locate|mention|mentioned|mentions|contains|containing|about|with|title|filename|named)\b"
)
_PERSONAL_MEMORY_RE = re.compile(r"\b(where did i|did i mention|did i write|did i note|what did i say)\b")


@dataclass
class RouteDecision:
    mode: RouteMode
    confidence: float
    reasons: list[str]
    command_name: str | None = None


def normalize_user_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_command(text: str) -> tuple[str | None, list[str]]:
    normalized = normalize_user_text(text)
    if not normalized:
        return None, []

    if normalized in _COMMAND_ALIASES:
        return _COMMAND_ALIASES[normalized], ["explicit slash command"]

    if normalized.startswith("/eval-promote"):
        return "eval_promote", ["explicit slash command with arguments"]

    if normalized in _COMMAND_PHRASES:
        return _COMMAND_PHRASES[normalized], ["natural-language command phrase"]

    if normalized.startswith("promote eval candidates"):
        return "eval_promote", ["natural-language command phrase with arguments"]

    return None, []


def detect_notes_search_signals(text: str) -> list[str]:
    normalized = normalize_user_text(text)
    reasons: list[str] = []

    for phrase in _NOTES_SEARCH_PHRASES:
        if phrase in normalized:
            reasons.append(f"matched phrase '{phrase}'")
            break

    if _PERSONAL_MEMORY_RE.search(normalized):
        reasons.append("personal memory wording")

    if _NOTES_NOUN_RE.search(normalized) and _NOTES_SEARCH_VERB_RE.search(normalized):
        reasons.append("notes noun + search wording")

    if normalized.startswith("find ") and _NOTES_NOUN_RE.search(normalized):
        reasons.append("find query aimed at files or notes")

    return reasons


def route_user_input(text: str) -> RouteDecision:
    command_name, command_reasons = detect_command(text)
    if command_name is not None:
        return RouteDecision(
            mode="command",
            confidence=1.0,
            reasons=command_reasons,
            command_name=command_name,
        )

    search_reasons = detect_notes_search_signals(text)
    if search_reasons:
        return RouteDecision(
            mode="notes_search",
            confidence=0.85,
            reasons=search_reasons,
        )

    return RouteDecision(
        mode="general_chat",
        confidence=0.65,
        reasons=["defaulted to general chat"],
    )
