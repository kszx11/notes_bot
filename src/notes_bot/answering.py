from __future__ import annotations

import re
import textwrap

from openai import OpenAI


SYSTEM_PROMPT = """You answer questions using only the provided note excerpts.

Rules:
- Use only the provided sources.
- If the sources do not support an answer, reply exactly: I can't find that in your notes.
- Keep the answer concise, at most 3 short sentences.
- Do not include citations, bullet points, section headers, or discussion of the retrieval process.
"""


def build_question_prompt(question: str, sources_text: str) -> str:
    return f"SOURCES:\n{sources_text}\n\nQUESTION:\n{question}"


def synthesize_answer(
    *,
    client: OpenAI,
    model: str,
    question: str,
    sources_text: str,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_question_prompt(question, sources_text)},
        ],
        temperature=0.1,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return "I can't find that in your notes."
    return text


def _clean_quote(text: str, max_chars: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _wrap_block(text: str, *, width: int = 92, initial_indent: str = "", subsequent_indent: str = "") -> str:
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        return ""
    return "\n".join(
        textwrap.fill(
            part,
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for part in parts
    )


def format_grounded_answer(
    answer: str,
    used_sources: list[dict],
    source_texts: list[str],
    answer_note: str = "",
) -> str:
    answer = answer.strip() or "I can't find that in your notes."
    if answer == "I can't find that in your notes." or not used_sources or not source_texts:
        return "I can't find that in your notes."

    evidence_lines: list[str] = []
    max_evidence = min(4, len(used_sources), len(source_texts))
    for idx, (meta, text) in enumerate(zip(used_sources[:max_evidence], source_texts[:max_evidence]), start=1):
        quote = _clean_quote(text)
        if not quote:
            continue
        location = f'{meta["rel_path"]}:{meta["start_line"]}-{meta["end_line"]}'
        evidence_lines.append(
            f"{idx}. {location}\n"
            + _wrap_block(f'"{quote}"', initial_indent="   ", subsequent_indent="   ")
        )

    if not evidence_lines:
        return "I can't find that in your notes."

    answer_block = _wrap_block(answer, initial_indent="  ", subsequent_indent="  ")
    extra_note_block = _wrap_block(answer_note.strip(), initial_indent="  ", subsequent_indent="  ") if answer_note else ""
    if answer_note:
        answer_block += "\n\nNote:\n" + extra_note_block
    return "Answer:\n" + answer_block + "\n\nEvidence:\n" + "\n\n".join(evidence_lines)
