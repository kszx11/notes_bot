import re

GENERAL_CHAT_PROMPT = """You are a helpful assistant."""

SYSTEM_PROMPT = """You are a notes assistant. You must follow these rules:

GROUNDING
- Use ONLY the provided SOURCES. Do not use outside knowledge.
- If the answer is not explicitly supported by the sources, say exactly:
  "I can't find that in your notes."
  and output nothing else.

FORMAT (required)
- Output exactly two sections in this order:

Answer:
<1-3 short sentences. No bullet points. No citations in this section. No hedging.>

Evidence:
- "<verbatim quote 1>" (rel_path: start_line-end_line)
- "<verbatim quote 2>" (rel_path: start_line-end_line)

EVIDENCE RULES
- Evidence MUST be verbatim quotes copied from the sources.
- Every Evidence bullet must end with a citation in the exact form:
  (rel_path: start_line-end_line)
- Provide 2–6 evidence bullets when possible.
- Do not paraphrase unless the user explicitly asks to paraphrase.
"""

CITATION_RE = re.compile(r"\(([^:()]+):\s*(\d+)-(\d+)\)\s*$")

def build_sources_block(results, max_chars: int) -> tuple[str, list[dict]]:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[None]*len(docs)])[0]

    blocks = []
    used = 0
    used_sources = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        header = f"[SOURCE {i}] {meta['rel_path']}:{meta['start_line']}-{meta['end_line']}\n"
        body = doc.strip()
        block = header + body + "\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
        used_sources.append({**meta, "rank": i, "distance": dist})

    return ("\n".join(blocks).strip(), used_sources)

def allowed_citation_set(used_sources: list[dict]) -> set[str]:
    """
    Create a set of allowed citation strings like:
      "path/to/file.md:12-34"
    """
    out = set()
    for m in used_sources:
        out.add(f"{m['rel_path']}:{m['start_line']}-{m['end_line']}")
    return out
    
