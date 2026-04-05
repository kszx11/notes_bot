import re
from .prompt import CITATION_RE

def validate_structured_answer(text: str, allowed: set[str]) -> tuple[bool, str]:
    """
    Returns (ok, reason).
    Requirements:
      - Has 'Answer:' and 'Evidence:' sections
      - Evidence has at least 1 bullet with a quoted string and a valid citation
      - All citations used are in the allowed set (prevents made-up citations)
    """
    if "Answer:" not in text or "Evidence:" not in text:
        return False, "Missing required sections."

    # Split evidence lines
    parts = text.split("Evidence:", 1)
    if len(parts) != 2:
        return False, "Malformed Evidence section."

    evidence = parts[1].strip()
    lines = [ln.strip() for ln in evidence.splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if ln.startswith("-")]

    if not bullet_lines:
        return False, "No evidence bullets."

    seen_valid_citation = False
    for ln in bullet_lines:
        # Must contain a quoted segment
        if '"' not in ln:
            return False, "Evidence bullet missing quotes."

        m = CITATION_RE.search(ln)
        if not m:
            return False, "Evidence bullet missing valid citation format."

        rel_path, a, b = m.group(1), m.group(2), m.group(3)
        key = f"{rel_path}:{a}-{b}"
        if key not in allowed:
            return False, f"Citation not in provided sources: {key}"
        seen_valid_citation = True

    if not seen_valid_citation:
        return False, "No valid citations found."

    return True, "OK"
    
