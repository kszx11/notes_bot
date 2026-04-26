from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from notes_bot.eval_candidates import load_eval_candidates, parse_candidate_selections, promote_eval_candidates


if __name__ == "__main__":
    candidate_path = ROOT / "data" / "eval_candidates.json"
    eval_path = ROOT / "data" / "eval_queries.json"

    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        candidate_path = Path(sys.argv[1]).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (ROOT / candidate_path).resolve()
        selection_args = sys.argv[2:]
    else:
        selection_args = sys.argv[1:]

    candidates = load_eval_candidates(candidate_path)
    if not candidates:
        print(f"No eval candidates found in {candidate_path}")
        raise SystemExit(0)

    selections = parse_candidate_selections(selection_args, len(candidates))
    promoted, remaining = promote_eval_candidates(
        candidate_path=candidate_path,
        eval_path=eval_path,
        selections=selections,
    )
    print(f"Promoted {promoted} candidate(s) into {eval_path}")
    print(f"Remaining candidates: {remaining}")
