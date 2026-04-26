from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from notes_bot.config import load_config
from notes_bot.eval_candidates import write_eval_candidates


if __name__ == "__main__":
    cfg = load_config(ROOT / "config.yaml")
    output_path = ROOT / "data" / "eval_candidates.json"
    limit = 25
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1]).expanduser()
        if not output_path.is_absolute():
            output_path = (ROOT / output_path).resolve()
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])

    count = write_eval_candidates(
        log_path=cfg.data_dir / "search_queries.jsonl",
        output_path=output_path,
        existing_eval_path=ROOT / "data" / "eval_queries.json",
        limit=limit,
    )
    print(f"Wrote {count} eval candidate(s) to {output_path}")
