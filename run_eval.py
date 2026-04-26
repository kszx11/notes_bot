from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from notes_bot.eval_runner import run_eval

if __name__ == "__main__":
    eval_path = ROOT / "data" / "eval_queries.json"
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1]).expanduser()
        if not eval_path.is_absolute():
            eval_path = (ROOT / eval_path).resolve()
    limit = 8
    stream = False
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    if len(sys.argv) > 3:
        stream = sys.argv[3].lower() in ("1", "true", "yes", "stream", "--stream")
    if stream:
        from notes_bot.eval_runner import stream_eval
        stream_eval(config_path=ROOT / "config.yaml", eval_path=eval_path, limit=limit)
    else:
        print(run_eval(config_path=ROOT / "config.yaml", eval_path=eval_path, limit=limit))
