import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from notes_bot.chat import main
except ModuleNotFoundError as exc:
    from notes_bot.runtime import format_missing_dependency_error

    print(format_missing_dependency_error(exc, ROOT), file=sys.stderr)
    raise SystemExit(1) from exc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="notes_bot chat CLI")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config YAML")
    parser.add_argument(
        "--no-background-index",
        action="store_true",
        help="Disable periodic background indexing loop",
    )
    args = parser.parse_args()
    main(config_path=args.config, enable_background=not args.no_background_index)
