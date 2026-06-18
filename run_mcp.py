from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from notes_bot.mcp_server import main
except ModuleNotFoundError as exc:
    from notes_bot.runtime import format_missing_dependency_error

    print(format_missing_dependency_error(exc, ROOT), file=sys.stderr)
    raise SystemExit(1) from exc

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(ROOT / "config.yaml")])
    main()
