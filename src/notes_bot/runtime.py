from __future__ import annotations

import os
from pathlib import Path

from .config import Config


def validate_runtime_config(cfg: Config) -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before starting the chat."
        )

    missing: list[str] = []
    invalid: list[str] = []
    for root in cfg.doc_roots:
        if root.path.exists() and not root.path.is_dir():
            invalid.append(f"{root.name}: {root.path}")
        elif not root.path.exists():
            missing.append(f"{root.name}: {root.path}")

    if invalid:
        raise RuntimeError(
            "Configured doc root is not a directory:\n" + "\n".join(invalid)
        )

    if missing:
        raise RuntimeError(
            "Configured doc root does not exist:\n"
            + "\n".join(missing)
            + "\nCreate those directories or update config.yaml to point at your notes folders."
        )


def format_missing_dependency_error(exc: ModuleNotFoundError, root: Path) -> str:
    missing = exc.name or "a required package"
    req_path = root / "requirements.txt"
    return (
        f"Missing dependency: {missing}\n"
        f"Install the project requirements first, for example:\n"
        f"  python3 -m pip install -r {req_path}"
    )
