from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocRoot:
    name: str
    path: Path


def _sanitize_root_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return cleaned or "notes"


def _derive_root_name(path: Path) -> str:
    parts = [part for part in path.parts if part not in ("/", "\\")]
    if parts:
        return _sanitize_root_name(parts[-1])
    return "notes"


def normalize_doc_roots(items: Iterable[str | Path | dict]) -> tuple[DocRoot, ...]:
    raw_items = list(items)
    if not raw_items:
        raise ValueError("At least one doc root must be configured")

    normalized: list[DocRoot] = []
    seen_names: dict[str, int] = {}
    for item in raw_items:
        if isinstance(item, dict):
            path_value = item.get("path")
            if not path_value:
                raise ValueError("Each doc_roots entry must include a path")
            base_name = str(item.get("name") or _derive_root_name(Path(path_value)))
            path = Path(path_value).expanduser()
        else:
            path = Path(item).expanduser()
            base_name = _derive_root_name(path)

        name = _sanitize_root_name(base_name)
        suffix = seen_names.get(name, 0)
        seen_names[name] = suffix + 1
        if suffix:
            name = f"{name}-{suffix + 1}"

        normalized.append(DocRoot(name=name, path=path))

    return tuple(normalized)


def has_multiple_roots(doc_roots: Iterable[DocRoot]) -> bool:
    return len(tuple(doc_roots)) > 1


def to_virtual_rel_path(doc_roots: tuple[DocRoot, ...], root_name: str, rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").strip("/")
    if len(doc_roots) <= 1:
        return rel
    return f"{root_name}/{rel}"


def split_virtual_rel_path(doc_roots: tuple[DocRoot, ...], rel_path: str) -> tuple[DocRoot, str]:
    clean = rel_path.replace("\\", "/").strip("/")
    if not clean:
        raise ValueError("Empty rel_path")

    if len(doc_roots) == 1:
        return doc_roots[0], clean

    prefix, sep, remainder = clean.partition("/")
    if not sep or not remainder:
        raise ValueError(
            "Invalid rel_path for multi-root configuration. Expected '<root>/<path>'."
        )

    for root in doc_roots:
        if root.name == prefix:
            return root, remainder
    raise ValueError(f"Unknown doc root prefix: {prefix}")


def resolve_doc_path(doc_roots: tuple[DocRoot, ...], rel_path: str) -> tuple[DocRoot, str, Path]:
    root, inner_rel_path = split_virtual_rel_path(doc_roots, rel_path)
    root_path = root.path.resolve()
    candidate = (root_path / inner_rel_path).resolve()
    if os.path.commonpath([str(root_path), str(candidate)]) != str(root_path):
        raise ValueError("Invalid rel_path outside doc_root")
    return root, inner_rel_path, candidate
