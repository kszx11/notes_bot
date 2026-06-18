from dataclasses import dataclass
from pathlib import Path
import yaml

from .doc_roots import DocRoot, normalize_doc_roots

@dataclass(frozen=True)
class Config:
    doc_roots: tuple[DocRoot, ...]
    include_ext: tuple

    data_dir: Path
    index_dir: Path
    manifest_path: Path
    chat_history_path: Path

    chunk_chars: int
    chunk_overlap: int
    top_k: int

    scan_interval_minutes: int
    background_index_start_delay_seconds: int
    retry_error_after_minutes: int

    embedding_model: str
    chat_model: str

    max_history_turns: int
    max_sources_chars: int
    answer_context_before_lines: int
    answer_context_after_lines: int
    adjacent_chunk_window: int
    max_file_size_mb: int
    max_chunks_per_file: int

    @property
    def doc_root(self) -> Path:
        return self.doc_roots[0].path

def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    def _cfg_path(key: str) -> Path:
        p = Path(raw[key]).expanduser()
        return p if p.is_absolute() else (base_dir / p)

    data_dir = _cfg_path("data_dir")
    raw_doc_roots = raw.get("doc_roots")
    if raw_doc_roots:
        doc_roots = normalize_doc_roots(raw_doc_roots)
    else:
        doc_roots = normalize_doc_roots([raw.get("doc_root", "./notes")])
    resolved_doc_roots = tuple(
        DocRoot(
            name=root.name,
            path=root.path if root.path.is_absolute() else (base_dir / root.path),
        )
        for root in doc_roots
    )
    return Config(
        doc_roots=resolved_doc_roots,
        include_ext=tuple(raw.get("include_ext", [".md", ".txt"])),

        data_dir=data_dir,
        index_dir=_cfg_path("index_dir"),
        manifest_path=_cfg_path("manifest_path"),
        chat_history_path=_cfg_path("chat_history_path"),

        chunk_chars=int(raw.get("chunk_chars", 4000)),
        chunk_overlap=int(raw.get("chunk_overlap", 500)),
        top_k=int(raw.get("top_k", 10)),

        scan_interval_minutes=int(raw.get("scan_interval_minutes", 15)),
        background_index_start_delay_seconds=int(raw.get("background_index_start_delay_seconds", 300)),
        retry_error_after_minutes=int(raw.get("retry_error_after_minutes", 60)),

        embedding_model=str(raw.get("embedding_model", "text-embedding-3-small")),
        chat_model=str(raw.get("chat_model", "gpt-4.1-mini")),

        max_history_turns=int(raw.get("max_history_turns", 12)),
        max_sources_chars=int(raw.get("max_sources_chars", 35000)),
        answer_context_before_lines=int(raw.get("answer_context_before_lines", 8)),
        answer_context_after_lines=int(raw.get("answer_context_after_lines", 8)),
        adjacent_chunk_window=int(raw.get("adjacent_chunk_window", 1)),
        max_file_size_mb=int(raw.get("max_file_size_mb", 8)),
        max_chunks_per_file=int(raw.get("max_chunks_per_file", 2000)),
    )
    
