from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Config:
    doc_root: Path
    include_ext: tuple

    data_dir: Path
    index_dir: Path
    manifest_path: Path
    chat_history_path: Path

    chunk_chars: int
    chunk_overlap: int
    top_k: int

    scan_interval_minutes: int

    embedding_model: str
    chat_model: str

    max_history_turns: int
    max_sources_chars: int
    max_file_size_mb: int
    max_chunks_per_file: int

def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    def _cfg_path(key: str) -> Path:
        p = Path(raw[key]).expanduser()
        return p if p.is_absolute() else (base_dir / p)

    data_dir = _cfg_path("data_dir")
    return Config(
        doc_root=_cfg_path("doc_root"),
        include_ext=tuple(raw.get("include_ext", [".md", ".txt"])),

        data_dir=data_dir,
        index_dir=_cfg_path("index_dir"),
        manifest_path=_cfg_path("manifest_path"),
        chat_history_path=_cfg_path("chat_history_path"),

        chunk_chars=int(raw.get("chunk_chars", 4000)),
        chunk_overlap=int(raw.get("chunk_overlap", 500)),
        top_k=int(raw.get("top_k", 10)),

        scan_interval_minutes=int(raw.get("scan_interval_minutes", 15)),

        embedding_model=str(raw.get("embedding_model", "text-embedding-3-small")),
        chat_model=str(raw.get("chat_model", "gpt-4.1-mini")),

        max_history_turns=int(raw.get("max_history_turns", 12)),
        max_sources_chars=int(raw.get("max_sources_chars", 35000)),
        max_file_size_mb=int(raw.get("max_file_size_mb", 8)),
        max_chunks_per_file=int(raw.get("max_chunks_per_file", 2000)),
    )
    
