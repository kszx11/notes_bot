from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass(frozen=True)
class DiscoveredFile:
    abs_path: Path
    rel_path: str
    mtime: float
    size: int

def iter_files(doc_root: Path, include_ext: tuple[str, ...]) -> Iterable[DiscoveredFile]:
    doc_root = doc_root.resolve()
    for p in doc_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in include_ext:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(doc_root).as_posix()
        yield DiscoveredFile(abs_path=p, rel_path=rel, mtime=st.st_mtime, size=st.st_size)
        
