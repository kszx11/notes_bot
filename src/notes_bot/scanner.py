from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .doc_roots import DocRoot, to_virtual_rel_path

@dataclass(frozen=True)
class DiscoveredFile:
    root_name: str
    abs_path: Path
    rel_path: str
    mtime: float
    size: int

def iter_files(doc_roots: tuple[DocRoot, ...], include_ext: tuple[str, ...]) -> Iterable[DiscoveredFile]:
    for root in doc_roots:
        root_path = root.path.resolve()
        for p in root_path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in include_ext:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            rel = p.relative_to(root_path).as_posix()
            yield DiscoveredFile(
                root_name=root.name,
                abs_path=p,
                rel_path=to_virtual_rel_path(doc_roots, root.name, rel),
                mtime=st.st_mtime,
                size=st.st_size,
            )
        
