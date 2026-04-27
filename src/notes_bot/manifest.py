import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass
class FileState:
    rel_path: str
    mtime: float
    size: int

class Manifest:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
        finally:
            con.close()

    def _init(self):
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    rel_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            con.commit()

    def get(self, rel_path: str) -> FileState | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT rel_path, mtime, size FROM files WHERE rel_path = ?",
                (rel_path,)
            ).fetchone()
        if not row:
            return None
        return FileState(*row)

    def upsert(self, st: FileState) -> None:
        with self._connect() as con:
            con.execute("""
                INSERT INTO files(rel_path, mtime, size)
                VALUES(?,?,?)
                ON CONFLICT(rel_path) DO UPDATE SET
                    mtime=excluded.mtime,
                    size=excluded.size
            """, (st.rel_path, st.mtime, st.size))
            con.commit()

    def delete(self, rel_path: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))
            con.commit()

    def all_paths(self) -> set[str]:
        with self._connect() as con:
            rows = con.execute("SELECT rel_path FROM files").fetchall()
        return {r[0] for r in rows}

    def iter_all(self) -> Iterable[FileState]:
        with self._connect() as con:
            rows = con.execute("SELECT rel_path, mtime, size FROM files").fetchall()
        for r in rows:
            yield FileState(*r)

    def get_meta(self, key: str) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT value FROM meta WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as con:
            con.execute("""
                INSERT INTO meta(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value
            """, (key, value))
            con.commit()
            
