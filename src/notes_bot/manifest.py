from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEARCHABLE_STATUSES = ("indexed", "stale")


@dataclass
class FileState:
    rel_path: str
    mtime: float
    size: int
    status: str = "indexed"
    last_error: str = ""
    updated_at: float = 0.0
    chunk_count: int = 0
    last_indexed_at: float = 0.0
    last_success_at: float = 0.0


class Manifest:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init(self):
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    rel_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'indexed',
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    last_indexed_at REAL NOT NULL DEFAULT 0,
                    last_success_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            cols = {row["name"] for row in con.execute("PRAGMA table_info(files)").fetchall()}
            if "status" not in cols:
                con.execute("ALTER TABLE files ADD COLUMN status TEXT NOT NULL DEFAULT 'indexed'")
            if "last_error" not in cols:
                con.execute("ALTER TABLE files ADD COLUMN last_error TEXT NOT NULL DEFAULT ''")
            if "updated_at" not in cols:
                con.execute("ALTER TABLE files ADD COLUMN updated_at REAL NOT NULL DEFAULT 0")
            if "chunk_count" not in cols:
                con.execute("ALTER TABLE files ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0")
            if "last_indexed_at" not in cols:
                con.execute("ALTER TABLE files ADD COLUMN last_indexed_at REAL NOT NULL DEFAULT 0")
            if "last_success_at" not in cols:
                con.execute("ALTER TABLE files ADD COLUMN last_success_at REAL NOT NULL DEFAULT 0")
            con.commit()

    def get(self, rel_path: str) -> FileState | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT
                    rel_path, mtime, size, status, last_error, updated_at,
                    chunk_count, last_indexed_at, last_success_at
                FROM files
                WHERE rel_path = ?
                """,
                (rel_path,),
            ).fetchone()
        if not row:
            return None
        return FileState(**dict(row))

    def upsert(self, st: FileState) -> None:
        updated_at = st.updated_at or time.time()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO files(
                    rel_path, mtime, size, status, last_error, updated_at,
                    chunk_count, last_indexed_at, last_success_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(rel_path) DO UPDATE SET
                    mtime=excluded.mtime,
                    size=excluded.size,
                    status=excluded.status,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at,
                    chunk_count=excluded.chunk_count,
                    last_indexed_at=excluded.last_indexed_at,
                    last_success_at=excluded.last_success_at
                """,
                (
                    st.rel_path,
                    st.mtime,
                    st.size,
                    st.status,
                    st.last_error,
                    updated_at,
                    st.chunk_count,
                    st.last_indexed_at,
                    st.last_success_at,
                ),
            )
            con.commit()

    def mark_status(
        self,
        *,
        rel_path: str,
        mtime: float,
        size: int,
        status: str,
        last_error: str = "",
        chunk_count: int = 0,
        last_indexed_at: float | None = None,
        last_success_at: float | None = None,
    ) -> None:
        now = time.time()
        self.upsert(
            FileState(
                rel_path=rel_path,
                mtime=mtime,
                size=size,
                status=status,
                last_error=last_error,
                updated_at=now,
                chunk_count=chunk_count,
                last_indexed_at=last_indexed_at if last_indexed_at is not None else now,
                last_success_at=last_success_at or 0.0,
            )
        )

    def delete(self, rel_path: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))
            con.commit()

    def all_paths(self, statuses: tuple[str, ...] = SEARCHABLE_STATUSES) -> set[str]:
        with self._connect() as con:
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                rows = con.execute(
                    f"SELECT rel_path FROM files WHERE status IN ({placeholders})",
                    tuple(statuses),
                ).fetchall()
            else:
                rows = con.execute("SELECT rel_path FROM files").fetchall()
        return {r["rel_path"] for r in rows}

    def iter_all(self) -> Iterable[FileState]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT
                    rel_path, mtime, size, status, last_error, updated_at,
                    chunk_count, last_indexed_at, last_success_at
                FROM files
                """
            ).fetchall()
        for row in rows:
            yield FileState(**dict(row))

    def counts_by_status(self) -> dict[str, int]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT status, COUNT(*) AS n FROM files GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}
