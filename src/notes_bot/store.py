from __future__ import annotations

from collections import Counter
import json
import math
import re
import sqlite3
from pathlib import Path


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if tok.strip()]


def _keyword_overlap(query: str, doc: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    d = Counter(_tokens(doc))
    if not d:
        return 0.0

    total = 0
    for tok, freq in Counter(q).items():
        total += min(freq, d.get(tok, 0))
    return total / max(1, len(q))


class VectorStore:
    def __init__(self, index_dir: Path, collection_name: str = "notes"):
        index_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = index_dir
        self.collection_name = collection_name
        self._sqlite_path = index_dir / f"{collection_name}.sqlite3"
        self._fts_enabled = True
        self._init_sqlite()

    def _connect(self):
        con = sqlite3.connect(self._sqlite_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_sqlite(self) -> None:
        with self._connect() as con:
            self._ensure_chunk_schema(con)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(rel_path, chunk_index)"
            )
            try:
                con.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        rel_path,
                        title,
                        section_path,
                        document,
                        tokenize = 'unicode61'
                    )
                    """
                )
            except sqlite3.OperationalError:
                self._fts_enabled = False
            con.commit()

    def _ensure_chunk_schema(self, con: sqlite3.Connection) -> None:
        required = {
            "id",
            "rel_path",
            "start_line",
            "end_line",
            "chunk_index",
            "content_type",
            "title",
            "section_path",
            "mtime",
            "document",
            "embedding",
        }
        existing = con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'chunks'"
        ).fetchone()
        if existing:
            cols = {
                row["name"]
                for row in con.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if not required.issubset(cols):
                con.execute("DROP TABLE IF EXISTS chunks")
                con.execute("DROP TABLE IF EXISTS chunks_fts")

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                rel_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                title TEXT NOT NULL,
                section_path TEXT NOT NULL,
                mtime REAL NOT NULL,
                document TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 1.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))

    @staticmethod
    def _fts_query(query_text: str) -> str:
        toks = _tokens(query_text)
        if not toks:
            return ""
        return " AND ".join(f'"{tok}"' for tok in toks[:12])

    def delete_file(self, rel_path: str) -> None:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id FROM chunks WHERE rel_path = ?",
                (rel_path,),
            ).fetchall()
            chunk_ids = [row["id"] for row in rows]
            con.execute("DELETE FROM chunks WHERE rel_path = ?", (rel_path,))
            if self._fts_enabled and chunk_ids:
                con.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    [(chunk_id,) for chunk_id in chunk_ids],
                )
            con.commit()

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._connect() as con:
            placeholders = ",".join("?" for _ in ids)
            con.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", tuple(ids))
            if self._fts_enabled:
                con.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    [(chunk_id,) for chunk_id in ids],
                )
            con.commit()

    def delete_file_except(self, rel_path: str, keep_ids: list[str]) -> None:
        if not keep_ids:
            self.delete_file(rel_path)
            return
        with self._connect() as con:
            placeholders = ",".join("?" for _ in keep_ids)
            rows = con.execute(
                f"""
                SELECT id FROM chunks
                WHERE rel_path = ? AND id NOT IN ({placeholders})
                """,
                tuple([rel_path] + keep_ids),
            ).fetchall()
            delete_ids = [row["id"] for row in rows]
            if delete_ids:
                placeholders_delete = ",".join("?" for _ in delete_ids)
                con.execute(
                    f"DELETE FROM chunks WHERE id IN ({placeholders_delete})",
                    tuple(delete_ids),
                )
                if self._fts_enabled:
                    con.executemany(
                        "DELETE FROM chunks_fts WHERE chunk_id = ?",
                        [(chunk_id,) for chunk_id in delete_ids],
                    )
            con.commit()

    def add_chunks(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        rows = [
            (
                chunk_id,
                metadata["rel_path"],
                int(metadata["start_line"]),
                int(metadata["end_line"]),
                int(metadata["chunk_index"]),
                str(metadata.get("content_type", "prose")),
                str(metadata.get("title", "")),
                str(metadata.get("section_path", "")),
                float(metadata["mtime"]),
                text,
                json.dumps(embedding or []),
            )
            for chunk_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
        ]

        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO chunks(
                    id, rel_path, start_line, end_line, chunk_index,
                    content_type, title, section_path, mtime, document, embedding
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    rel_path = excluded.rel_path,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    chunk_index = excluded.chunk_index,
                    content_type = excluded.content_type,
                    title = excluded.title,
                    section_path = excluded.section_path,
                    mtime = excluded.mtime,
                    document = excluded.document,
                    embedding = excluded.embedding
                """,
                rows,
            )
            if self._fts_enabled:
                con.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    [(chunk_id,) for chunk_id in ids],
                )
                con.executemany(
                    """
                    INSERT INTO chunks_fts(chunk_id, rel_path, title, section_path, document)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk_id,
                            metadata["rel_path"],
                            str(metadata.get("title", "")),
                            str(metadata.get("section_path", "")),
                            text,
                        )
                        for chunk_id, text, metadata in zip(ids, texts, metadatas)
                    ],
                )
            con.commit()

    def get_chunk_window(self, rel_path: str, center_chunk_index: int, window_size: int = 1) -> list[dict]:
        window_size = max(0, int(window_size))
        start_chunk = max(0, int(center_chunk_index) - window_size)
        end_chunk = int(center_chunk_index) + window_size
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT
                    id, rel_path, start_line, end_line, chunk_index,
                    content_type, title, section_path, mtime, document
                FROM chunks
                WHERE rel_path = ?
                  AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index ASC
                """,
                (rel_path, start_chunk, end_chunk),
            ).fetchall()
        return [dict(row) for row in rows]

    def _semantic_candidates(
        self,
        query_embedding: list[float] | None,
        limit: int,
        rel_path_prefix: str | None = None,
    ) -> list[dict]:
        if not query_embedding:
            return []

        with self._connect() as con:
            if rel_path_prefix:
                rows = con.execute(
                    """
                    SELECT
                        id, rel_path, start_line, end_line, chunk_index,
                        content_type, title, section_path, mtime, document, embedding
                    FROM chunks
                    WHERE rel_path LIKE ?
                    """,
                    (f"{rel_path_prefix}%",),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT
                        id, rel_path, start_line, end_line, chunk_index,
                        content_type, title, section_path, mtime, document, embedding
                    FROM chunks
                    """
                ).fetchall()

        scored: list[dict] = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            if not embedding:
                continue
            distance = self._cosine_distance(query_embedding, embedding)
            scored.append({"id": row["id"], "distance": distance})
        scored.sort(key=lambda item: item["distance"])
        return scored[:limit]

    def _lexical_candidates(
        self,
        query_text: str,
        limit: int,
        rel_path_prefix: str | None = None,
    ) -> list[dict]:
        toks = _tokens(query_text)
        low = query_text.lower()
        candidates: dict[str, dict] = {}

        with self._connect() as con:
            if self._fts_enabled and toks:
                try:
                    if rel_path_prefix:
                        rows = con.execute(
                            """
                            SELECT chunk_id, bm25(chunks_fts) AS fts_rank
                            FROM chunks_fts
                            WHERE chunks_fts MATCH ?
                              AND rel_path LIKE ?
                            ORDER BY fts_rank
                            LIMIT ?
                            """,
                            (self._fts_query(query_text), f"{rel_path_prefix}%", limit),
                        ).fetchall()
                    else:
                        rows = con.execute(
                            """
                            SELECT chunk_id, bm25(chunks_fts) AS fts_rank
                            FROM chunks_fts
                            WHERE chunks_fts MATCH ?
                            ORDER BY fts_rank
                            LIMIT ?
                            """,
                            (self._fts_query(query_text), limit),
                        ).fetchall()
                    for rank, row in enumerate(rows):
                        candidates[row["chunk_id"]] = {
                            "id": row["chunk_id"],
                            "fts_rank": float(row["fts_rank"]),
                            "fts_pos": rank,
                        }
                except sqlite3.OperationalError:
                    self._fts_enabled = False

            like = f"%{low}%"
            if rel_path_prefix:
                rows = con.execute(
                    """
                    SELECT id
                    FROM chunks
                    WHERE (
                        lower(rel_path) LIKE ?
                        OR lower(title) LIKE ?
                        OR lower(section_path) LIKE ?
                    )
                      AND rel_path LIKE ?
                    LIMIT ?
                    """,
                    (like, like, like, f"{rel_path_prefix}%", limit),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT id
                    FROM chunks
                    WHERE lower(rel_path) LIKE ?
                       OR lower(title) LIKE ?
                       OR lower(section_path) LIKE ?
                    LIMIT ?
                    """,
                    (like, like, like, limit),
                ).fetchall()
            for row in rows:
                entry = candidates.setdefault(row["id"], {"id": row["id"]})
                entry["meta_match"] = True

        return list(candidates.values())[:limit]

    def query(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        top_k: int,
        rel_path_prefix: str | None = None,
    ):
        top_k = max(1, top_k)
        semantic = self._semantic_candidates(
            query_embedding,
            limit=max(top_k * 6, 24),
            rel_path_prefix=rel_path_prefix,
        )
        lexical = self._lexical_candidates(
            query_text,
            limit=max(top_k * 6, 24),
            rel_path_prefix=rel_path_prefix,
        )

        semantic_rank = {item["id"]: idx for idx, item in enumerate(semantic)}
        lexical_rank = {item["id"]: idx for idx, item in enumerate(lexical)}

        candidate_ids: set[str] = set(semantic_rank) | set(lexical_rank)
        if not candidate_ids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        with self._connect() as con:
            placeholders = ",".join("?" for _ in candidate_ids)
            rows = con.execute(
                f"""
                SELECT
                    id, rel_path, start_line, end_line, chunk_index,
                    content_type, title, section_path, mtime, document, embedding
                FROM chunks
                WHERE id IN ({placeholders})
                """,
                tuple(candidate_ids),
            ).fetchall()

        rows_by_id = {row["id"]: row for row in rows}
        file_counts: Counter[str] = Counter()
        ranked: list[tuple[float, sqlite3.Row, float]] = []

        for chunk_id in candidate_ids:
            row = rows_by_id.get(chunk_id)
            if row is None:
                continue

            semantic_pos = semantic_rank.get(chunk_id)
            lexical_pos = lexical_rank.get(chunk_id)
            embedding = json.loads(row["embedding"])
            distance = self._cosine_distance(query_embedding, embedding) if query_embedding and embedding else 1.0
            overlap = _keyword_overlap(query_text, row["document"])

            score = 0.0
            if semantic_pos is not None:
                score += 0.45 * (1.0 - (semantic_pos / max(1, len(semantic))))
            if lexical_pos is not None:
                score += 0.35 * (1.0 - (lexical_pos / max(1, len(lexical))))
            score += 0.12 * overlap

            low_query = query_text.lower()
            low_path = row["rel_path"].lower()
            low_title = row["title"].lower()
            low_section = row["section_path"].lower()
            if low_query and low_query in low_path:
                score += 0.08
            if low_query and (low_query in low_title or low_query in low_section):
                score += 0.06

            ranked.append((score, row, distance))

        ranked.sort(key=lambda item: (item[0], -item[2]), reverse=True)

        selected: list[tuple[sqlite3.Row, float]] = []
        for score, row, distance in ranked:
            if file_counts[row["rel_path"]] >= 3:
                continue
            file_counts[row["rel_path"]] += 1
            selected.append((row, distance))
            if len(selected) >= top_k:
                break

        return {
            "ids": [[row["id"] for row, _ in selected]],
            "documents": [[row["document"] for row, _ in selected]],
            "metadatas": [[
                {
                    "rel_path": row["rel_path"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "chunk_index": row["chunk_index"],
                    "mtime": row["mtime"],
                    "content_type": row["content_type"],
                    "title": row["title"],
                    "section_path": row["section_path"],
                }
                for row, _ in selected
            ]],
            "distances": [[distance for _, distance in selected]],
        }

    def search_files(
        self,
        query_text: str,
        mode: str = "both",
        limit: int = 100,
        rel_path_prefix: str | None = None,
    ) -> list[dict]:
        limit = max(1, limit)
        needle = query_text.strip().lower()
        if not needle:
            return []

        with self._connect() as con:
            like = f"%{needle}%"
            params: list[object] = []
            clauses: list[str] = []

            if mode in ("filename", "both"):
                clauses.append("lower(rel_path) LIKE ?")
                params.append(like)
            if mode in ("text", "both"):
                clauses.extend([
                    "lower(document) LIKE ?",
                    "lower(title) LIKE ?",
                    "lower(section_path) LIKE ?",
                ])
                params.extend([like, like, like])

            where = " OR ".join(clauses) if clauses else "1=0"
            if rel_path_prefix:
                where = f"({where}) AND rel_path LIKE ?"
                params.append(f"{rel_path_prefix}%")
            rows = con.execute(
                f"""
                SELECT
                    rel_path,
                    MIN(title) AS best_title,
                    MIN(section_path) AS best_section_path,
                    SUM(CASE WHEN lower(rel_path) LIKE ? THEN 1 ELSE 0 END) AS filename_hits,
                    SUM(CASE WHEN lower(document) LIKE ? OR lower(title) LIKE ? OR lower(section_path) LIKE ? THEN 1 ELSE 0 END) AS text_hits,
                    COUNT(*) AS section_hits,
                    MIN(start_line) AS min_start_line,
                    MAX(end_line) AS max_end_line
                FROM chunks
                WHERE {where}
                GROUP BY rel_path
                ORDER BY
                    filename_hits DESC,
                    text_hits DESC,
                    section_hits DESC,
                    rel_path ASC
                LIMIT ?
                """,
                tuple([like, like, like, like] + params + [limit]),
            ).fetchall()

        out: list[dict] = []
        for row in rows:
            filename_match = int(row["filename_hits"] or 0) > 0
            text_match = int(row["text_hits"] or 0) > 0
            if mode == "filename" and not filename_match:
                continue
            if mode == "text" and not text_match:
                continue
            out.append(
                {
                    "rel_path": row["rel_path"],
                    "filename_match": filename_match,
                    "text_match": text_match,
                    "section_hits": int(row["section_hits"] or 0),
                    "best_title": row["best_title"] or "",
                    "best_section_path": row["best_section_path"] or "",
                    "start_line": row["min_start_line"],
                    "end_line": row["max_end_line"],
                }
            )
        return out
