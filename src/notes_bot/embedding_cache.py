from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from openai import OpenAI


class QueryEmbeddingCache:
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

    def _init(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS query_embeddings (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            con.commit()

    def _cache_key(self, model: str, text: str) -> str:
        raw = f"{model}\n{text}".encode("utf-8", errors="ignore")
        return hashlib.sha256(raw).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        key = self._cache_key(model, text)
        with self._connect() as con:
            row = con.execute(
                "SELECT embedding_json FROM query_embeddings WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return [float(x) for x in json.loads(row[0])]

    def put(self, model: str, text: str, embedding: list[float]) -> None:
        key = self._cache_key(model, text)
        payload = json.dumps(embedding, separators=(",", ":"))
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO query_embeddings(cache_key, model, query_text, embedding_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    embedding_json=excluded.embedding_json
                """,
                (key, model, text, payload),
            )
            con.commit()


def get_query_embedding(
    *,
    client: OpenAI,
    model: str,
    text: str,
    cache: QueryEmbeddingCache | None = None,
) -> list[float]:
    if cache is not None:
        cached = cache.get(model, text)
        if cached is not None:
            return cached

    resp = client.embeddings.create(model=model, input=text)
    embedding = resp.data[0].embedding
    if cache is not None:
        cache.put(model, text, embedding)
    return embedding
