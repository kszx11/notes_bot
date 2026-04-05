from pathlib import Path
import chromadb

class VectorStore:
    def __init__(self, index_dir: Path, collection_name: str = "notes"):
        index_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(index_dir))
        self.col = self.client.get_or_create_collection(name=collection_name)

    def delete_file(self, rel_path: str) -> None:
        # delete chunks with metadata match
        self.col.delete(where={"rel_path": rel_path})

    def add_chunks(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        self.col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int):
        return self.col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        )

        
