"""Vector database: persist chunk embeddings in ChromaDB and search them."""
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.errors import InvalidArgumentError

from src.chunking.chunker import Chunk

logger = logging.getLogger(__name__)


class IndexNotBuiltError(RuntimeError):
    """The index is empty, or was built with a different embedding model than the one in use."""


@dataclass
class RetrievedChunk:
    """A stored chunk returned by a search."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict
    score: float                        # cosine similarity to the query (higher is closer)
    rerank_score: float | None = None   # cross-encoder relevance, set when reranking is on


class VectorStore:
    """A ChromaDB collection of chunks, compared by cosine similarity."""

    def __init__(self, path: str | Path, collection: str):
        self.client = chromadb.PersistentClient(
            path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.name = collection
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        """Delete every stored chunk."""
        self.client.delete_collection(self.name)
        self.collection = self._open()

    def count(self) -> int:
        return self.collection.count()

    def ids(self) -> set[str]:
        """Ids of every stored chunk."""
        return set(self.collection.get(include=[])["ids"])

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Store chunks with their embeddings. Re-adding an existing chunk id overwrites it."""
        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings[start:start + batch_size],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
            )

    def search(self, query_embedding: np.ndarray, k: int, sources: list[str] | None = None) -> list[RetrievedChunk]:
        """Return the `k` most similar chunks, optionally restricted to some knowledge sources."""
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"source": {"$in": sources}} if sources else None,
                include=["documents", "metadatas", "distances"],
            )
        except InvalidArgumentError as e:
            if "dimension" not in str(e):
                raise
            # Happens after changing embeddings.model or embeddings.dimensions without re-indexing.
            raise IndexNotBuiltError(
                "The index was built with a different embedding model or size. "
                "Run `python main.py ingest` to rebuild it."
            ) from e
        return [
            RetrievedChunk(chunk_id, metadata["doc_id"], text, metadata, score=1.0 - distance)
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
