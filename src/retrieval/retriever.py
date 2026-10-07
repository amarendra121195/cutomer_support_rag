"""Retrieval: find the chunks most relevant to a question."""
import logging

from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder that reads the question and a chunk together and scores their relevance.

    Slower than comparing embeddings, so it is only run on the shortlist that the vector
    search returns. It is a local PyTorch model: no API cost, but it needs
    sentence-transformers installed and about a second of CPU per question.
    """

    def __init__(self, model_name: str, num_threads: int | None = None):
        import torch  # heavy imports, deferred until reranking is actually switched on
        from sentence_transformers import CrossEncoder

        if num_threads:
            torch.set_num_threads(num_threads)
        logger.info("Loading reranker %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        scores = self.model.predict([(query, f"{chunk.metadata['title']}\n{chunk.text}") for chunk in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda chunk: chunk.rerank_score, reverse=True)


class Retriever:
    """Vector search over the store, optionally followed by cross-encoder reranking."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        fetch_k: int = 30,
        reranker: Reranker | None = None,
        sources: list[str] | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.reranker = reranker
        self.sources = sources

    def retrieve(self, query: str, top_k: int | None = None, sources: list[str] | None = None) -> list[RetrievedChunk]:
        """Return the best `top_k` chunks for `query`, most relevant first."""
        top_k = top_k or self.top_k
        candidates = max(top_k, self.fetch_k) if self.reranker else top_k
        chunks = self.store.search(self.embedder.embed_query(query), candidates, sources or self.sources)
        if self.reranker and chunks:
            chunks = self.reranker.rerank(query, chunks)
        return chunks[:top_k]
