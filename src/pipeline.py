"""The RAG pipeline: wires ingestion, chunking, embedding, storage, retrieval and generation together."""
import logging
import time
from dataclasses import dataclass
from functools import cached_property

from tqdm import tqdm

from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import Embedder
from src.ingestion.loader import load_documents
from src.llm.llm_client import LLMClient, LLMResponse
from src.prompts.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.retriever import Reranker, Retriever
from src.utils.helpers import batched, resolve_path
from src.vectordb.vector_store import IndexNotBuiltError, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "The model declined to answer this question."
INDEX_BATCH = 2048      # chunks embedded and written to the vector store at a time


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    response: LLMResponse           # model, status and token usage of the generation call


class RAGPipeline:
    """Builds the index and answers questions from it.

    Components are created on first use, so tests can pass in fakes and a command only
    needs the parts it actually uses.
    """

    def __init__(self, config: dict, embedder=None, store=None, llm=None):
        self.config = config
        for name, component in (("embedder", embedder), ("store", store), ("llm", llm)):
            if component is not None:
                setattr(self, name, component)

    @cached_property
    def embedder(self) -> Embedder:
        cfg = self.config["embeddings"]
        return Embedder(cfg["model"], cfg["batch_size"], cfg["dimensions"])

    @cached_property
    def store(self) -> VectorStore:
        cfg = self.config["vectordb"]
        return VectorStore(resolve_path(cfg["path"]), cfg["collection"])

    @cached_property
    def retriever(self) -> Retriever:
        cfg = self.config["retrieval"]
        rerank = cfg["rerank"]
        reranker = Reranker(rerank["model"], rerank["num_threads"]) if rerank["enabled"] else None
        return Retriever(self.embedder, self.store, cfg["top_k"], cfg["fetch_k"], reranker, cfg["sources"])

    @cached_property
    def llm(self) -> LLMClient:
        cfg = self.config["llm"]
        return LLMClient(cfg["model"], cfg["max_output_tokens"], cfg["effort"])

    def build_index(self, resume: bool = False) -> dict:
        """Load, chunk, embed and store every knowledge source.

        By default the existing index is replaced. With `resume=True` chunks that are already
        stored are kept and only the missing ones are embedded, which finishes an interrupted
        run without paying for the same embeddings twice.
        """
        started = time.perf_counter()
        embedder = self.embedder    # created first, so a missing API key stops the run before the slow loading
        documents = load_documents(self.config)
        cfg = self.config["chunking"]
        chunks = chunk_documents(documents, cfg["chunk_size"], cfg["chunk_overlap"])

        if resume:
            stored = self.store.ids()
            pending = [chunk for chunk in chunks if chunk.chunk_id not in stored]
        else:
            self.store.reset()
            pending = chunks
        logger.info("Embedding %d of %d chunks from %d documents", len(pending), len(chunks), len(documents))

        tokens_before = embedder.tokens_used
        with tqdm(total=len(pending), desc="Embedding", unit="chunk") as progress:
            for batch in batched(pending, INDEX_BATCH):
                self.store.add(batch, embedder.embed_documents([chunk.embedding_text for chunk in batch]))
                progress.update(len(batch))

        stats = {
            "documents": len(documents),
            "chunks": self.store.count(),
            "embedded": len(pending),
            "embedding_tokens": embedder.tokens_used - tokens_before,
            "seconds": round(time.perf_counter() - started, 1),
        }
        logger.info("Index built: %s", stats)
        return stats

    def retrieve(self, question: str, top_k: int | None = None, sources: list[str] | None = None) -> list[RetrievedChunk]:
        """Find the chunks most relevant to the question."""
        if self.store.count() == 0:
            raise IndexNotBuiltError("The index is empty. Run `python main.py ingest` first.")
        return self.retriever.retrieve(question, top_k, sources)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> RAGAnswer:
        """Ask the LLM to answer the question from the given chunks."""
        response = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks))
        answer = response.text.strip()
        if response.status == "refused" and not answer:
            answer = REFUSAL_MESSAGE
        logger.info(
            "Answered with %s (%d in / %d out tokens, status=%s)",
            response.model, response.input_tokens, response.output_tokens, response.status,
        )
        return RAGAnswer(question, answer, chunks, response)

    def answer(self, question: str, top_k: int | None = None, sources: list[str] | None = None) -> RAGAnswer:
        """Retrieve, then generate."""
        return self.generate(question, self.retrieve(question, top_k, sources))
