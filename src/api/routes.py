"""REST API: FastAPI endpoints over the RAG pipeline."""
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from src.pipeline import IndexNotBuiltError, RAGPipeline
from src.utils.helpers import APIError, load_config, setup_logging
from src.vectordb.vector_store import RetrievedChunk

router = APIRouter()


class QuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    top_k: int | None = Field(default=None, ge=1, le=20, description="Chunks to retrieve (default from config.yaml)")
    sources: list[str] | None = Field(default=None, description='Restrict the search, e.g. ["wixqa"]')


class Source(BaseModel):
    id: int = Field(description="Citation number used in the answer, e.g. [1]")
    title: str
    url: str | None
    source: str
    score: float
    text: str


class RetrieveResponse(BaseModel):
    question: str
    sources: list[Source]


class AskResponse(RetrieveResponse):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int


@lru_cache
def get_pipeline() -> RAGPipeline:
    """One pipeline for the whole process, so the models are loaded once."""
    config = load_config()
    setup_logging(config)
    return RAGPipeline(config)


def _sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            id=number,
            title=chunk.metadata["title"],
            url=chunk.metadata.get("url"),
            source=chunk.metadata["source"],
            score=round(chunk.score, 4),
            text=chunk.text,
        )
        for number, chunk in enumerate(chunks, start=1)
    ]


@router.get("/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    return {"status": "ok", "indexed_chunks": pipeline.store.count()}


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QuestionRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> RetrieveResponse:
    """Search the knowledge base without calling the chat model."""
    try:
        chunks = pipeline.retrieve(request.question, request.top_k, request.sources)
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except APIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return RetrieveResponse(question=request.question, sources=_sources(chunks))


@router.post("/ask", response_model=AskResponse)
def ask(request: QuestionRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    """Answer a question from the knowledge base."""
    try:
        result = pipeline.answer(request.question, request.top_k, request.sources)
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except APIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return AskResponse(
        question=request.question,
        answer=result.answer,
        sources=_sources(result.sources),
        model=result.response.model,
        input_tokens=result.response.input_tokens,
        output_tokens=result.response.output_tokens,
    )


app = FastAPI(
    title="Customer Support RAG",
    description="Question answering over the Wix help center and Bitext support datasets.",
    version="1.0.0",
)
app.include_router(router)
