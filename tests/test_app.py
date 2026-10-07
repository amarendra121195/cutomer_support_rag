"""Tests for every pipeline stage.

OpenAI is replaced by small fakes, so the suite runs in seconds and needs neither the
datasets nor an API key:

    python -m pytest
"""
import json
import re
import zlib
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.routes import app, get_pipeline
from src.chunking.chunker import chunk_documents, split_text
from src.embeddings.embedder import Embedder
from src.evaluation.evaluator import Judge, evaluate_generation, evaluate_retrieval, judge_scores
from src.evaluation.metrics import hit_at_k, mean_ci, ndcg_at_k, recall_at_k, reciprocal_rank, rouge_l, token_f1
from src.ingestion.loader import (
    Document, EvalExample, html_to_text, load_bitext_documents, load_eval_examples, parse_article_ids,
)
from src.llm.llm_client import LLMClient, LLMResponse
from src.pipeline import IndexNotBuiltError, RAGPipeline
from src.prompts.prompt_templates import JUDGE_SCHEMA, build_user_prompt
from src.retrieval.retriever import Retriever
from src.utils.helpers import PROJECT_ROOT, APIError, openai_client
from src.vectordb.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------

DOCUMENTS = [
    Document(
        "doc-password", "Resetting your password",
        "To reset your password, open Settings and click Reset password. A reset link is emailed to you.",
        "wixqa", {"url": "https://example.com/password", "article_type": "article"},
    ),
    Document(
        "doc-domain", "Connecting a domain",
        "To connect a custom domain, go to the Domains page and click Connect a domain you already own.",
        "wixqa", {"url": "https://example.com/domain", "article_type": "article"},
    ),
    Document(
        "bitext-1", "Refund: get refund",
        "Customer question: how do I get a refund\nSupport answer: To request a refund for order {{Order Number}}, contact billing.",
        "bitext", {"category": "REFUND", "intent": "get_refund"},
    ),
]

VERDICT = {
    "relevance_reasoning": "Addresses the question.",
    "relevant": True,
    "unsupported_claims": [],
    "correctness_reasoning": "Matches the reference.",
    "correctness": "correct",
    "abstained": False,
}


class FakeEmbedder:
    """Bag-of-words hashing: texts that share words get similar vectors."""

    tokens_used = 0

    def embed_documents(self, texts):
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % 256] += 1.0
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeLLM:
    """Stands in for LLMClient: returns a canned reply and records the prompts it was sent."""

    def __init__(self, text="Open Settings and click Reset password [1].", status="ok", fail_on=None):
        self.text, self.status, self.fail_on = text, status, fail_on
        self.calls = []

    def generate(self, system, user, json_schema=None):
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if self.fail_on and self.fail_on in user:
            raise APIError("simulated API failure")
        return LLMResponse(self.text, "gpt-6.1-sol", self.status, input_tokens=1000, output_tokens=100)


@pytest.fixture
def config(tmp_path):
    return {
        "data": {
            "dir": str(tmp_path / "data"),
            "wixqa": {"kb_file": "kb.csv"},
            "bitext": {"enabled": True, "file": "bitext.csv", "max_per_intent": 2, "eval_per_intent": 1, "seed": 0},
        },
        "chunking": {"chunk_size": 200, "chunk_overlap": 40},
        "embeddings": {"model": "fake"},
        "retrieval": {"top_k": 2, "fetch_k": 3, "sources": None, "rerank": {"enabled": False}},
        "llm": {"model": "gpt-6.1-sol", "max_output_tokens": 16000, "effort": "medium"},
        "evaluation": {"k_values": [1, 3], "output_dir": str(tmp_path / "results")},
    }


@pytest.fixture
def pipeline(config, tmp_path):
    pipe = RAGPipeline(config, embedder=FakeEmbedder(), store=VectorStore(tmp_path / "db", "test_kb"), llm=FakeLLM())
    chunks = chunk_documents(DOCUMENTS, 200, 40)
    pipe.store.add(chunks, pipe.embedder.embed_documents([chunk.embedding_text for chunk in chunks]))
    return pipe


@pytest.fixture
def examples():
    return [
        EvalExample("q-password", "How do I reset my password?", "Open Settings and click Reset password.", ["doc-password"]),
        EvalExample("q-domain", "How can I connect a custom domain?", "Go to the Domains page and click Connect.", ["doc-domain"]),
    ]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_html_to_text_keeps_headings_and_lists():
    html = (
        '<div><div>Intro&nbsp;text with <strong>bold</strong> words.</div></div>'
        '<div><h2>Steps</h2></div>'
        '<div><ol><li>Click <strong>Add</strong>.<img src="x.png"/></li><li><img src="y.png"/></li><li>Click Save.</li></ol></div>'
        '<div><ul><li>First option</li><li>Second option</li></ul></div>'
    )
    assert html_to_text(html) == (
        "Intro text with bold words.\n"
        "\n"
        "## Steps\n"
        "1. Click Add.\n"
        "2. Click Save.\n"
        "- First option\n"
        "- Second option"
    )


def test_html_to_text_labels_tabs_and_flattens_tables():
    html = (
        '<div class="tabs-wrapper">'
        '<div class="headings-wrapper"><div class="tab-heading">Wix Editor</div><div class="tab-heading">Studio Editor</div></div>'
        '<div class="tab-contents-wrapper"><div class="tab-content">Click Embed.</div><div class="tab-content">Click Add.</div></div>'
        '</div>'
        '<table><tr><th><div>Fee</div></th><td><div>2.6%</div></td></tr></table>'
    )
    assert html_to_text(html) == "### Wix Editor\nClick Embed.\n\n### Studio Editor\nClick Add.\nFee | 2.6%"


def test_parse_article_ids_reads_numpy_style_lists():
    first, second = "a" * 64, "0f" * 32
    assert parse_article_ids(f"['{first}'\n '{second}']") == [first, second]
    assert parse_article_ids("[]") == []


@pytest.fixture
def dataset_files(config, tmp_path):
    """Tiny CSVs with the same columns as the real downloads."""
    data = tmp_path / "data"
    data.mkdir()
    article_id = "ab" * 32
    pd.DataFrame([{
        "id": article_id, "url": "https://example.com/a", "contents": "Adding a form\nflat text",
        "title": "Adding a form", "html_content": "<div><h2>Steps</h2><ol><li>Click Add.</li></ol></div>",
        "article_type": "article",
    }]).to_csv(data / "kb.csv", index=False)
    pd.DataFrame([
        {"flags": "B", "instruction": f"refund question {i}", "category": "REFUND", "intent": "get_refund", "response": f"Refund answer {i}."}
        for i in range(5)
    ]).to_csv(data / "bitext.csv", index=False)
    pd.DataFrame([{"question": "How do I add a form?", "answer": "Click Add.", "article_ids": f"['{article_id}']"}]).to_csv(
        data / "wixqa_wixqa_expertwritten_train.csv", index=False
    )
    return article_id


def test_build_index_from_dataset_files(config, tmp_path, dataset_files):
    pipe = RAGPipeline(config, embedder=FakeEmbedder(), store=VectorStore(tmp_path / "db", "test_kb"), llm=FakeLLM())
    stats = pipe.build_index()
    # One article plus two of the five Bitext rows (max_per_intent); one more row is held out for testing.
    assert stats["documents"] == 3
    assert stats["chunks"] == stats["embedded"] == 3

    article = pipe.retrieve("steps to add a form", top_k=1)[0]
    assert article.doc_id == dataset_files
    assert article.text == "## Steps\n1. Click Add."
    assert article.metadata["url"] == "https://example.com/a"

    # Resuming embeds nothing that is already stored; a fresh build replaces the index.
    assert pipe.build_index(resume=True)["embedded"] == 0
    assert pipe.build_index()["embedded"] == 3


def test_eval_sets_load_with_ground_truth(config, dataset_files):
    wix = load_eval_examples(config, "expertwritten")
    assert [(e.question, e.relevant, e.match_field) for e in wix] == [("How do I add a form?", [dataset_files], "doc_id")]

    bitext = load_eval_examples(config, "bitext")
    assert [(e.relevant, e.match_field) for e in bitext] == [(["get_refund"], "intent")]
    # A held-out test question must never also be in the index.
    indexed = load_bitext_documents(config)
    assert len(indexed) == 2
    assert all(bitext[0].question not in doc.text for doc in indexed)

    with pytest.raises(ValueError):
        load_eval_examples(config, "nonexistent")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_split_text_respects_size_and_loses_nothing():
    sentences = [f"Sentence number {i} explains step {i} of the process." for i in range(40)]
    text = " ".join(sentences)
    chunks = split_text(text, chunk_size=200, chunk_overlap=60)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(any(sentence in chunk for chunk in chunks) for sentence in sentences)
    # Neighbouring chunks share a whole sentence.
    assert chunks[1].split(". ")[0] + "." in chunks[0]


def test_split_text_cuts_at_headings_first():
    section_a = "## Section A\n" + "Alpha line of text.\n" * 6
    section_b = "## Section B\n" + "Beta line of text.\n" * 6
    chunks = split_text(section_a + "\n" + section_b, chunk_size=160, chunk_overlap=0)
    assert chunks == [section_a.strip(), section_b.strip()]


def test_short_text_is_a_single_chunk():
    assert split_text("Just one line.", 200, 40) == ["Just one line."]
    assert split_text("   ", 200, 40) == []


def test_chunk_documents_carries_metadata():
    chunk = chunk_documents(DOCUMENTS[:1], 200, 40)[0]
    assert chunk.chunk_id == "doc-password::0"
    assert chunk.metadata == {
        "doc_id": "doc-password", "title": "Resetting your password", "source": "wixqa",
        "chunk_index": 0, "url": "https://example.com/password", "article_type": "article",
    }
    assert chunk.embedding_text.startswith("Resetting your password\n")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def fake_openai(reply=None, error=None):
    """A stand-in for openai.OpenAI that records the requests it receives."""
    requests = []

    def create_response(**kwargs):
        requests.append(kwargs)
        if error:
            raise error
        return reply

    def create_embeddings(**kwargs):
        requests.append(kwargs)
        if error:
            raise error
        # Return the items out of order, as the API is allowed to.
        data = [SimpleNamespace(index=i, embedding=[float(len(text)), 0.0]) for i, text in enumerate(kwargs["input"])]
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(total_tokens=10 * len(kwargs["input"])))

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create_response),
        embeddings=SimpleNamespace(create=create_embeddings),
    )
    return client, requests


def test_embedder_batches_requests_and_keeps_input_order():
    client, requests = fake_openai()
    embedder = Embedder("text-embedding-3-large", batch_size=2, client=client)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = embedder.embed_documents(texts)
    assert vectors.dtype == np.float32
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [request["input"] for request in requests] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert requests[0] == {"model": "text-embedding-3-large", "input": ["a", "bb"]}
    assert embedder.tokens_used == 50
    assert embedder.cost_usd == pytest.approx(50 * 0.13 / 1e6)
    assert embedder.embed_query("abc").tolist() == [3.0, 0.0]


def test_embedder_passes_dimensions_only_when_set():
    client, requests = fake_openai()
    Embedder("text-embedding-3-large", dimensions=1024, client=client).embed_documents(["a"])
    assert requests[0]["dimensions"] == 1024


# ---------------------------------------------------------------------------
# Vector store and retrieval
# ---------------------------------------------------------------------------


def test_retrieve_finds_the_matching_document(pipeline):
    chunks = pipeline.retrieve("How do I reset my password?")
    assert len(chunks) == 2
    assert chunks[0].doc_id == "doc-password"
    assert chunks[0].score > chunks[1].score


def test_retrieve_can_be_restricted_to_a_source(pipeline):
    chunks = pipeline.retrieve("How do I reset my password?", sources=["bitext"])
    assert [chunk.metadata["source"] for chunk in chunks] == ["bitext"]


def test_reranker_reorders_the_candidate_pool(pipeline):
    class DomainFirst:
        def rerank(self, query, chunks):
            for chunk in chunks:
                chunk.rerank_score = float(chunk.doc_id == "doc-domain")
            return sorted(chunks, key=lambda chunk: chunk.rerank_score, reverse=True)

    retriever = Retriever(pipeline.embedder, pipeline.store, top_k=1, fetch_k=3, reranker=DomainFirst())
    chunks = retriever.retrieve("How do I reset my password?")
    assert [chunk.doc_id for chunk in chunks] == ["doc-domain"]


def test_reindexing_the_same_chunks_does_not_duplicate(pipeline):
    before = pipeline.store.count()
    chunks = chunk_documents(DOCUMENTS, 200, 40)
    pipeline.store.add(chunks, pipeline.embedder.embed_documents([chunk.embedding_text for chunk in chunks]))
    assert pipeline.store.count() == before
    assert pipeline.store.ids() == {chunk.chunk_id for chunk in chunks}
    pipeline.store.reset()
    assert pipeline.store.count() == 0


def test_empty_index_is_reported(config, tmp_path):
    empty = RAGPipeline(config, embedder=FakeEmbedder(), store=VectorStore(tmp_path / "empty", "test_kb"), llm=FakeLLM())
    with pytest.raises(IndexNotBuiltError, match="index is empty"):
        empty.retrieve("anything")


def test_index_built_with_another_embedding_size_is_reported(pipeline):
    # e.g. embeddings.model or embeddings.dimensions was changed without re-running ingest
    with pytest.raises(IndexNotBuiltError, match="different embedding model"):
        pipeline.store.search(np.ones(8, dtype=np.float32), k=3)


# ---------------------------------------------------------------------------
# Prompts, LLM client and the full pipeline
# ---------------------------------------------------------------------------


def test_prompt_numbers_the_sources(pipeline):
    chunks = pipeline.retrieve("How do I reset my password?")
    prompt = build_user_prompt("  How do I reset my password?  ", chunks)
    assert '<source id="1">\nTitle: Resetting your password\nTo reset your password' in prompt
    assert '<source id="2">' in prompt
    assert prompt.endswith("<question>\nHow do I reset my password?\n</question>")
    assert "No relevant sources" in build_user_prompt("Anything?", [])


def openai_reply(text="Hello.", status="completed", incomplete_reason=None, refusal=None):
    """A Responses API reply: a reasoning item followed by one message."""
    part = SimpleNamespace(type="refusal", refusal=refusal) if refusal else SimpleNamespace(type="output_text", text=text)
    return SimpleNamespace(
        output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="message", content=[part])],
        output_text="" if refusal else text,
        status=status,
        incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
        model="gpt-6.1-sol",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
    )


def test_llm_client_sends_expected_request_and_reads_reply():
    client, requests = fake_openai(openai_reply())
    llm = LLMClient("gpt-6.1-sol", max_output_tokens=16000, effort="medium", client=client)

    response = llm.generate("system prompt", "user prompt")
    assert requests[0] == {
        "model": "gpt-6.1-sol",
        "instructions": "system prompt",
        "input": "user prompt",
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 16000,
    }
    assert (response.text, response.status, response.model) == ("Hello.", "ok", "gpt-6.1-sol")
    assert response.cost_usd == pytest.approx(1000 * 2 / 1e6 + 500 * 10 / 1e6)

    llm.generate("system prompt", "user prompt", json_schema=JUDGE_SCHEMA)
    assert requests[1]["text"] == {
        "format": {"type": "json_schema", "name": "structured_output", "schema": JUDGE_SCHEMA, "strict": True}
    }


def test_llm_client_classifies_truncated_and_refused_replies():
    truncated, _ = fake_openai(openai_reply("Half an ans", status="incomplete", incomplete_reason="max_output_tokens"))
    response = LLMClient("gpt-6.1-sol", client=truncated).generate("system", "user")
    assert (response.status, response.text) == ("truncated", "Half an ans")

    refused, _ = fake_openai(openai_reply(refusal="I can't help with that."))
    response = LLMClient("gpt-6.1-sol", client=refused).generate("system", "user")
    assert (response.status, response.text) == ("refused", "I can't help with that.")

    filtered, _ = fake_openai(openai_reply("", status="incomplete", incomplete_reason="content_filter"))
    assert LLMClient("gpt-6.1-sol", client=filtered).generate("system", "user").status == "refused"


def test_openai_errors_become_readable_messages():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    def failing_with(error):
        client, _ = fake_openai(error=error)
        return LLMClient("gpt-9-imaginary", client=client)

    not_found = openai.NotFoundError("no such model", response=httpx.Response(404, request=request), body=None)
    with pytest.raises(APIError, match="'gpt-9-imaginary' was not found"):
        failing_with(not_found).generate("system", "user")

    quota = openai.RateLimitError("You exceeded your current quota", response=httpx.Response(429, request=request), body=None)
    with pytest.raises(APIError, match="You exceeded your current quota"):
        failing_with(quota).generate("system", "user")

    with pytest.raises(APIError, match="internet connection"):
        failing_with(openai.APIConnectionError(request=request)).generate("system", "user")

    # The embedder reports failures the same way.
    client, _ = fake_openai(error=openai.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None))
    with pytest.raises(APIError, match="OPENAI_API_KEY"):
        Embedder("text-embedding-3-large", client=client).embed_query("hello")


def test_missing_api_key_is_explained(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(APIError, match="OPENAI_API_KEY is not set"):
        openai_client()
    monkeypatch.setenv("OPENAI_API_KEY", "   ")  # an empty line in .env
    with pytest.raises(APIError, match="OPENAI_API_KEY is not set"):
        openai_client()


def test_unknown_model_has_no_cost():
    assert LLMResponse("x", "some-future-model", "ok", 10, 10).cost_usd is None


def test_pipeline_answers_from_retrieved_sources(pipeline):
    result = pipeline.answer("How do I reset my password?")
    assert result.answer == "Open Settings and click Reset password [1]."
    assert result.sources[0].doc_id == "doc-password"
    assert "open Settings and click Reset password" in pipeline.llm.calls[0]["user"]


def test_refusal_gets_a_readable_answer(pipeline):
    pipeline.llm = FakeLLM(text="", status="refused")
    assert "declined" in pipeline.answer("How do I reset my password?").answer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_retrieval_metrics():
    ranked, relevant = ["a", "b", "c", "d"], {"c", "x"}
    assert hit_at_k(ranked, relevant, 2) == 0.0
    assert hit_at_k(ranked, relevant, 3) == 1.0
    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"x"}) == 0.0
    # One relevant item found at rank 3 of an ideal two at ranks 1 and 2.
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx((1 / np.log2(4)) / (1 + 1 / np.log2(3)))
    assert ndcg_at_k(["c", "x"], relevant, 3) == pytest.approx(1.0)


def test_answer_metrics_reward_matching_the_reference():
    reference = "Open Settings and click [Reset password](https://example.com/reset)."
    # The reference answer itself must score 1, an empty or unrelated answer 0.
    assert token_f1(reference, reference) == 1.0
    assert rouge_l(reference, reference) == 1.0
    assert token_f1("", reference) == rouge_l("", reference) == 0.0
    assert token_f1("Bananas are yellow", reference) == 0.0
    # Links and citation markers are not counted as words.
    assert token_f1("Open Settings and click Reset password [1].", reference) == 1.0
    # Same words in the wrong order: full word overlap, lower ROUGE-L.
    shuffled = "password Reset click and Settings Open"
    assert token_f1(shuffled, reference) == 1.0
    assert rouge_l(shuffled, reference) < 0.5


def test_mean_ci():
    assert mean_ci([1.0, 0.0, 1.0, 0.0]) == {"mean": 0.5, "ci95": pytest.approx(0.5658, abs=1e-4), "n": 4}
    assert mean_ci([1.0]) == {"mean": 1.0, "ci95": None, "n": 1}
    assert mean_ci([])["mean"] is None


def test_evaluate_retrieval(pipeline, examples):
    details, summary = evaluate_retrieval(pipeline, examples, k_values=[1, 3])
    assert summary["questions"] == 2
    assert summary["metrics"]["hit@1"]["mean"] == 1.0
    assert summary["metrics"]["mrr"]["mean"] == 1.0
    assert summary["metrics"]["context_recall"]["mean"] == 1.0
    assert list(details["id"]) == ["q-password", "q-domain"]

    wrong_labels = [EvalExample("q", "How do I reset my password?", "", ["doc-that-does-not-exist"])]
    _, summary = evaluate_retrieval(pipeline, wrong_labels, k_values=[1, 3])
    assert summary["metrics"]["hit@3"]["mean"] == 0.0


def test_evaluate_retrieval_can_match_on_intent(pipeline):
    example = EvalExample("b", "how do I get a refund for my order", "", ["get_refund"], match_field="intent")
    _, summary = evaluate_retrieval(pipeline, [example], k_values=[1])
    assert summary["metrics"]["hit@1"]["mean"] == 1.0


def test_judge_scores():
    assert judge_scores(VERDICT) == {"correctness": 1.0, "faithfulness": 1.0, "relevance": 1.0, "abstained": 0.0}
    bad = {**VERDICT, "correctness": "partially_correct", "unsupported_claims": ["Refunds take 3 days"], "relevant": False}
    assert judge_scores(bad) == {"correctness": 0.5, "faithfulness": 0.0, "relevance": 0.0, "abstained": 0.0}


def test_evaluate_generation_scores_every_case(pipeline, examples, tmp_path):
    judge_llm = FakeLLM(text=json.dumps(VERDICT))
    summary = evaluate_generation(pipeline, examples, tmp_path, Judge(judge_llm), concurrency=2)

    assert summary["cases"] == {"scored": 2, "truncated": 0, "refused": 0, "errors": 0}
    assert summary["metrics"]["correctness"] == {"mean": 1.0, "ci95": 0.0, "n": 2}
    assert summary["metrics"]["faithfulness"]["mean"] == 1.0
    assert summary["metrics"]["context_recall"]["mean"] == 1.0
    assert 0 < summary["metrics"]["token_f1"]["mean"] < 1
    assert 0 < summary["metrics"]["semantic_similarity"]["mean"] <= 1
    assert summary["generator"] == {
        "models": ["gpt-6.1-sol"], "input_tokens": 2000, "output_tokens": 200,
        "cost_usd": pytest.approx(2 * (1000 * 2 + 100 * 10) / 1e6),
    }
    assert summary["judge"]["input_tokens"] == 2000

    rows = [json.loads(line) for line in (tmp_path / "generation_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in rows} == {"q-password", "q-domain"}
    assert rows[0]["judge_verdict"] == VERDICT
    # The judge is shown the question, the sources, the reference and the answer, and asked for JSON.
    graded = judge_llm.calls[0]
    assert graded["json_schema"] == JUDGE_SCHEMA
    for tag in ("<question>", "<sources>", "<reference_answer>", "<system_answer>"):
        assert tag in graded["user"]


def test_api_failures_are_not_scored_as_wrong_answers(pipeline, examples, tmp_path):
    pipeline.llm = FakeLLM(fail_on="custom domain")
    summary = evaluate_generation(pipeline, examples, tmp_path, judge=None, concurrency=2)
    assert summary["cases"] == {"scored": 1, "truncated": 0, "refused": 0, "errors": 1}
    assert summary["metrics"]["token_f1"]["n"] == 1
    errors = (tmp_path / "generation_errors.jsonl").read_text(encoding="utf-8")
    assert json.loads(errors)["id"] == "q-domain"

    # Resuming re-runs only the failed case.
    pipeline.llm = FakeLLM()
    summary = evaluate_generation(pipeline, examples, tmp_path, judge=None, concurrency=2, resume=True)
    assert len(pipeline.llm.calls) == 1
    assert summary["cases"] == {"scored": 2, "truncated": 0, "refused": 0, "errors": 0}


def test_unusable_judge_output_is_an_error_not_a_score(pipeline, examples, tmp_path):
    summary = evaluate_generation(pipeline, examples, tmp_path, Judge(FakeLLM(text="{}", status="truncated")))
    assert summary["cases"] == {"scored": 0, "truncated": 0, "refused": 0, "errors": 2}


def test_truncated_and_refused_answers_are_counted_separately(pipeline, examples, tmp_path):
    pipeline.llm = FakeLLM(status="truncated")
    summary = evaluate_generation(pipeline, examples, tmp_path, judge=None)
    assert summary["cases"] == {"scored": 0, "truncated": 2, "refused": 0, "errors": 0}
    assert summary["metrics"] == {}

    pipeline.llm = FakeLLM(text="", status="refused")
    summary = evaluate_generation(pipeline, examples, tmp_path, judge=None)
    assert summary["cases"] == {"scored": 0, "truncated": 0, "refused": 2, "errors": 0}


def test_evaluation_stops_when_the_first_call_fails(pipeline, examples, tmp_path):
    pipeline.llm = FakeLLM(fail_on="<question>")  # every call fails, e.g. a rejected API key
    with pytest.raises(APIError):
        evaluate_generation(pipeline, examples, tmp_path, judge=None)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture
def client(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_health(client, pipeline):
    assert client.get("/health").json() == {"status": "ok", "indexed_chunks": pipeline.store.count()}


def test_api_retrieve(client):
    body = client.post("/retrieve", json={"question": "How do I reset my password?", "top_k": 1}).json()
    assert [source["title"] for source in body["sources"]] == ["Resetting your password"]
    assert body["sources"][0]["id"] == 1
    assert body["sources"][0]["url"] == "https://example.com/password"


def test_api_ask(client):
    response = client.post("/ask", json={"question": "How do I reset my password?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Open Settings and click Reset password [1]."
    assert body["model"] == "gpt-6.1-sol"
    assert len(body["sources"]) == 2


def test_api_rejects_blank_questions(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_api_reports_openai_failures(client, pipeline):
    pipeline.llm = FakeLLM(fail_on="<question>")
    response = client.post("/ask", json={"question": "How do I reset my password?"})
    assert response.status_code == 502
    assert response.json()["detail"] == "simulated API failure"

    def broken_embedding(text):
        raise APIError("OPENAI_API_KEY is not set.")

    pipeline.embedder.embed_query = broken_embedding
    response = client.post("/retrieve", json={"question": "How do I reset my password?"})
    assert response.status_code == 502
    assert "OPENAI_API_KEY" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------


def test_streamlit_page_answers_a_question(pipeline, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # The page builds its pipeline through these two names; hand it the test pipeline instead.
    monkeypatch.setattr("src.pipeline.RAGPipeline", lambda config: pipeline)
    monkeypatch.setattr("src.utils.helpers.setup_logging", lambda config: None)
    streamlit.cache_resource.clear()

    page = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not page.exception
    assert len(page.chat_message) == 0

    page.sidebar.radio[0].set_value("Wix help center")
    page.chat_input[0].set_value("How do I reset my password?").run()
    assert not page.exception
    user, assistant = page.chat_message
    assert user.markdown[0].value == "How do I reset my password?"
    assert assistant.markdown[0].value == "Open Settings and click Reset password [1]."
    assert "gpt-6.1-sol · 1,000 input / 100 output tokens" in assistant.caption[0].value
    # Both Wix documents are listed as sources; the Bitext one was filtered out in the sidebar.
    assert assistant.expander[0].label == "Sources (2)"
    assert assistant.text[0].value.startswith("To reset your password")
    assert all("bitext" not in caption.value for caption in assistant.caption)

    # An OpenAI failure is shown as an error message, not a crash.
    pipeline.llm = FakeLLM(fail_on="<question>")
    page.chat_input[0].set_value("How can I connect a custom domain?").run()
    assert not page.exception
    assert page.error[0].value == "simulated API failure"
    streamlit.cache_resource.clear()
