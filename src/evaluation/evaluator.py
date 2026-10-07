"""Evaluation: measure retrieval quality and answer quality on a labelled question set.

Two stages, each usable on its own:

* Retrieval  - is the right article retrieved? Scored against the ground-truth article ids.
               Only the question embeddings are paid for (a fraction of a cent in total).
* Generation - is the final answer right? Each answer is compared with the expert reference
               (word overlap, embedding similarity) and graded by a second model, the judge.
"""
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import hit_at_k, mean_ci, ndcg_at_k, recall_at_k, reciprocal_rank, rouge_l, token_f1
from src.ingestion.loader import EvalExample, load_eval_examples
from src.llm.llm_client import LLMClient, LLMResponse
from src.pipeline import RAGPipeline
from src.prompts.prompt_templates import JUDGE_SCHEMA, JUDGE_SYSTEM_PROMPT, build_judge_prompt
from src.utils.helpers import APIError, resolve_path, unique
from src.vectordb.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

CORRECTNESS_SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def ranked_items(chunks: list[RetrievedChunk], match_field: str) -> list[str]:
    """Collapse ranked chunks into the ranked list of distinct articles (or intents) they belong to."""
    return unique(chunk.metadata.get(match_field, chunk.doc_id) for chunk in chunks)


# ---------------------------------------------------------------------------
# Stage 1: retrieval
# ---------------------------------------------------------------------------


def evaluate_retrieval(pipeline: RAGPipeline, examples: list[EvalExample], k_values: list[int]) -> tuple[pd.DataFrame, dict]:
    """Score the retriever on every example. Returns per-question details and a summary."""
    cfg = pipeline.config["retrieval"]
    rows = []
    for example in tqdm(examples, desc="Retrieval", unit="q"):
        started = time.perf_counter()
        # Rank the whole candidate pool, so hit@10 can be measured even though the LLM sees fewer.
        chunks = pipeline.retrieve(example.question, top_k=cfg["fetch_k"])
        latency = time.perf_counter() - started

        relevant = set(example.relevant)
        ranked = ranked_items(chunks, example.match_field)
        row = {"id": example.example_id, "question": example.question, "mrr": reciprocal_rank(ranked, relevant)}
        for k in k_values:
            row[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
            row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        # Share of the correct articles present in the chunks the LLM is actually given.
        served = ranked_items(chunks[:cfg["top_k"]], example.match_field)
        row["context_recall"] = recall_at_k(served, relevant, len(served))
        row["latency_s"] = round(latency, 3)
        rows.append(row)

    details = pd.DataFrame(rows)
    metric_names = [name for name in details.columns if name not in ("id", "question", "latency_s")]
    summary = {
        "questions": len(details),
        "metrics": {name: mean_ci(details[name].tolist()) for name in metric_names},
        "mean_latency_s": round(details["latency_s"].mean(), 3),
    }
    return details, summary


# ---------------------------------------------------------------------------
# Stage 2: generation
# ---------------------------------------------------------------------------


class JudgeError(RuntimeError):
    """The judge did not return a usable verdict."""


class Judge:
    """Grades an answer with a second model, using a fixed rubric and structured output."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def grade(self, question: str, chunks: list[RetrievedChunk], reference_answer: str, answer: str) -> tuple[dict, LLMResponse]:
        response = self.llm.generate(
            JUDGE_SYSTEM_PROMPT,
            build_judge_prompt(question, chunks, reference_answer, answer),
            json_schema=JUDGE_SCHEMA,
        )
        if response.status != "ok":
            raise JudgeError(f"Judge response was {response.status}")
        try:
            return json.loads(response.text), response
        except json.JSONDecodeError as e:
            raise JudgeError(f"Judge returned invalid JSON: {e}") from e


def judge_scores(verdict: dict) -> dict[str, float]:
    """Turn the judge's verdict into numeric scores."""
    return {
        "correctness": CORRECTNESS_SCORES[verdict["correctness"]],
        "faithfulness": float(not verdict["unsupported_claims"]),
        "relevance": float(verdict["relevant"]),
        "abstained": float(verdict["abstained"]),
    }


def _usage(response: LLMResponse) -> dict:
    return {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}


def _run_case(pipeline: RAGPipeline, judge: Judge | None, example: EvalExample, reranker_lock: threading.Lock) -> dict:
    """Answer one question through the real pipeline and score the answer."""
    with reranker_lock:  # the local reranking model handles one question at a time
        chunks = pipeline.retrieve(example.question)
    started = time.perf_counter()
    result = pipeline.generate(example.question, chunks)
    latency = time.perf_counter() - started

    response = result.response
    row = {
        "id": example.example_id,
        "status": response.status,
        "question": example.question,
        "reference_answer": example.reference_answer,
        "answer": result.answer,
        "retrieved": [
            {"chunk_id": chunk.chunk_id, "title": chunk.metadata["title"], "score": round(chunk.score, 4)}
            for chunk in chunks
        ],
        "scores": {},
        "model": response.model,
        "usage": _usage(response),
        "cost_usd": response.cost_usd,
        "latency_s": round(latency, 2),
    }
    # A refused or cut-off answer is reported on its own, not averaged in as a wrong answer.
    if response.status != "ok":
        return row

    served = ranked_items(chunks, example.match_field)
    similarity = 0.0
    if result.answer:  # the embeddings API rejects empty input
        answer_vector, reference_vector = pipeline.embedder.embed_documents([result.answer, example.reference_answer])
        similarity = float(answer_vector @ reference_vector)
    row["scores"] = {
        "context_recall": recall_at_k(served, set(example.relevant), len(served)),
        "token_f1": token_f1(result.answer, example.reference_answer),
        "rouge_l": rouge_l(result.answer, example.reference_answer),
        "semantic_similarity": similarity,
    }
    if judge:
        verdict, judge_response = judge.grade(example.question, chunks, example.reference_answer, result.answer)
        row["scores"].update(judge_scores(verdict))
        row.update(
            judge_verdict=verdict,
            judge_model=judge_response.model,
            judge_usage=_usage(judge_response),
            judge_cost_usd=judge_response.cost_usd,
        )
    return row


def evaluate_generation(
    pipeline: RAGPipeline,
    examples: list[EvalExample],
    run_dir: Path,
    judge: Judge | None = None,
    concurrency: int = 4,
    resume: bool = False,
) -> dict:
    """Answer and score every example, writing one JSON line per finished case.

    Cases that fail because of the API (rate limit, timeout, unusable judge output) go to
    generation_errors.jsonl instead of being scored as wrong; `resume=True` re-runs only those
    and whatever was never reached.
    """
    results_path = run_dir / "generation_results.jsonl"
    errors_path = run_dir / "generation_errors.jsonl"
    if not resume:
        results_path.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)
    done = {row["id"] for row in _read_jsonl(results_path)}
    pending = [example for example in examples if example.example_id not in done]
    reranker_lock = threading.Lock()

    def finish(example: EvalExample, get_row, abort_on_api_error: bool = False) -> None:
        try:
            _append_jsonl(results_path, get_row())
        except (APIError, JudgeError) as error:
            if abort_on_api_error and isinstance(error, APIError):
                raise
            logger.warning("Case %s failed: %s", example.example_id, error)
            _append_jsonl(errors_path, {"id": example.example_id, "error_type": type(error).__name__, "error": str(error)})

    if pending:
        # Run one case alone first: a missing key or wrong model name should stop the run, not fail every case.
        finish(pending[0], lambda: _run_case(pipeline, judge, pending[0], reranker_lock), abort_on_api_error=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_case, pipeline, judge, example, reranker_lock): example for example in pending[1:]}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generation", unit="q"):
            finish(futures[future], future.result)

    return summarize_generation(_read_jsonl(results_path), _read_jsonl(errors_path))


def summarize_generation(rows: list[dict], errors: list[dict]) -> dict:
    """Aggregate per-case rows. Only cases with a complete answer are averaged."""
    scored = [row for row in rows if row["status"] == "ok"]
    finished = {row["id"] for row in rows}

    def spend(model_key: str, usage_key: str, cost_key: str) -> dict:
        used = [row for row in rows if row.get(usage_key)]
        costs = [row[cost_key] for row in used]
        return {
            "models": unique(row[model_key] for row in used),
            "input_tokens": sum(row[usage_key]["input_tokens"] for row in used),
            "output_tokens": sum(row[usage_key]["output_tokens"] for row in used),
            # None when a serving model's price is unknown, rather than a total that leaves it out.
            "cost_usd": None if None in costs else round(sum(costs), 4),
        }

    return {
        "cases": {
            "scored": len(scored),
            "truncated": sum(row["status"] == "truncated" for row in rows),
            "refused": sum(row["status"] == "refused" for row in rows),
            "errors": len({error["id"] for error in errors} - finished),
        },
        "metrics": {
            name: mean_ci([row["scores"][name] for row in scored])
            for name in unique(name for row in scored for name in row["scores"])
        },
        "mean_latency_s": round(sum(row["latency_s"] for row in rows) / len(rows), 2) if rows else None,
        "generator": spend("model", "usage", "cost_usd"),
        "judge": spend("judge_model", "judge_usage", "judge_cost_usd"),
    }


def check_judge(pipeline: RAGPipeline, judge: Judge, examples: list[EvalExample]) -> list[dict]:
    """Give the judge answers whose correct grade is known in advance.

    A judge that passes the reference answer and fails an empty answer, "I don't know" and
    the answer to a different question can be trusted to tell good from bad.
    """
    results = []
    for index, example in enumerate(examples):
        chunks = pipeline.retrieve(example.question)
        unrelated = examples[index - 1].reference_answer
        probes = [
            ("reference answer", example.reference_answer, {"correctness": "correct"}),
            ("empty answer", "", {"correctness": "incorrect", "abstained": True}),
            ('"I don\'t know."', "I don't know.", {"correctness": "incorrect", "abstained": True}),
            ("answer to another question", unrelated, {"correctness": "incorrect", "relevant": False}),
        ]
        for name, answer, expected in probes:
            verdict, _ = judge.grade(example.question, chunks, example.reference_answer, answer)
            got = {key: verdict[key] for key in expected}
            results.append({"id": example.example_id, "probe": name, "expected": expected, "got": got, "passed": got == expected})
    return results


# ---------------------------------------------------------------------------
# Running an evaluation end to end
# ---------------------------------------------------------------------------


def run_evaluation(
    pipeline: RAGPipeline,
    dataset: str,
    limit: int | None = None,
    retrieval_only: bool = False,
    use_judge: bool = True,
    resume: bool = False,
    run_name: str | None = None,
) -> dict:
    """Evaluate on one dataset and save the results under evaluation_results/<run_name>/."""
    cfg = pipeline.config["evaluation"]
    examples = load_eval_examples(pipeline.config, dataset, limit)
    run_dir = resolve_path(cfg["output_dir"]) / (run_name or dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "dataset": dataset,
        "questions": len(examples),
        **{key: pipeline.config[key] for key in ("chunking", "embeddings", "retrieval", "llm")},
    }

    details, retrieval = evaluate_retrieval(pipeline, examples, cfg["k_values"])
    details.to_csv(run_dir / "retrieval_details.csv", index=False)
    _write_json(run_dir / "retrieval_summary.json", {"settings": settings, **retrieval})
    summaries = {"run_dir": str(run_dir), "retrieval": retrieval}

    if not retrieval_only:
        judge = None
        if use_judge:
            judge = Judge(LLMClient(cfg["judge_model"], pipeline.config["llm"]["max_output_tokens"], cfg["judge_effort"]))
        generation = evaluate_generation(pipeline, examples, run_dir, judge, cfg["concurrency"], resume)
        _write_json(run_dir / "generation_summary.json", {"settings": settings, **generation})
        summaries["generation"] = generation
    return summaries


def format_metrics(metrics: dict) -> str:
    """Render {name: mean_ci(...)} as an aligned text table."""
    lines = []
    for name, stat in metrics.items():
        interval = f"+/- {stat['ci95']:.3f}" if stat["ci95"] is not None else ""
        lines.append(f"  {name:<20} {stat['mean']:.3f} {interval:<10} (n={stat['n']})")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
