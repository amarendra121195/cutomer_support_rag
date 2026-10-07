"""Command-line entry point for the RAG project.

    python main.py ingest                        build the vector index from rag_datasets/
    python main.py search "your question"        show the retrieved chunks
    python main.py ask "your question"           answer a question
    python main.py ask                           interactive question loop
    python main.py evaluate --retrieval-only     measure retrieval quality
    python main.py evaluate --limit 20           measure retrieval and answer quality
    python main.py serve                         start the REST API

Every command needs OPENAI_API_KEY in .env.
"""
import argparse
import sys
import textwrap

from src.utils.helpers import load_config, setup_logging


def print_sources(chunks, show_text: bool = False) -> None:
    for number, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        score = f"similarity {chunk.score:.2f}"
        if chunk.rerank_score is not None:
            score += f", rerank {chunk.rerank_score:.2f}"
        print(f"[{number}] {meta['title']}  ({meta['source']}, {score})")
        if meta.get("url"):
            print(f"    {meta['url']}")
        if show_text:
            print(textwrap.indent(textwrap.shorten(chunk.text, width=400, placeholder=" ..."), "    "))
            print()


def cmd_ingest(pipeline, args) -> None:
    from src.utils.helpers import APIError

    try:
        stats = pipeline.build_index(resume=args.resume)
    except APIError as error:
        stored = pipeline.store.count()
        hint = f" {stored:,} chunks are already stored; run `python main.py ingest --resume` to continue." if stored else ""
        raise APIError(f"{error}{hint}") from error
    cost = pipeline.embedder.cost_usd
    print(
        f"Index holds {stats['chunks']:,} chunks from {stats['documents']:,} documents. "
        f"Embedded {stats['embedded']:,} chunks ({stats['embedding_tokens']:,} tokens"
        + ("" if cost is None else f", ${cost:.2f}")
        + f") in {stats['seconds']:.0f}s."
    )


def cmd_search(pipeline, args) -> None:
    print_sources(pipeline.retrieve(args.question, args.top_k), show_text=True)


def cmd_ask(pipeline, args) -> None:
    def answer(question: str) -> None:
        result = pipeline.answer(question, args.top_k)
        print(f"\n{result.answer}\n\nSources:")
        print_sources(result.sources)

    if args.question:
        answer(args.question)
        return
    print("Ask a question (blank line to quit).")
    while question := input("\n> ").strip():
        answer(question)


def cmd_evaluate(pipeline, args) -> None:
    from src.evaluation.evaluator import Judge, check_judge, format_metrics, run_evaluation
    from src.ingestion.loader import load_eval_examples
    from src.llm.llm_client import LLMClient

    cfg = pipeline.config["evaluation"]
    dataset = args.dataset or cfg["dataset"]

    if args.check_judge:
        judge = Judge(LLMClient(cfg["judge_model"], pipeline.config["llm"]["max_output_tokens"], cfg["judge_effort"]))
        results = check_judge(pipeline, judge, load_eval_examples(pipeline.config, dataset, limit=3))
        for result in results:
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"{mark}  {result['id']:<18} {result['probe']:<28} expected {result['expected']}  got {result['got']}")
        print(f"\n{sum(r['passed'] for r in results)}/{len(results)} judge checks passed.")
        return

    summaries = run_evaluation(
        pipeline, dataset, args.limit, args.retrieval_only, not args.no_judge, args.resume, args.run_name
    )
    retrieval = summaries["retrieval"]
    print(f"\nRetrieval on '{dataset}' ({retrieval['questions']} questions, {retrieval['mean_latency_s']}s per query)")
    print(format_metrics(retrieval["metrics"]))

    if "generation" in summaries:
        generation = summaries["generation"]
        cases = generation["cases"]
        print(
            f"\nGeneration: {cases['scored']} scored, {cases['truncated']} truncated, "
            f"{cases['refused']} refused, {cases['errors']} errors"
        )
        print(format_metrics(generation["metrics"]))
        for role in ("generator", "judge"):
            spend = generation[role]
            if spend["models"]:
                cost = "unknown" if spend["cost_usd"] is None else f"${spend['cost_usd']:.2f}"
                print(
                    f"  {role}: {', '.join(spend['models'])} - {spend['input_tokens']:,} input / "
                    f"{spend['output_tokens']:,} output tokens - cost {cost}"
                )
    print(f"\nResults saved to {summaries['run_dir']}")


def cmd_serve(pipeline, args) -> None:
    import uvicorn

    api = pipeline.config["api"]
    uvicorn.run("src.api.routes:app", host=args.host or api["host"], port=args.port or api["port"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Customer-support RAG over the WixQA and Bitext datasets.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_retrieval_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--sources", nargs="+", metavar="SOURCE", help="restrict the search, e.g. --sources wixqa")
        command.add_argument(
            "--rerank", action=argparse.BooleanOptionalAction,
            help="turn cross-encoder reranking on or off (default: retrieval.rerank.enabled in config.yaml)",
        )

    ingest = commands.add_parser("ingest", help="build the vector index")
    ingest.add_argument("--resume", action="store_true", help="keep stored chunks and embed only the missing ones")
    ingest.set_defaults(run=cmd_ingest)

    search = commands.add_parser("search", help="show the chunks retrieved for a question")
    search.add_argument("question")
    search.add_argument("--top-k", type=int)
    add_retrieval_options(search)
    search.set_defaults(run=cmd_search)

    ask = commands.add_parser("ask", help="answer a question (omit the question for interactive mode)")
    ask.add_argument("question", nargs="?")
    ask.add_argument("--top-k", type=int)
    add_retrieval_options(ask)
    ask.set_defaults(run=cmd_ask)

    evaluate = commands.add_parser("evaluate", help="measure retrieval and answer quality")
    evaluate.add_argument("--dataset", choices=["expertwritten", "simulated", "synthetic", "bitext"])
    evaluate.add_argument("--limit", type=int, help="only use the first N questions")
    evaluate.add_argument("--retrieval-only", action="store_true", help="skip answer generation and the judge")
    evaluate.add_argument("--no-judge", action="store_true", help="score answers without the LLM judge")
    evaluate.add_argument("--resume", action="store_true", help="continue an interrupted generation run")
    evaluate.add_argument("--run-name", help="results folder name (default: the dataset name)")
    evaluate.add_argument("--check-judge", action="store_true", help="test the judge on answers with known grades")
    add_retrieval_options(evaluate)
    evaluate.set_defaults(run=cmd_evaluate)

    serve = commands.add_parser("serve", help="start the REST API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(run=cmd_serve)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # Knowledge-base text contains characters the default Windows console encoding cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = load_config()
    setup_logging(config)
    if getattr(args, "sources", None):
        config["retrieval"]["sources"] = args.sources
    if getattr(args, "rerank", None) is not None:
        config["retrieval"]["rerank"]["enabled"] = args.rerank

    from src.pipeline import IndexNotBuiltError, RAGPipeline
    from src.utils.helpers import APIError

    try:
        args.run(RAGPipeline(config), args)
    except (IndexNotBuiltError, APIError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
