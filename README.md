# Customer Support RAG

A retrieval-augmented question-answering system for customer support. It indexes the Wix help
center (6,221 articles from the [WixQA](https://huggingface.co/datasets/Wix/WixQA) dataset) and a
sample of the [Bitext customer-support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
dataset, retrieves the passages relevant to a question, and has an OpenAI model write an answer
that cites them. It also measures how well it does, on 200 real customer questions with
expert-written answers.

It uses two OpenAI models: `text-embedding-3-large` for embeddings and `gpt-6.1-sol` for answers
(both set in [config.yaml](config.yaml)). The vector database, ChromaDB, runs locally.

## How it works

```
BUILD THE INDEX (once)

  rag_datasets/*.csv -> ingestion -> chunking -> embeddings -> vectordb
                        HTML to     900-char    OpenAI,       ChromaDB,
                        clean text  pieces      3,072 dims    saved to disk

ANSWER A QUESTION (every request)

  question -> embeddings -> vectordb -> prompts -> llm -> answer with [1][2] citations
              same model    5 nearest   sources +  gpt-6.1-sol
                            chunks      question
```

| Step | Code | What it does |
|---|---|---|
| Ingestion | [src/ingestion/loader.py](src/ingestion/loader.py) | Reads the CSVs. Converts each article's HTML into text that keeps headings, numbered steps, tables and tab labels ("Wix Editor" vs "Studio Editor"), which the dataset's flat text column loses. |
| Chunking | [src/chunking/chunker.py](src/chunking/chunker.py) | Splits articles into pieces of at most 900 characters (about 150 tokens on average), cutting at headings first, then lines, then sentences. Neighbouring pieces of a split section overlap by up to 150 characters. |
| Embeddings | [src/embeddings/embedder.py](src/embeddings/embedder.py) | Turns each chunk (prefixed with its article title) into a vector with OpenAI's `text-embedding-3-large`, in batches, and tracks the tokens billed. |
| Vector DB | [src/vectordb/vector_store.py](src/vectordb/vector_store.py) | Stores vectors, text and metadata in a persistent ChromaDB collection and finds the nearest chunks by cosine similarity. |
| Retrieval | [src/retrieval/retriever.py](src/retrieval/retriever.py) | Embeds the question and fetches the nearest chunks. Optionally a local cross-encoder then reranks a pool of 30 (off by default). |
| Prompts | [src/prompts/prompt_templates.py](src/prompts/prompt_templates.py) | The system prompt (answer only from the sources, cite them, say so when the answer is not there) and the judge's grading rubric. |
| LLM | [src/llm/llm_client.py](src/llm/llm_client.py) | The single place that calls the chat model (OpenAI Responses API), with token and cost accounting. |
| Pipeline | [src/pipeline.py](src/pipeline.py) | Wires the steps together: `build_index()`, `retrieve()`, `answer()`. |
| API | [src/api/routes.py](src/api/routes.py) | FastAPI endpoints: `/health`, `/retrieve`, `/ask`. |
| Web UI | [streamlit_app.py](streamlit_app.py) | A chat page in the browser, built with Streamlit. |
| Evaluation | [src/evaluation/](src/evaluation/) | Retrieval metrics, answer metrics and the LLM judge. |

## Project structure

```
├── README.md
├── requirements.txt
├── .env                    your OpenAI API key (not committed)
├── .env.example
├── .gitignore
├── config.yaml             models, chunk size, top-k, paths
├── download_datasets.py    fetches the datasets into rag_datasets/
├── main.py                 command-line entry point
├── streamlit_app.py        web UI
├── src/
│   ├── ingestion/loader.py
│   ├── chunking/chunker.py
│   ├── embeddings/embedder.py
│   ├── vectordb/vector_store.py
│   ├── retrieval/retriever.py
│   ├── prompts/prompt_templates.py
│   ├── llm/llm_client.py
│   ├── api/routes.py
│   ├── evaluation/metrics.py, evaluator.py
│   ├── utils/helpers.py
│   └── pipeline.py
├── tests/test_app.py
├── logs/app.log
├── rag_datasets/           downloaded CSVs (not committed)
├── vector_store/           ChromaDB files (not committed)
└── evaluation_results/     one folder per evaluation run
```

## Setup

Python 3.10 or newer.

**1. Create an environment and install the packages:**

```bash
conda create -n support_rag python=3.12 -y
conda activate support_rag
pip install -r requirements.txt
```

In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter** and pick `support_rag`, so
the editor and its terminal use the same environment. (Plain `python -m venv` works too; keep the
environment folder outside OneDrive.)

**2. Add your OpenAI key** to the `.env` file (create a key at https://platform.openai.com/api-keys):

```
OPENAI_API_KEY=sk-...
```

**3. Download the data and build the index:**

```bash
python download_datasets.py        # ~1 minute, writes rag_datasets/
python main.py ingest              # embeds 25,795 chunks from 8,921 documents
```

`ingest` sends 3.8 million tokens to the embeddings API, which costs about **$0.49** at
`text-embedding-3-large`'s price of $0.13 per million tokens. It prints the tokens and cost when it
finishes. If it is interrupted (network, rate limit), `python main.py ingest --resume` continues
where it stopped without paying for the same chunks twice. It only has to run again when the data
or the `chunking` / `embeddings` settings change.

## Usage

```bash
# Show what is retrieved for a question (embeds the question, no chat-model call)
python main.py search "How do I connect a domain I already own?"

# Full answer with citations
python main.py ask "How do I connect a domain I already own?"
python main.py ask                     # interactive mode

# Options for search / ask / evaluate
python main.py ask "..." --sources wixqa      # search only the Wix help center
python main.py ask "..." --top-k 8            # give the LLM more chunks
python main.py ask "..." --rerank             # add cross-encoder reranking
```

### Web UI

```bash
streamlit run streamlit_app.py         # opens http://localhost:8501 in your browser
```

A chat page: type a question (or click one of the suggestions) and get the cited answer, with the
sources it was written from in an expandable list underneath and the tokens and cost of the call.
The sidebar chooses which knowledge source to search and how many sources each answer gets. Stop
the server with `Ctrl+C` in the terminal.

### REST API

```bash
python main.py serve                   # http://127.0.0.1:8000
```

The easiest way to try it is the interactive page at http://127.0.0.1:8000/docs. From a second
terminal:

```powershell
# PowerShell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{ question = "How do I add a contact form?"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/retrieve -Method Post -ContentType "application/json" -Body $body
Invoke-RestMethod http://127.0.0.1:8000/ask      -Method Post -ContentType "application/json" -Body $body
```

```bash
# bash / macOS / Linux
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I add a contact form?", "top_k": 5}'
```

| Endpoint | Returns |
|---|---|
| `GET /health` | Status and the number of indexed chunks. |
| `POST /retrieve` | The retrieved sources only (no chat-model call). |
| `POST /ask` | The answer, the numbered sources it cites (title, URL, text, score), the model that served the request and the token counts. |

## Configuration

All settings live in [config.yaml](config.yaml). The ones you are most likely to change:

| Setting | Default | Effect |
|---|---|---|
| `llm.model` | `gpt-6.1-sol` | The answering model. `gpt-6-luna` is 20x cheaper, `gpt-6-astra` the most capable at 5x the price. |
| `llm.effort` | `medium` | Reasoning effort: `low`, `medium`, `high`, `xhigh`, `max`. Lower is faster and cheaper. |
| `embeddings.model` | `text-embedding-3-large` | Re-run `ingest` after changing. |
| `embeddings.dimensions` | null (3,072) | A smaller size such as 1024 makes the index 3x smaller (106 MB of vectors instead of 317 MB) for a small loss in accuracy. Re-run `ingest` after changing. |
| `chunking.chunk_size` / `chunk_overlap` | 900 / 150 | Size of the pieces that get embedded. Re-run `ingest` after changing. |
| `retrieval.top_k` | 5 | Chunks handed to the LLM. |
| `retrieval.sources` | null | `[wixqa]` or `[bitext]` to search one knowledge source only. |
| `retrieval.rerank.enabled` | false | Local cross-encoder reranking. Needs an extra install: `pip install sentence-transformers` (pulls in PyTorch). |
| `data.bitext.enabled` | true | Whether Bitext is indexed at all. |
| `evaluation.judge_model` | `gpt-6-luna` | Model that grades answers. Kept different from `llm.model` so the generator does not grade itself. |

Prices per million tokens (input / output) at the time of writing: `gpt-6-luna` $0.10 / $0.50,
`gpt-6.1-sol` $2 / $10, `gpt-6-astra` $10 / $50. The table in
[llm_client.py](src/llm/llm_client.py) is what the cost reports use; update it if prices change.

## Evaluation

```bash
python main.py evaluate --retrieval-only          # retrieval metrics only (costs a fraction of a cent)
python main.py evaluate --check-judge             # 12 judge calls: is the judge trustworthy?
python main.py evaluate --limit 5                 # pilot: retrieval + generation on 5 questions
python main.py evaluate                           # the full 200 questions
python main.py evaluate --dataset simulated       # or: synthetic, bitext
```

Results are written to `evaluation_results/<dataset>/` (use `--run-name` to keep several runs side
by side): summaries as JSON, plus one row per question so any score can be traced back to the
question, the retrieved chunks, the answer and the judge's reasoning.

**Cost.** Measured with the default models: about $0.0046 per question ($0.0043 for the answer,
$0.0004 for the judge), so roughly $0.92 for all 200. If you change models, run the 5-question
pilot first; it prints the tokens and dollars actually used, and you can multiply by 40.

**Test sets.** `expertwritten` (the default) is 200 real customer questions with answers written
by Wix support experts and the ids of the articles that answer them. `simulated` is 200 questions
distilled from support conversations. `synthetic` is 6,221 LLM-generated questions, one per
article. `bitext` is 270 customer messages held out of the index, scored on whether retrieval
finds responses with the same intent.

### Stage 1: retrieval

Was the right article retrieved? Chunks are collapsed to the articles they come from and compared
with the ground-truth article ids.

| Metric | Meaning |
|---|---|
| `hit@k` | Share of questions with at least one correct article in the top k. |
| `recall@k` | Share of the correct articles found in the top k (some questions have 2-3). |
| `mrr` | Mean of 1 / rank of the first correct article. 1.0 means always ranked first. |
| `ndcg@k` | Like recall, but a correct article counts for more the higher it is ranked. |
| `context_recall` | Share of the correct articles present in the 5 chunks actually given to the LLM. |

**Results** on the 200 expert-written questions (2026-10-08, default settings). Each figure has
a 95% confidence interval of ±0.04 to ±0.07.

| Embeddings | Searched | hit@1 | hit@5 | hit@10 | MRR | context recall |
|---|---|---|---|---|---|---|
| **`text-embedding-3-large`** (current) | Wix + Bitext | **0.480** | **0.840** | **0.910** | **0.644** | **0.731** |
| `text-embedding-3-large` | Wix only (`--sources wixqa`) | 0.490 | 0.850 | 0.920 | 0.654 | 0.738 |
| `all-MiniLM-L6-v2`, local (baseline) | Wix + Bitext | 0.370 | 0.745 | 0.835 | 0.535 | 0.596 |
| `all-MiniLM-L6-v2`, local | Wix only | 0.390 | 0.785 | 0.870 | 0.562 | 0.632 |
| `all-MiniLM-L6-v2`, local + rerank | Wix + Bitext | 0.395 | 0.770 | 0.855 | 0.551 | 0.648 |

The baseline rows were measured during development with a small local embedding model, before the
switch to OpenAI embeddings. What the comparison shows, using paired differences over the same 200
questions:

- **The embedding model matters most.** `text-embedding-3-large` beats the local model on every
  metric by more than the noise: hit@5 +0.095 (±0.058), context recall +0.135 (±0.059).
- **Bitext no longer gets in the way.** With the local model, sharing the index with Bitext cost
  0.04 of hit@5. With `text-embedding-3-large` the difference is 0.01, inside the noise, so
  searching everything is a safe default.
- **Reranking did not earn its cost.** On the local index it added about a second per query for a
  gain inside the noise, so it is off by default. It has not been measured on the new index.

In 16% of questions the right article is still not in the top 5, and `context_recall` of 0.73 means
roughly a quarter of the needed articles never reach the LLM. That is the ceiling on answer quality
and the first thing to improve (larger chunks, more sources per answer, or reranking).

Other test sets, same settings:

| Test set | Questions | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| `simulated` (distilled from support conversations) | 200 | 0.435 | 0.770 | 0.845 | 0.584 |
| `bitext` (held-out customer messages, matched on intent) | 270 | 0.959 | 0.996 | 0.996 | 0.978 |

Per-run summaries are in `evaluation_results/`.

### Stage 2: generation

Is the final answer right? Every question goes through the same `pipeline.answer()` path the CLI
and API use, and the answer is scored three ways:

| Metric | How | Meaning |
|---|---|---|
| `correctness` | LLM judge | Does the answer convey the key facts and steps of the expert answer? correct = 1, partially correct = 0.5, incorrect = 0. |
| `faithfulness` | LLM judge | 1 if every claim in the answer is supported by the retrieved sources (no hallucination). The judge lists each unsupported claim. |
| `relevance` | LLM judge | 1 if the answer addresses what was asked. |
| `abstained` | LLM judge | Share of questions where the system said it could not answer. |
| `semantic_similarity` | embeddings | Cosine similarity between answer and expert answer. |
| `token_f1`, `rouge_l` | word overlap | Shared words with the expert answer. Cheap and deterministic, but low even for good answers that are worded differently, so read them as a trend, not a grade. |

The judge is a second model working from a fixed rubric
([prompt_templates.py](src/prompts/prompt_templates.py)) and returning structured JSON, with its
reasoning saved next to every verdict. `--check-judge` feeds it answers whose grade is known in
advance (the expert answer, an empty answer, "I don't know", the answer to a different question)
and reports whether it grades each one as expected. Run it before trusting judge scores; if the
default judge fails checks, set `evaluation.judge_model` to a stronger model.

Things the evaluation is careful about:

- A question that fails because of the API (rate limit, timeout) is written to
  `generation_errors.jsonl` and left out of the averages, not scored as a wrong answer.
  `--resume` re-runs only those.
- Answers cut off at the token limit and refusals are counted separately for the same reason.
- Every average is reported with a 95% confidence interval. With 200 questions that is about
  ±0.07 on a yes/no metric, so differences smaller than that are noise.
- Token counts come from the API response, and the cost of the answering model and of the judge
  are reported separately.

**Pilot results.** A 5-question pilot (`--limit 5`) with `gpt-6.1-sol` answering and `gpt-6-luna`
judging scored correctness 0.80, faithfulness 1.00 and relevance 1.00, with no abstentions. The
judge passed all 12 `--check-judge` probes. Five questions are far too few to draw conclusions
from (the interval on correctness is ±0.24); the pilot shows that the pipeline and the judge work
and what a full run costs. Run `python main.py evaluate` for the 200-question numbers.

## Tests

```bash
python -m pytest
```

42 tests cover every stage, including the web page. OpenAI is replaced by small fakes, so the
suite runs in seconds with no datasets, index or API key.

## Notes and limitations

- **Bitext in the same index.** Bitext is generic e-commerce support (orders, refunds, invoices)
  whose answers are templates with placeholders such as `{{Order Number}}`; it is not about Wix.
  Only 100 examples per intent are indexed, since the full set is ~1,000 near-identical rewordings
  of each. Use `--sources wixqa` or `retrieval.sources` for a Wix-only assistant.
- **Single-turn.** Each question is answered on its own; there is no conversation memory.
- **Split lists.** Where an article interrupts a numbered list with a screenshot, the numbering
  restarts in the extracted text. The steps stay in the right order.
- **Data leaves your machine.** Chunks, questions and prompts are sent to OpenAI. That is fine for
  these public datasets; think about it before indexing private documents.
