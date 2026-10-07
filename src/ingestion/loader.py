"""Ingestion: load the raw CSV datasets into documents and evaluation questions."""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from src.utils.helpers import resolve_path

logger = logging.getLogger(__name__)

WIXQA_QUESTION_SETS = ("expertwritten", "simulated", "synthetic")
EVAL_DATASETS = (*WIXQA_QUESTION_SETS, "bitext")

_DROPPED_TAGS = ["script", "style", "img", "svg", "iframe", "video", "figure"]
_BLOCK_TAGS = ["div", "p", "ul", "ol", "li", "table", "pre", "hr", "blockquote"]


@dataclass
class Document:
    """One knowledge-base entry before chunking."""
    doc_id: str
    title: str
    text: str
    source: str                         # "wixqa" or "bitext"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalExample:
    """One test question with its ground truth."""
    example_id: str
    question: str
    reference_answer: str
    relevant: list[str]                 # values of `match_field` that count as a correct retrieval
    match_field: str = "doc_id"         # chunk metadata field compared against `relevant`


def html_to_text(html: str) -> str:
    """Convert help-article HTML to plain text, keeping headings and list structure.

    Headings become "## ..." lines and list items "- ..." / "1. ..." lines, which gives the
    chunker natural places to cut and keeps step-by-step instructions readable for the LLM.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROPPED_TAGS):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Tabbed sections (e.g. "Wix Editor" / "Studio Editor") list all tab names first and the
    # panels after. Put each name above its own panel so the steps stay attributable.
    for tabs in soup.select(".tabs-wrapper"):
        labels = tabs.select(":scope > .headings-wrapper > .tab-heading")
        panels = tabs.select(":scope > .tab-contents-wrapper > .tab-content")
        for label, panel in zip(labels, panels):
            panel.insert_before(f"\n### {label.get_text(' ', strip=True)}\n")
        for wrapper in tabs.select(":scope > .headings-wrapper"):
            wrapper.decompose()

    for row in soup.find_all("tr"):
        if row.find_parent("tr"):       # nested table: already folded into the outer row
            continue
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        row.replace_with("\n" + " | ".join(cell for cell in cells if cell) + "\n")

    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        if not heading.get_text(strip=True):
            heading.decompose()
            continue
        heading.insert_before("\n## " if heading.name in ("h1", "h2", "h3") else "\n### ")
        heading.insert_after("\n")

    for lst in soup.find_all(["ul", "ol"]):
        number = 0
        for item in lst.find_all("li", recursive=False):
            if not item.get_text(strip=True):
                item.decompose()
                continue
            number += 1
            item.insert_before(f"\n{number}. " if lst.name == "ol" else "\n- ")

    for block in soup.find_all(_BLOCK_TAGS):
        block.insert_before("\n")
        block.insert_after("\n")

    lines = [re.sub(r"\s+", " ", line).strip() for line in soup.get_text().split("\n")]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"^(-|\d+\.|#{2,3})\n", r"\1 ", text, flags=re.M)   # re-attach markers to their line
    return re.sub(r"\n(?=#{2,3} )", "\n\n", text)                     # blank line before headings


def parse_article_ids(raw: Any) -> list[str]:
    """Extract article ids from WixQA's `article_ids` column.

    The column holds the printed form of a numpy array ("['id1'\\n 'id2']"), not valid JSON.
    """
    return re.findall(r"[0-9a-f]{64}", str(raw))


def _data_file(config: dict, name: str) -> Path:
    path = resolve_path(config["data"]["dir"]) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python download_datasets.py` first.")
    return path


def load_wixqa_documents(config: dict) -> list[Document]:
    """Load the Wix help-center articles (the main knowledge base)."""
    df = pd.read_csv(_data_file(config, config["data"]["wixqa"]["kb_file"])).fillna("")
    documents = []
    for row in df.itertuples(index=False):
        # A few articles are only an image or embed; fall back to the dataset's flat text.
        text = html_to_text(row.html_content) or row.contents.strip() or row.title
        documents.append(Document(
            doc_id=row.id,
            title=row.title,
            text=text,
            source="wixqa",
            metadata={"url": row.url, "article_type": row.article_type},
        ))
    logger.info("Loaded %d WixQA articles", len(documents))
    return documents


def _split_bitext(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split Bitext into rows to index and held-out rows to test with."""
    cfg = config["data"]["bitext"]
    df = pd.read_csv(_data_file(config, cfg["file"])).drop_duplicates("instruction")
    shuffled = df.sample(frac=1, random_state=cfg["seed"])
    rank = shuffled.groupby("intent").cumcount()
    n_eval = cfg["eval_per_intent"]
    held_out = shuffled[rank < n_eval]
    indexed = shuffled[(rank >= n_eval) & (rank < n_eval + cfg["max_per_intent"])]
    return indexed, held_out


def load_bitext_documents(config: dict) -> list[Document]:
    """Load a per-intent sample of Bitext question/response pairs as FAQ-style documents."""
    indexed, _ = _split_bitext(config)
    documents = [
        Document(
            doc_id=f"bitext-{index}",
            title=f"{row.category.title()}: {row.intent.replace('_', ' ')}",
            text=f"Customer question: {row.instruction}\nSupport answer: {row.response}",
            source="bitext",
            metadata={"category": row.category, "intent": row.intent},
        )
        for index, row in zip(indexed.index, indexed.itertuples(index=False))
    ]
    logger.info("Loaded %d Bitext documents", len(documents))
    return documents


def load_documents(config: dict) -> list[Document]:
    """Load every enabled knowledge source."""
    documents = load_wixqa_documents(config)
    if config["data"]["bitext"]["enabled"]:
        documents += load_bitext_documents(config)
    return documents


def load_eval_examples(config: dict, dataset: str, limit: int | None = None) -> list[EvalExample]:
    """Load test questions with ground truth for one of EVAL_DATASETS."""
    if dataset in WIXQA_QUESTION_SETS:
        df = pd.read_csv(_data_file(config, f"wixqa_wixqa_{dataset}_train.csv"))
        examples = [
            EvalExample(
                example_id=f"{dataset}-{i}",
                question=row.question,
                reference_answer=row.answer,
                relevant=parse_article_ids(row.article_ids),
            )
            for i, row in enumerate(df.itertuples(index=False))
        ]
    elif dataset == "bitext":
        _, held_out = _split_bitext(config)
        examples = [
            EvalExample(
                example_id=f"bitext-{index}",
                question=row.instruction,
                reference_answer=row.response,
                relevant=[row.intent],
                match_field="intent",
            )
            for index, row in zip(held_out.index, held_out.itertuples(index=False))
        ]
    else:
        raise ValueError(f"Unknown evaluation dataset {dataset!r}. Choose from {EVAL_DATASETS}.")
    return examples[:limit] if limit else examples
