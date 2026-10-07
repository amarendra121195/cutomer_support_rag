"""Evaluation metrics. Every metric is a pure function returning a score between 0 and 1."""
import math
import re
import statistics
from collections import Counter

# ---------------------------------------------------------------------------
# Retrieval metrics. `ranked` is the list of distinct retrieved items, best first;
# `relevant` is the set of items that are correct for the question.
# ---------------------------------------------------------------------------


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 if at least one relevant item is in the top k, else 0."""
    return float(any(item in relevant for item in ranked[:k]))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item, 0 if none was retrieved. Its mean over questions is MRR."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain: like recall, but relevant items count for more near the top."""
    gain = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return gain / ideal


# ---------------------------------------------------------------------------
# Answer metrics that compare the generated answer with the reference answer.
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    """Lower-cased words, ignoring URLs and citation markers like [1]."""
    text = re.sub(r"https?://[^\s)\]]+|\[\d+\]", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def _f1(matched: int, n_prediction: int, n_reference: int) -> float:
    if matched == 0:
        return 0.0
    precision, recall = matched / n_prediction, matched / n_reference
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    """Word-overlap F1 between answer and reference, ignoring word order."""
    predicted, expected = _tokens(prediction), _tokens(reference)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return _f1(overlap, len(predicted), len(expected))


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1: based on the longest common subsequence of words, so matching order is rewarded."""
    predicted, expected = _tokens(prediction), _tokens(reference)
    previous = [0] * (len(expected) + 1)
    for word in predicted:
        current = [0]
        for j, other in enumerate(expected, start=1):
            current.append(previous[j - 1] + 1 if word == other else max(previous[j], current[j - 1]))
        previous = current
    return _f1(previous[-1], len(predicted), len(expected))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def mean_ci(values: list[float]) -> dict:
    """Mean with the half-width of its 95% confidence interval (normal approximation).

    The interval says how far the mean could move from sampling different questions; two
    configurations whose intervals overlap heavily have not been shown to differ.
    """
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    mean = statistics.fmean(values)
    ci95 = 1.96 * statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return {"mean": round(mean, 4), "ci95": None if ci95 is None else round(ci95, 4), "n": n}
