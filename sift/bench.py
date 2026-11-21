"""`sift bench`: how well each search mode finds what a set of questions should find, on your own documents.

A bench file is JSON: a list of cases, each a query and the documents it should find, by their ids as search results
show them (or as `get` takes them): {"query": "...", "relevant": ["notes/plan.md", ...]}, or with grades of relevance,
{"relevant": {"notes/plan.md": 2, "notes/old-plan.md": 1}}. The measures are those of the BEIR and TREC benchmarks."""

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .hits import Hit

if TYPE_CHECKING:
    from .search import Index

MODES = ("search", "vsearch", "query")
K = 10
Grades = dict[str, float]  # document id: grade of relevance
Json = list[Any] | dict[str, Any] | str | float | None  # a parsed JSON value, its depths unchecked
Case = tuple[str, Grades]  # a query and the documents it should find


def ndcg(ranked: list[str], relevant: Grades, k: int = K) -> float:
    """Normalised discounted cumulative gain of the first k results, with graded relevance."""
    dcg = sum((2 ** relevant.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:k])
    )
    return dcg / ideal


def reciprocal_rank(ranked: list[str], relevant: Grades, k: int = K) -> float:
    """1 / the rank of the first relevant result among the first k, else 0."""
    return next((1 / (i + 1) for i, doc in enumerate(ranked[:k]) if doc in relevant), 0.0)


def recall(ranked: list[str], relevant: Grades, k: int = K) -> float:
    """The share of the relevant documents among the first k results."""
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def _grades(relevant: Json) -> Grades | None:
    """A case's relevant documents as {id: grade}: a list of ids, each of grade 1, or ids with grades. Grade 0 means
    read and found irrelevant, as in TREC's judgements, so those stay out. None if nothing is relevant."""
    if isinstance(relevant, list) and all(isinstance(id, str) for id in relevant):
        relevant = dict.fromkeys(relevant, 1)
    if not isinstance(relevant, dict):
        return None
    if not all(isinstance(g, int | float) and not isinstance(g, bool) and g >= 0 for g in relevant.values()):
        return None
    return {id: grade for id, grade in relevant.items() if grade > 0} or None


def load(path: str | Path, index: "Index") -> list[Case]:
    """The cases of a bench file as (query, {document id: grade}), each id as the index knows it."""
    cases: Json = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f'{path} must hold a list of cases: {{"query": ..., "relevant": [document ids]}}')
    loaded: list[Case] = []
    for n, case in enumerate(cases, 1):
        entry: Json = case
        grades = _grades(entry.get("relevant")) if isinstance(entry, dict) else None
        if not (grades and isinstance(case.get("query"), str)):
            raise ValueError(
                f"case {n} needs a query and the ids of the documents it should find: a list of them, or a dict of "
                "them with grades above 0"
            )
        try:
            loaded.append((case["query"], {index.document(id)["id"]: grade for id, grade in grades.items()}))
        except LookupError as e:
            raise LookupError(f"case {n}: {e}") from None
    return loaded


def run(index: "Index", cases: list[Case], **options: Any) -> dict[str, dict[str, float]]:
    """Each mode's mean nDCG@10, MRR@10 and Recall@10 over the cases, and its time per query in ms. `options` go
    to every search, e.g. collection."""
    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        index.forget()  # each mode embeds and searches as if first asked: vsearch's embeddings don't time query
        scores: list[tuple[float, float, float]] = []
        start = time.perf_counter()
        for query, relevant in cases:
            ranked, graded = _found(getattr(index, mode)(query, K, compact=True, **options), relevant)
            scores.append((ndcg(ranked, graded), reciprocal_rank(ranked, graded), recall(ranked, graded)))
        elapsed = time.perf_counter() - start
        means = [sum(s[i] for s in scores) / len(scores) for i in range(3)]
        results[mode] = dict(zip(("nDCG@10", "MRR@10", "Recall@10"), means)) | {"ms": 1000 * elapsed / len(cases)}
    return results


def _found(hits: Iterable[Hit], relevant: Grades) -> tuple[list[str], Grades]:
    """The ids the hits stand for, and the case's grades as they count: a hit is the one of it and its copies in other
    collections (`also`), the same paper, that the case grades best, and takes the best grade of them all."""
    relevant, ranked = dict(relevant), list[str]()
    for hit in hits:
        copies = [hit["id"], *hit.get("also", ())]
        best = max(copies, key=lambda id: relevant.get(id, 0))
        if graded := [relevant.pop(id) for id in copies if id in relevant]:
            relevant[best] = max(graded)
        ranked.append(best)
    return ranked, relevant
