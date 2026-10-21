"""`sift bench`: how well each search mode finds what a set of questions should find, on your own documents.

A bench file is JSON: a list of cases, each a query and the documents it should find, by their ids as search results
show them (or as `get` takes them): {"query": "...", "relevant": ["notes/plan.md", ...]}, or with grades of relevance,
{"relevant": {"notes/plan.md": 2, "notes/old-plan.md": 1}}. The measures are those of the BEIR and TREC benchmarks."""

import json
import math
import time
from pathlib import Path

MODES = ("search", "vsearch", "query")
K = 10


def ndcg(ranked, relevant, k=K):
    """Normalised discounted cumulative gain of the first k results, with graded relevance."""
    dcg = sum((2 ** relevant.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:k])
    )
    return dcg / ideal


def reciprocal_rank(ranked, relevant, k=K):
    """1 / the rank of the first relevant result among the first k, else 0."""
    return next((1 / (i + 1) for i, doc in enumerate(ranked[:k]) if doc in relevant), 0.0)


def recall(ranked, relevant, k=K):
    """The share of the relevant documents among the first k results."""
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def _grades(relevant):
    """A case's relevant documents as {id: grade}: a list of ids, each of grade 1, or ids with grades. Grade 0 means
    read and found irrelevant, as in TREC's judgements, so those stay out. None if nothing is relevant."""
    if isinstance(relevant, list) and all(isinstance(id, str) for id in relevant):
        relevant = dict.fromkeys(relevant, 1)
    if not isinstance(relevant, dict):
        return None
    if not all(isinstance(g, int | float) and not isinstance(g, bool) and g >= 0 for g in relevant.values()):
        return None
    return {id: grade for id, grade in relevant.items() if grade > 0} or None


def load(path, index):
    """The cases of a bench file as (query, {document id: grade}), each id as the index knows it."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f'{path} must hold a list of cases: {{"query": ..., "relevant": [document ids]}}')
    loaded = []
    for n, case in enumerate(cases, 1):
        grades = _grades(case.get("relevant")) if isinstance(case, dict) else None
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


def run(index, cases, **options):
    """Each mode's mean nDCG@10, MRR@10 and Recall@10 over the cases, and its time per query in ms. `options` go
    to every search, e.g. collection."""
    results = {}
    for mode in MODES:
        scores, start = [], time.perf_counter()
        for query, relevant in cases:
            ranked = [hit["id"] for hit in getattr(index, mode)(query, K, compact=True, **options)]
            scores.append((ndcg(ranked, relevant), reciprocal_rank(ranked, relevant), recall(ranked, relevant)))
        elapsed = time.perf_counter() - start
        means = [sum(s[i] for s in scores) / len(scores) for i in range(3)]
        results[mode] = dict(zip(("nDCG@10", "MRR@10", "Recall@10"), means)) | {"ms": 1000 * elapsed / len(cases)}
    return results
