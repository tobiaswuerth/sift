"""`sift bench`: how well each search mode finds the documents a set of questions should find, on your own index."""

import json
import math

import pytest

from sift import bench
from sift.search import Index

CASES = [
    {"query": "reward shaping reinforcement", "relevant": ["notes/rl.md"]},
    {"query": "painting robot brush strokes", "relevant": {"notes/brush.md": 2}},  # a file name, as `get` takes
]


def test_metrics():
    assert bench.ndcg(["a", "b"], {"b": 1}) == pytest.approx(1 / math.log2(3))
    assert bench.ndcg(["b", "a"], {"a": 2, "b": 1}) < 1 == bench.ndcg(["a", "b"], {"a": 2, "b": 1})
    assert bench.ndcg([], {"b": 1}) == 0
    assert bench.reciprocal_rank(["a", "b"], {"b": 1}) == 0.5
    assert bench.reciprocal_rank(["a"] * 10 + ["b"], {"b": 1}) == 0  # beyond the first 10
    assert bench.recall(["a", "b", "c"], {"b": 1, "x": 1}) == 0.5


@pytest.fixture
def cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(CASES), encoding="utf-8")
    return path


def test_bench_measures_every_mode(embedded, cases):
    index = Index(embedded[0])
    results = bench.run(index, bench.load(cases, index))
    assert list(results) == ["search", "vsearch", "query"]
    for scores in results.values():
        assert scores["nDCG@10"] == scores["MRR@10"] == scores["Recall@10"] == 1.0
        assert scores["ms"] >= 0


def test_bench_takes_ids_as_get_does(embedded, cases):
    index = Index(embedded[0])
    assert [relevant for _, relevant in bench.load(cases, index)] == [{"notes/rl.md": 1}, {"notes/robots/brush.md": 2}]


def test_documents_graded_0_are_not_relevant(embedded, tmp_path):
    # As in TREC's judgements, which list documents read and found irrelevant with grade 0.
    path = tmp_path / "graded.json"
    path.write_text('[{"query": "reward", "relevant": {"notes/rl.md": 1, "notes/plain.txt": 0}}]', encoding="utf-8")
    assert bench.load(path, Index(embedded[0])) == [("reward", {"notes/rl.md": 1})]


@pytest.mark.parametrize(
    "content, error",
    [
        ('[{"query": "x", "relevant": ["notes/missing.md"]}]', "case 1: unknown document"),
        ('[{"query": "x", "relevant": []}]', "case 1 needs"),
        ('[{"query": "x", "relevant": "notes/rl.md"}]', "case 1 needs"),  # no list: it was split into letters
        ('[{"query": "x", "relevant": [["notes/rl.md"]]}]', "case 1 needs"),
        ('[{"query": "x", "relevant": {"notes/rl.md": 0}}]', "case 1 needs"),  # nothing relevant: nDCG divided by 0
        ('[{"query": "x", "relevant": {"notes/rl.md": "high"}}]', "case 1 needs"),
        ('[{"query": "x", "relevant": {"notes/rl.md": -1}}]', "case 1 needs"),
        ('[{"relevant": ["notes/rl.md"]}]', "case 1 needs"),
        ('{"query": "x"}', "a list of cases"),
        ("[]", "a list of cases"),
    ],
)
def test_bad_bench_files_are_clear_errors(embedded, tmp_path, content, error):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, LookupError), match=error):
        bench.load(path, Index(embedded[0]))
