"""The best documents for a keyword query from the postings (sift.topk): exactly what scoring every posting finds, and
what FTS5's bm25 finds, for random expressions of AND, OR and NOT, through changes, across stores, with filters."""

import math
import random
from pathlib import Path

import numpy as np
import pytest

from sift import indexing, postings, topk
from sift.catalog import Catalog

from .conftest import PROSE_WORDS, write, write_prose

TERMS = [w.lower() for w in PROSE_WORDS[:30]] + ["muller", "absent"]  # as FTS5 tokenizes them; one in no chunk


def expression(rng, depth=0):
    """A random expression of terms, AND, OR and NOT."""
    if depth >= 2 or rng.random() < 0.35:
        return ("term", rng.choice(TERMS))
    kind = rng.choice(["and", "or", "or", "not"])
    if kind == "not":
        return ("not", expression(rng, depth + 1), expression(rng, depth + 1))
    return (kind, [expression(rng, depth + 1) for _ in range(rng.randint(2, 3))])


def fts(node):
    """The expression as an FTS5 query."""
    if node[0] == "term":
        return f'"{node[1]}"'
    if node[0] == "not":
        return f"({fts(node[1])} NOT {fts(node[2])})"
    return "(" + f" {node[0].upper()} ".join(fts(child) for child in node[1]) + ")"


def brute(readers, query, k, by_document=True, allowed=None):
    """Every live posting scored: the reference, written apart from topk."""
    named = topk.terms_of(query)
    chunks = sum(r.chunks for r in readers)
    average = sum(r.tokens for r in readers) / chunks
    score, holding, owner = {}, {}, {}
    df = dict.fromkeys(named, 0)
    for reader in readers:
        for t in set(named):
            for part in reader.term(t):
                df[t] += part.df
                ids, tf = part.everything()
                alive = reader.alive(part.segment, ids)
                ids, tf = ids[alive], tf[alive]
                lengths, docs = reader.lengths(part.segment, ids), reader.documents(ids)
                term_scores = postings.term_scores(tf, lengths, average)  # as bm25 (see test_postings), to the bit
                for chunk, s, doc in zip(ids.tolist(), term_scores.tolist(), docs.tolist()):
                    holding.setdefault(chunk, set()).add(t)
                    owner[chunk] = doc
                    score.setdefault(chunk, {})[t] = s
    weight = {t: max(math.log((chunks - df[t] + 0.5) / (df[t] + 0.5)), 1e-6) for t in set(named)}

    def counts(node, held):
        """(whether it holds, the terms that count): FTS5 clears the terms of an AND or NOT that fails."""
        if node[0] == "term":
            return node[1] in held, [node[1]] if node[1] in held else []
        if node[0] == "not":
            left, terms = counts(node[1], held)
            ok = left and not counts(node[2], held)[0]
            return ok, terms if ok else []
        parts = [counts(child, held) for child in node[1]]
        ok = (all if node[0] == "and" else any)(p[0] for p in parts)
        terms = [t for _, ts in parts for t in ts]
        return ok, terms if ok or node[0] == "or" else []

    found = []
    for c, held in holding.items():
        ok, terms = counts(query, held)
        if ok and (allowed is None or allowed(np.array([c]))[0]):
            found.append((sum(weight[t] * score[c][t] for t in terms), c))
    found.sort(key=lambda f: (-f[0], f[1]))
    if by_document:
        seen, best = set(), []
        for s, c in found:
            if owner[c] not in seen:
                seen.add(owner[c])
                best.append((s, c))
        found = best
    return [(c, s) for s, c in found[:k]]


TIE = 1e-12  # sums of the same terms in another order differ in their last bits (Python's sum() compensates)


def same(got, more, k, query=None):
    """The same results as the reference's best k, of the `more` it ranked: scores equal to rounding, and each chunk
    the reference's at its place, or one tied with it there (ties can go either way, at the k-th place too)."""
    want = more[:k]
    assert len(got) == len(want), query
    assert np.allclose([s for _, s in got], [s for _, s in want], rtol=1e-9), (query, got, want)
    ranked = dict(more)
    for (chunk, _), (expected, expected_score) in zip(got, want):
        if chunk != expected:
            assert chunk in ranked, (query, got, want)
            assert abs(ranked[chunk] - expected_score) <= TIE * abs(expected_score), (query, got, want)


def reader(catalog, name="prose"):
    return postings.Reader(catalog.store(name))


@pytest.fixture
def built(prose):
    catalog, _ = prose
    postings.build(catalog.store("prose"), catalog.get("prose").store)
    return catalog


@pytest.mark.usefixtures("small_lists")
@pytest.mark.parametrize("seed", range(6))
def test_finds_what_scoring_every_posting_finds(built, seed):
    rng = random.Random(seed)
    readers = [reader(built)]
    for _ in range(25):
        query, k = expression(rng), rng.choice([1, 3, 10, 50])
        by_document = rng.random() < 0.7
        same(topk.search(readers, query, k, by_document), brute(readers, query, k + 20, by_document), k, query)


@pytest.mark.parametrize("seed", range(3))
def test_finds_what_fts5_finds(built, seed):
    # On postings just built, the statistics are FTS5's own: the same documents, the same scores.
    rng = random.Random(100 + seed)
    db = built.store("prose")
    sql = """WITH p AS MATERIALIZED (SELECT fts.rowid passage, c.doc doc, -bm25(fts) score FROM fts
             JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?)
             SELECT passage, max(score) FROM p GROUP BY doc ORDER BY 2 DESC, 1 LIMIT ?"""
    for _ in range(25):
        query, k = expression(rng), rng.choice([1, 5, 20])
        more = [tuple(r) for r in db.execute(sql, (fts(query), k + 20))]
        same(topk.search([reader(built)], query, k), more, k, query)


@pytest.mark.usefixtures("small_lists")
def test_through_changes_and_deltas(built, monkeypatch):
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    root, rng = Path(built.get("prose").path), random.Random(9)
    for round_ in range(3):
        for i in rng.sample(range(120), 8):
            write(root / f"n{i:03}.md", f"# changed {round_}\n\nw1 w2 w{i % 30} muller w3\n")
        indexing.index(built)
        postings.update(built.store("prose"), built.get("prose").store)
        readers = [reader(built)]
        assert len(readers[0].segments) == round_ + 2
        for _ in range(15):
            query, k = expression(rng), rng.choice([1, 10, 40])
            same(topk.search(readers, query, k), brute(readers, query, k + 20), k, query)


@pytest.mark.usefixtures("small_lists")
def test_across_stores_with_their_statistics_as_one(built, tmp_path):
    write_prose(tmp_path / "more", 8, notes=60)
    built.add(tmp_path / "more", "more")
    indexing.index(built)
    postings.build(built.store("more"), built.get("more").store)
    readers = [reader(built), reader(built, "more")]
    rng = random.Random(4)
    for _ in range(20):
        query, k = expression(rng), rng.choice([1, 10, 30])
        same(topk.search(readers, query, k), brute(readers, query, k + 20), k, query)


@pytest.mark.usefixtures("small_lists")
def test_a_filter_is_kept_to(built):
    readers, rng = [reader(built)], random.Random(5)

    def allowed(chunks):
        return chunks % 3 == 0

    for _ in range(20):
        query, k = expression(rng), rng.choice([1, 10])
        same(topk.search(readers, query, k, allowed=allowed), brute(readers, query, k + 20, allowed=allowed), k, query)


def test_a_term_in_no_chunk(built):
    readers = [reader(built)]
    assert topk.search(readers, ("term", "absent"), 10) == []
    assert topk.search(readers, ("and", [("term", "w1"), ("term", "absent")]), 10) == []
    assert topk.search(readers, ("or", [("term", "w1"), ("term", "absent")]), 3) == topk.search(
        readers, ("term", "w1"), 3
    )


def test_no_chunks_find_nothing(tmp_path, home):
    catalog = Catalog(home)
    (tmp_path / "empty").mkdir()
    catalog.add(tmp_path / "empty", "prose")
    indexing.index(catalog)
    postings.build(catalog.store("prose"), catalog.get("prose").store)
    assert topk.search([reader(catalog)], ("term", "w1"), 10) == []


@pytest.mark.usefixtures("small_lists")
def test_terms_in_ranges_apart(tmp_path, home):
    # One term's ranges hold none of the other's: each long list read only where it occurs.
    root = tmp_path / "apart"
    for i in range(80):
        write(root / f"n{i:03}.md", f"# n{i}\n\n{'alpha' if i < 40 else 'beta'} words {'alpha ' * (i % 3)}\n")
    catalog = Catalog(home)
    catalog.add(root, "apart")
    indexing.index(catalog)
    readers = [reader(catalog, "apart")]
    assert all(t.long for term in ("alpha", "beta") for t in readers[0].term(term))
    query = ("or", [("term", "alpha"), ("term", "beta")])
    same(topk.search(readers, query, 100), brute(readers, query, 120), 100, query)


@pytest.mark.usefixtures("small_lists")
def test_most_ranges_are_never_read(built, monkeypatch):
    # The point of the bounds: a common term's best few documents are found having scored a few of its ranges.
    scored = []
    real = topk._Store._score
    monkeypatch.setattr(topk._Store, "_score", lambda self, group: scored.append(len(group)) or real(self, group))
    monkeypatch.setattr(topk, "FIRST_BATCH", 2)  # this collection is tiny: a few ranges a batch
    readers = [reader(built)]
    (common,) = readers[0].term("w0")
    ranges = len(common.ranges()[0])
    topk.search(readers, ("term", "w0"), 1)
    assert sum(scored) < ranges / 3
