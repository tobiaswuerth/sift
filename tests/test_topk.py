"""The best documents for a keyword query from the postings (sift.topk): exactly what scoring every posting finds, and
what FTS5's bm25 finds, for random expressions of AND, OR and NOT, through changes, across stores, with filters."""

import collections
import itertools
import math
import random
from pathlib import Path

import numpy as np
import pytest

from sift import embed, indexing, postings, store, topk
from sift.catalog import Catalog
from sift.expression import tokenize

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
    if node[0] == "phrase":
        return '"' + " ".join(node[1]) + '"'
    if node[0] == "not":
        return f"({fts(node[1])} NOT {fts(node[2])})"
    return "(" + f" {node[0].upper()} ".join(fts(child) for child in node[1]) + ")"


def occurrences(tokens, phrase):
    """How often the phrase (tokens) occurs in tokens, overlapping occurrences each."""
    return sum(tuple(tokens[i : i + len(phrase)]) == phrase for i in range(len(tokens) - len(phrase) + 1))


def postings_of(reader, leaf):
    """(chunk ids, counts, lengths, df) of the live chunks holding a leaf, per segment: a term's from its postings, with
    the df the segment keeps; a phrase's from the chunks' texts, with the number of chunks holding it."""
    if isinstance(leaf, str):
        for part in reader.term(leaf):
            places, tf = part.everything()
            alive = reader.alive(part.segment, places)
            places, tf = places[alive], tf[alive]
            yield reader.chunks_at(places), tf, reader.lengths(part.segment, places), part.df
        return
    rows = reader.db.execute("SELECT id, title, text, meta FROM chunk_text")
    counts = {chunk: sum(occurrences(tokenize(c), leaf) for c in columns if c) for chunk, *columns in rows}
    ids = np.array([chunk for chunk, n in counts.items() if n], np.int64)
    sizes = postings.sizes(reader.db, ids)
    yield ids, np.array([counts[c] for c in ids.tolist()]), np.array([sizes[c] for c in ids.tolist()]), len(ids)


def brute(readers, query, k, by_document=True, allowed=None):
    """Every live posting scored, and every phrase counted in every chunk's text: the reference, written apart from
    topk."""
    named = topk.terms_of(query)
    chunks = sum(r.chunks for r in readers)
    average = sum(r.tokens for r in readers) / chunks
    score, holding, owner = {}, {}, {}
    df = dict.fromkeys(named, 0)
    for reader in readers:
        for t in set(named):
            for ids, tf, lengths, n in postings_of(reader, t):
                df[t] += n
                term_scores = postings.term_scores(tf, lengths, average)  # as bm25 (see test_postings), to the bit
                for chunk, s, doc in zip(ids.tolist(), term_scores.tolist(), reader.documents(ids).tolist()):
                    holding.setdefault(chunk, set()).add(t)
                    owner[chunk] = doc
                    score.setdefault(chunk, {})[t] = s
    weight = {t: max(math.log((chunks - df[t] + 0.5) / (df[t] + 0.5)), 1e-6) for t in set(named)}

    def counts(node, held):
        """(whether it holds, the leaves that count): those of an AND or NOT that fails count nothing."""
        if node[0] in topk.LEAVES:
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


@pytest.fixture(params=["indexed", "topical"])
def built(prose, request):
    """The prose collection's postings, built with its chunks in the order indexed, or, embedded, in topical order."""
    catalog, _ = prose
    if request.param == "topical":
        request.getfixturevalue("fake_models")
        embed.embed_pending(catalog)
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


def phrased(rng, pairs, depth=0, deepest=2):
    """A random expression of terms and phrases (of `pairs`, tuples of tokens), AND, OR and NOT."""
    if depth >= deepest or rng.random() < 0.35:
        return ("phrase", rng.choice(pairs)) if rng.random() < 0.5 else ("term", rng.choice(TERMS))
    kind = rng.choice(["and", "or", "or", "not"])
    if kind == "not":
        return ("not", phrased(rng, pairs, depth + 1, deepest), phrased(rng, pairs, depth + 1, deepest))
    return (kind, [phrased(rng, pairs, depth + 1, deepest) for _ in range(rng.randint(2, 3))])


def adjacent(db):
    """Pairs of tokens one after the other in the collection's chunks: phrases it holds."""
    texts = [t for (t,) in db.execute("SELECT text FROM chunk_text ORDER BY id LIMIT 30")]
    return sorted({pair for text in texts for pair in itertools.pairwise(tokenize(text))})


def phrases(db):
    """Phrases the collection holds: pairs of tokens, and phrases that can overlap themselves ("a a", "a b a"), each
    occurrence of which FTS5 counts."""
    texts = [tokenize(t) for (t,) in db.execute("SELECT text FROM chunk_text ORDER BY id")]
    overlapping = {w[:3] for t in texts for w in zip(t, t[1:], t[2:]) if w[0] == w[2]}
    overlapping |= {pair for t in texts for pair in itertools.pairwise(t) if pair[0] == pair[1]}
    return adjacent(db) + sorted(overlapping)[:6]


@pytest.mark.usefixtures("small_lists")
@pytest.mark.parametrize("seed", range(4))
def test_phrases_find_what_counting_every_phrase_finds(built, seed):
    # Nested deeper, where FTS5's own scan counts a phrase by where its reader stands, and may miss it.
    rng, readers = random.Random(300 + seed), [reader(built)]
    pairs = phrases(built.store("prose"))
    assert any(p[0] == p[-1] for p in pairs)
    for _ in range(25):
        query, k, by_document = phrased(rng, pairs, deepest=3), rng.choice([1, 3, 10, 50]), rng.random() < 0.7
        allowed = (lambda chunks: chunks % 3 > 0) if rng.random() < 0.3 else None
        want = brute(readers, query, k + 20, by_document, allowed)
        same(topk.search(readers, query, k, by_document, allowed), want, k, query)


@pytest.mark.usefixtures("small_lists")
@pytest.mark.parametrize("seed", range(4))
def test_phrases_find_what_fts5_finds(built, seed):
    # A phrase, or a hyphenated word, went to FTS5 with the whole query, which read every posting of its other words:
    # over ten minutes for a question over 30 million PubMed passages. The postings bound it by its tokens, and FTS5
    # checks it in the chunks they leave.
    rng, db = random.Random(200 + seed), built.store("prose")
    pairs = adjacent(db)
    sql = """WITH p AS MATERIALIZED (SELECT fts.rowid passage, c.doc doc, -bm25(fts) score FROM fts
             JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?)
             SELECT passage, max(score) FROM p GROUP BY doc ORDER BY 2 DESC, 1 LIMIT ?"""
    queries = [("phrase", pairs[0]), ("and", [("phrase", pairs[1]), ("term", "w1")])]
    queries += [("or", [("phrase", pairs[2]), ("term", "w2"), ("term", "w0")])]
    queries += [phrased(rng, pairs) for _ in range(20)]
    for query in queries:
        k = rng.choice([1, 5, 20])
        more = [tuple(r) for r in db.execute(sql, (fts(query), k + 20))]
        same(topk.search([reader(built)], query, k), more, k, query)


@pytest.mark.usefixtures("small_lists")
def test_a_common_phrase_among_rarer_words_is_counted_only_where_they_bring_a_chunk_near(built, monkeypatch):
    # In an OR, a phrase all of whose words are common can't bring a chunk to the k-th best score alone (MaxScore).
    monkeypatch.setattr(topk, "FIRST_BATCH", 1)  # the k-th best score rises before the last range
    db = built.store("prose")
    texts = [t for (t,) in db.execute("SELECT text FROM chunk_text")]
    common = collections.Counter(p for text in texts for p in itertools.pairwise(tokenize(text))).most_common(1)[0][0]
    query = ("or", [("phrase", common), ("term", "w25"), ("term", "w27"), ("term", "muller")])
    sql = """WITH p AS MATERIALIZED (SELECT fts.rowid passage, c.doc doc, -bm25(fts) score FROM fts
             JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?)
             SELECT passage, max(score) FROM p GROUP BY doc ORDER BY 2 DESC, 1 LIMIT ?"""
    held, asked = reader(built), []
    real = topk._Store._holds
    monkeypatch.setattr(
        topk._Store, "_holds", lambda store, phrase, places: asked.append(len(places)) or real(store, phrase, places)
    )
    same(topk.search([held], query, 2), [tuple(r) for r in db.execute(sql, (fts(query), 22))], 2, query)
    assert asked  # looked up as a lazy leaf too


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
        pairs = phrases(built.store("prose"))
        for _ in range(15):
            query, k = phrased(rng, pairs) if rng.random() < 0.5 else expression(rng), rng.choice([1, 10, 40])
            same(topk.search(readers, query, k), brute(readers, query, k + 20), k, query)


@pytest.mark.usefixtures("small_lists")
def test_across_stores_with_their_statistics_as_one(built, tmp_path):
    write_prose(tmp_path / "more", 8, notes=60)
    built.add(tmp_path / "more", "more")
    indexing.index(built)
    postings.build(built.store("more"), built.get("more").store)
    readers = [reader(built), reader(built, "more")]
    rng, pairs = random.Random(4), phrases(built.store("prose"))
    for _ in range(20):
        query, k = phrased(rng, pairs) if rng.random() < 0.5 else expression(rng), rng.choice([1, 10, 30])
        same(topk.search(readers, query, k), brute(readers, query, k + 20), k, query)


@pytest.mark.usefixtures("small_lists")
def test_a_filter_is_kept_to(built):
    readers, rng = [reader(built)], random.Random(5)

    def allowed(chunks):
        return chunks % 3 == 0

    for _ in range(20):
        query, k = expression(rng), rng.choice([1, 10])
        same(topk.search(readers, query, k, allowed=allowed), brute(readers, query, k + 20, allowed=allowed), k, query)


def test_with_time_to_spare_a_deadline_changes_nothing(built):
    readers, query = [reader(built)], ("or", [("term", "w0"), ("term", "w1"), ("term", "w2")])
    deadline = topk.Deadline(60)
    assert topk.search(readers, query, 10, deadline=deadline) == topk.search(readers, query, 10)
    assert not deadline.cut


@pytest.mark.usefixtures("small_lists")
def test_out_of_time_a_search_returns_the_best_it_found_so_far(built, tmp_path, monkeypatch):
    # Ranges are scored best bound first: what a search stopped early found is good, and each store has its best
    # ranges scored, whichever comes first.
    monkeypatch.setattr(topk, "FIRST_BATCH", 1)
    write_prose(tmp_path / "more", 8, notes=60)
    built.add(tmp_path / "more", "more")
    indexing.index(built)
    readers, query = [reader(built), reader(built, "more")], ("or", [("term", "w0"), ("term", "w1")])
    deadline = topk.Deadline(0)
    found = topk.search(readers, query, 200, by_document=False, deadline=deadline)
    exact = dict(brute(readers, query, 10_000, by_document=False))
    assert deadline.cut
    assert 0 < len(found) < len(exact)
    assert all(math.isclose(score, exact[chunk], rel_tol=1e-9) for chunk, score in found)  # scored exactly
    assert {store.owner(chunk) for chunk, _ in found} == {store.owner(chunk) for chunk in exact}  # from both stores


def test_a_term_in_no_chunk(built):
    readers = [reader(built)]
    assert topk.search(readers, ("term", "absent"), 10) == []
    assert topk.search(readers, ("and", [("term", "w1"), ("term", "absent")]), 10) == []
    assert topk.search(readers, ("or", [("term", "w1"), ("term", "absent")]), 3) == topk.search(
        readers, ("term", "w1"), 3
    )


def test_a_chunk_deleted_while_searching_is_not_found(built):
    # Another sift deleted it after the postings were read, before its document was looked up: a KeyError, which
    # reached the agent.
    readers = [reader(built)]
    (victim, _), *_ = topk.search(readers, ("term", "w1"), 5)
    db = built.store("prose")
    db.execute("DELETE FROM chunks WHERE id = ?", (victim,))
    db.commit()
    found = topk.search(readers, ("term", "w1"), 5)
    assert len(found) == 5
    assert victim not in [chunk for chunk, _ in found]


def test_no_chunks_find_nothing(tmp_path, home):
    catalog = Catalog(home)
    (tmp_path / "empty").mkdir()
    catalog.add(tmp_path / "empty", "prose")
    indexing.index(catalog)
    postings.build(catalog.store("prose"), catalog.get("prose").store)
    assert topk.search([reader(catalog)], ("term", "w1"), 10) == []


@pytest.mark.usefixtures("small_lists")
def test_terms_in_ranges_apart(tmp_path, home, monkeypatch):
    # One term's ranges hold none of the other's: each long list read only where it occurs. Indexed one term's notes
    # after the other's, as file systems list a folder in an order of their own (ext4 by hash).
    monkeypatch.setattr(topk, "FIRST_BATCH", 1)  # a range or two at a time: groups of one term's ranges alone
    root, catalog = tmp_path / "apart", Catalog(home)
    root.mkdir()
    catalog.add(root, "apart")
    for term, notes in (("alpha", range(40)), ("beta", range(40, 60))):
        for i in notes:
            write(root / f"n{i:03}.md", f"# n{i}\n\n{term} words {f'{term} ' * (i % 3)}\n")
        indexing.index(catalog)
    readers, groups = [reader(catalog, "apart")], []
    assert all(t.long for term in ("alpha", "beta") for t in readers[0].term(term))
    real = topk._Store._score
    monkeypatch.setattr(topk._Store, "_score", lambda self, group: groups.append(group.tolist()) or real(self, group))
    query = ("or", [("term", "alpha"), ("term", "beta")])
    same(topk.search(readers, query, 100), brute(readers, query, 120), 100, query)
    (beta,) = readers[0].term("beta")
    assert [g for g in groups if not set(g) & set(beta.ranges()[0].tolist())]  # a group of alpha's ranges alone


def in_ranges(tmp_path, home, groups):
    """A collection of a note per text, each note one chunk, numbered in the order given: each group a range of
    places (small_lists: 4 a range), filled up with notes of "filler". Its postings built anew: (catalog, reads), the
    ranges each term was read in (see _Store._in), counted once the reading is counted (`count`)."""
    root, catalog = tmp_path / "ranged", Catalog(home)
    root.mkdir()
    catalog.add(root, "ranged")
    texts = ["filler"] * 3  # chunk ids start one past the collection's first, a multiple of 4: 3 places in range 0
    for group in groups:
        texts += [*group, *["filler"] * (4 - len(group))]
    for n, text in enumerate(texts):  # a note at a time: its chunk takes the next id, whatever order folders list in
        write(root / f"n{n:04}.md", f"{text}\n")
        indexing.index(catalog)
    postings.build(catalog.store("ranged"), catalog.get("ranged").store)
    return catalog


def counting_reads(monkeypatch):
    """How many ranges each term is read in, by _Store._in, from now on."""
    read, real = collections.Counter(), topk._Store._in

    def counted(store, term, group):
        read[term] += len(group)
        return real(store, term, group)

    monkeypatch.setattr(topk._Store, "_in", counted)
    return read


@pytest.mark.usefixtures("small_lists")
def test_all_of_some_words_reads_each_only_where_chunks_with_all_rarer_ones_are_left(tmp_path, home, monkeypatch):
    # A question's words, all required, that no chunk holds together: nothing found, nothing pruned, and every word,
    # "the" and "of" too, read and scored in every range they all occur in: 1.2 s over 3.2M arXiv abstracts.
    catalog = in_ranges(tmp_path, home, [["alpha beta gamma", "alpha beta delta", "alpha beta"]] * 10)
    readers, read = [reader(catalog, "ranged")], counting_reads(monkeypatch)
    query = ("and", [("term", "alpha"), ("term", "beta"), ("term", "gamma"), ("term", "delta")])
    assert topk.search(readers, query, 10) == brute(readers, query, 10) == []
    assert read["gamma"] == 10  # every range holds all four, in different chunks
    assert read["alpha"] == read["beta"] == 0  # no chunk was left with the rarer ones


@pytest.mark.usefixtures("small_lists")
def test_a_word_in_every_chunk_is_read_only_where_the_others_bring_a_chunk_near(tmp_path, home, monkeypatch):
    # "of", "in", "the": in an OR of a question's words, they were read and scored in every range scored, and made a
    # question ten times as slow, though all together they can't bring a chunk to the k-th best score. Here the best
    # chunk holds apple and banana; the other ranges' bounds add up an apple's and a banana's in different chunks.
    monkeypatch.setattr(topk, "FIRST_BATCH", 1)  # a range at a time: the best is found first, from the highest bound
    best = ["apple banana stop", "apple apple apple stop"]  # a range bounded higher by the second
    others = [["apple stop filler", "banana stop filler"]] * 8
    catalog = in_ranges(tmp_path, home, [best, *others])
    readers, read = [reader(catalog, "ranged")], counting_reads(monkeypatch)
    query = ("or", [("term", "stop"), ("term", "apple"), ("term", "banana")])
    same(topk.search(readers, query, 1), brute(readers, query, 21), 1, query)
    assert read["stop"] < read["apple"] / 2


def test_ranges_only_lazy_terms_are_in_offer_nothing(built):
    # Bounded with a segment's average length (several stores, deltas), ranges of lazy terms alone can seem to reach
    # the k-th best score; none of their chunks can.
    held, query = reader(built), ("or", [("term", "w0"), ("term", "w1")])
    named = topk.terms_of(query)
    lists = {t: held.term(t) for t in dict.fromkeys(named)}
    weight = {t: topk.idf(held.chunks, sum(p.df for p in lists[t])) for t in lists}
    best = topk.Best(3)
    store = topk._Store(held, lists, weight, named, held.tokens / held.chunks, query, best, None, None)
    best.threshold = 1e9  # out of reach of both
    store._score(np.arange(1 << 20))
    assert best.found == {}


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
