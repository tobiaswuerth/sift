"""Keyword search's own postings (sift.postings): their encoding, built out of FTS5's index posting for posting, kept
in step with it through changes (deltas, masks, rebuilds), and read back for searches."""

import random

import numpy as np
import pytest

from sift import indexing, postings, store
from sift.catalog import Catalog
from sift.search import _varints

from .conftest import prose_text as text
from .conftest import write


def fts5(db):
    """FTS5's postings: term -> {chunk: count}, from its vocabulary, one row per occurrence."""
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.truth USING fts5vocab(main, fts, instance)")
    found = {}
    for term, chunk in db.execute("SELECT term, doc FROM temp.truth"):
        found.setdefault(term, {}).setdefault(chunk, 0)
        found[term][chunk] += 1
    return found


def live(db):
    """The postings searches read: per term, its chunks and counts in every segment, those masked left out."""
    reader = postings.Reader(db)
    terms = [t for (t,) in db.execute("SELECT DISTINCT term FROM kw_terms")]
    found = {}
    for term in terms:
        for t in reader.term(term):
            ids, tf = t.everything()
            alive = reader.alive(t.segment, ids)
            for chunk, n in zip(ids[alive].tolist(), tf[alive].tolist()):
                assert chunk not in found.get(term, {}), "a chunk counted twice"
                found.setdefault(term, {})[chunk] = n
    return {term: chunks for term, chunks in found.items() if chunks}


def lengths(db):
    return {i: sum(_varints(sz)) for i, sz in db.execute("SELECT id, sz FROM fts_docsize")}


def path_of(catalog):
    return catalog.get("prose").store


# --- encoding ------------------------------------------------------------------


def test_blocks_round_trip():
    rng = np.random.default_rng(0)
    ids = np.cumsum(rng.integers(1, 1 << 31, 500)) + (5 << 40)
    for most in (200, 60000, 70000):  # counts of a byte, two, four
        tf = rng.integers(1, most, 500)
        got_ids, got_tf = postings.decode_block(postings.encode_block(ids, tf))
        assert np.array_equal(got_ids, ids)
        assert np.array_equal(got_tf, tf)
    single = postings.decode_block(postings.encode_block(ids[:1], np.array([3])))
    assert (single[0].tolist(), single[1].tolist()) == ([int(ids[0])], [3])


def test_ids_too_far_apart_are_refused():
    with pytest.raises(ValueError, match="2\\*\\*32"):
        postings.encode_block(np.array([1, 1 + (1 << 32)]), np.array([1, 1]))


def test_ranges_round_trip_with_bounds_rounded_up():
    rng = np.random.default_rng(1)
    rid = np.cumsum(rng.integers(1, 1000, 300)) + (3 << 33)
    counts = rng.integers(1, (1 << postings.RANGE_SHIFT) + 1, 300)
    scores = rng.uniform(0, postings.TOP, 300)
    got_rid, starts, ends, best = postings.decode_ranges(
        postings.encode_ranges(rid, counts, postings.quantized(scores))
    )
    assert np.array_equal(got_rid, rid)
    assert np.array_equal(ends - starts, counts)
    assert starts[0] == 0
    assert np.array_equal(starts[1:], ends[:-1])
    assert np.all(best >= scores)
    assert np.all(best - scores <= postings.TOP / 255 + 1e-9)


def test_chunks_round_trip():
    ids = np.array([10, 11, 12, 40, 41], np.int64) + (2 << 40)
    docs = np.array([7, 7, 8, 3, 3], np.int64) + (2 << 40)  # a chunk added to an older document later
    sizes = np.array([5, 300, 70000, 1, 0])
    got = postings.decode_chunks(postings.encode_chunks(ids, sizes, docs))
    assert [g.tolist() for g in got] == [ids.tolist(), sizes.tolist(), docs.tolist()]


def test_varints_are_read_as_sqlite_writes_them():
    rng = random.Random(3)
    numbers = [rng.choice([0, 1, 127, 128, 300, 16383, 16384, 2**21, 2**28 + 5]) for _ in range(200)]

    def varint(n):
        groups = [n & 0x7F]
        while n := n >> 7:
            groups.append(n & 0x7F | 0x80)
        return bytes(reversed(groups))

    blob = b"".join(map(varint, numbers))
    got, starts = postings.varints(blob)
    assert got.tolist() == numbers == _varints(blob)
    assert starts.tolist() == list(np.cumsum([0] + [len(varint(n)) for n in numbers[:-1]]))


def test_term_scores_are_fts5s():
    # bm25's term score, idf aside, as search._as_one computes it after FTS5
    assert postings.term_scores([2], [10], 20)[0] == pytest.approx(2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 10 / 20)))


# --- building ------------------------------------------------------------------


@pytest.mark.usefixtures("small_lists")
def test_a_build_holds_fts5s_postings(prose):
    catalog, _ = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == fts5(db)
    reader = postings.Reader(db)
    assert reader.chunks == store.counts(db)["chunks"]
    assert reader.tokens == sum(lengths(db).values())
    ids = np.array(sorted(lengths(db)), np.int64)
    got_lengths, got_docs = reader.chunks_of(1, ids)
    assert got_lengths.tolist() == [lengths(db)[i] for i in ids.tolist()]
    owner = dict(db.execute("SELECT id, doc FROM chunks"))
    assert got_docs.tolist() == [owner[i] for i in ids.tolist()]


@pytest.mark.usefixtures("small_lists")
def test_long_lists_have_ranges_that_bound_their_scores(prose):
    catalog, _ = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    reader = postings.Reader(db)
    average = reader.tokens / reader.chunks
    long = [t for (t,) in db.execute("SELECT term FROM kw_terms WHERE list IS NOT NULL")]
    assert len(long) > 5
    for term in long:
        (t,) = reader.term(term)
        rid, starts, ends, best = t.ranges()
        ids, tf = t.everything()
        assert np.array_equal(np.unique(ids >> postings.RANGE_SHIFT), rid)
        scores = postings.term_scores(tf, reader.chunks_of(1, ids)[0], average)
        for r, s, e, b in zip(rid, starts, ends, best):
            assert np.all(ids[s:e] >> postings.RANGE_SHIFT == r)
            assert scores[s:e].max() <= b
        picked = [1, len(rid) - 1]  # some ranges, across blocks
        got_ids, got_tf = t.postings(starts[picked], ends[picked])
        want = np.r_[np.arange(starts[1], ends[1]), np.arange(starts[-1], ends[-1])]
        assert np.array_equal(got_ids, ids[want])
        assert np.array_equal(got_tf, tf[want])


def test_a_build_in_worker_processes_is_the_same(prose, monkeypatch):
    catalog, _ = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    serial = sorted(db.execute("SELECT term, segment, df, data FROM kw_terms WHERE list IS NULL"))
    monkeypatch.setattr(postings, "PARALLEL", 0)
    postings.build(db, path_of(catalog), workers=2)
    assert sorted(db.execute("SELECT term, segment, df, data FROM kw_terms WHERE list IS NULL")) == serial
    assert live(db) == fts5(db)


def test_an_empty_store_builds_to_nothing(tmp_path, home):
    catalog = Catalog(home)
    (tmp_path / "empty").mkdir()
    catalog.add(tmp_path / "empty", "prose")
    indexing.index(catalog)
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == {}
    assert postings.Reader(db).chunks == 0


# --- updating ------------------------------------------------------------------


def change_notes(root, rng, n=5):
    """Change, add and delete some notes."""
    for i in rng.sample(range(120), n):
        write(root / f"n{i:03}.md", f"# {text(rng, 2)}\n\n{text(rng, 30)}\n")
    write(root / f"new{rng.randrange(10**6)}.md", f"# New\n\n{text(rng, 50)}\n")
    for i in rng.sample(range(120), 2):
        (root / f"n{i:03}.md").unlink(missing_ok=True)


@pytest.mark.usefixtures("small_lists")
def test_changes_go_into_deltas_that_follow_fts5(prose, monkeypatch):
    monkeypatch.setattr(postings, "SMALL", 0)  # a delta, however small the store
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    catalog, root = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    rng = random.Random(11)
    for round_ in range(3):
        change_notes(root, rng)
        indexing.index(catalog)
        assert postings.behind(db) > 0
        assert not postings.ready(db)
        postings.update(db, path_of(catalog))
        assert postings.ready(db)
        assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (round_ + 2,)
        assert live(db) == fts5(db)
    assert db.execute("SELECT count(*) FROM kw_masked").fetchone()[0] > 0


@pytest.mark.usefixtures("small_lists")
@pytest.mark.parametrize("limit", ["MOST_SEGMENTS", "REBUILD_SHARE", "SMALL"])
def test_deltas_grown_too_many_or_large_are_built_anew(prose, monkeypatch, limit):
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    monkeypatch.setattr(postings, "MOST_SEGMENTS", 100)
    catalog, root = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    rng = random.Random(5)
    change_notes(root, rng)
    indexing.index(catalog)
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (2,)
    monkeypatch.setattr(postings, limit, {"MOST_SEGMENTS": 2, "REBUILD_SHARE": 0.001, "SMALL": 10**9}[limit])
    change_notes(root, rng)
    indexing.index(catalog)
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT id FROM kw_segments").fetchall() == [(1,)]
    assert db.execute("SELECT count(*) FROM kw_masked").fetchone() == (0,)
    assert live(db) == fts5(db)


def test_an_update_builds_postings_never_built(prose):
    catalog, _ = prose
    db = catalog.store("prose")
    assert not postings.built(db)
    postings.update(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == fts5(db)
    postings.update(db, path_of(catalog))  # nothing changed: nothing to do
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (1,)


def test_chunks_changed_during_a_build_stay_dirty(prose, monkeypatch):
    # A build reads FTS5 while others may write: what changed meanwhile can't be trusted, and stays dirty.
    catalog, root = prose
    db = catalog.store("prose")
    real = postings._write_terms
    rng = random.Random(2)

    def meanwhile(*args):
        real(*args)
        db.commit()
        change_notes(root, rng, 2)
        indexing.index(catalog)

    monkeypatch.setattr(postings, "_write_terms", meanwhile)
    postings.build(db, path_of(catalog))
    assert postings.built(db)
    assert postings.behind(db) > 0
    assert not postings.ready(db)
    monkeypatch.setattr(postings, "_write_terms", real)
    postings.update(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == fts5(db)


def test_a_document_retitled_in_place_is_followed(prose, monkeypatch):
    # Its chunks keep their ids, their words change: masked in the base, held by the delta.
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    catalog, _ = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    indexing.retitle(catalog, "prose", "name")
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (2,)
    assert live(db) == fts5(db)
