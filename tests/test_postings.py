"""Keyword search's own postings (sift.postings): their encoding, built out of FTS5's index posting for posting, kept
in step with it through changes (deltas, masks, rebuilds), and read back for searches."""

import random
import sqlite3

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
        indexing.index(catalog)  # which updates the postings after
        assert postings.ready(db)
        assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (round_ + 2,)
        assert live(db) == fts5(db)
    assert db.execute("SELECT count(*) FROM kw_masked").fetchone()[0] > 0


def test_changes_are_followed_once_updated(prose):
    catalog, _ = prose
    db = catalog.store("prose")
    store.delete_docs(db, [db.execute("SELECT min(id) FROM docs").fetchone()[0]])  # as indexing does, then updates
    db.commit()
    assert postings.behind(db) > 0
    assert not postings.ready(db)
    postings.update(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == fts5(db)


def unbuilt(db):
    """The store as an upgrade from sift 1.9 leaves it: no postings built."""
    store.set_meta(db, "kw_built", "")
    db.execute("DELETE FROM meta WHERE key = 'kw_built'")
    for table in ("kw_terms", "kw_blocks", "kw_chunks", "kw_segments", "kw_masked", "kw_dirty"):
        db.execute(f"DELETE FROM {table}")
    db.commit()


@pytest.mark.usefixtures("small_lists")
@pytest.mark.parametrize("limit", ["REBUILD_SHARE", "SMALL"])
def test_postings_changed_too_much_are_built_anew(prose, monkeypatch, limit):
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    catalog, root = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    rng = random.Random(5)
    change_notes(root, rng)
    indexing.index(catalog)
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (2,)
    monkeypatch.setattr(postings, limit, {"REBUILD_SHARE": 0.001, "SMALL": 10**9}[limit])
    change_notes(root, rng)
    indexing.index(catalog)
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT id FROM kw_segments").fetchall() == [(1,)]
    assert db.execute("SELECT count(*) FROM kw_masked").fetchone() == (0,)
    assert live(db) == fts5(db)


@pytest.mark.usefixtures("small_lists")
def test_deltas_grown_too_many_are_merged(prose, monkeypatch):
    # Every reindex of a note adds a delta; rather than build anew, the deltas become one, read again from FTS5.
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    monkeypatch.setattr(postings, "MOST_SEGMENTS", 3)
    catalog, root = prose
    db = catalog.store("prose")
    postings.build(db, path_of(catalog))
    rng = random.Random(6)
    segments = []
    for _ in range(4):
        change_notes(root, rng, 3)
        indexing.index(catalog)
        postings.update(db, path_of(catalog))
        segments.append([s for (s,) in db.execute("SELECT id FROM kw_segments ORDER BY id")])
        assert live(db) == fts5(db)
    assert segments == [[1, 2], [1, 2, 3], [1, 2], [1, 2, 3]]  # with a third segment, the next update merges


def test_an_update_builds_postings_never_built(prose):
    catalog, _ = prose
    db = catalog.store("prose")
    unbuilt(db)
    assert not postings.built(db)
    assert not postings.ready(db)
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


@pytest.mark.usefixtures("small_lists")
def test_a_terms_occurrences_read_in_several_batches(prose, monkeypatch):
    catalog, _ = prose
    db = catalog.store("prose")
    monkeypatch.setattr(postings, "FETCH", 7)  # far fewer rows than a common term has
    postings.build(db, path_of(catalog))
    assert live(db) == fts5(db)


def test_chunks_added_before_the_build_reads_fts5_wait_for_the_next_update(tmp_path, home, monkeypatch):
    # Lengths are read first, then FTS5's vocabulary: a chunk added between the two has postings but no length yet.
    folder = tmp_path / "prose"
    folder.mkdir()
    catalog = Catalog(home)
    catalog.add(folder, "prose")
    indexing.index(catalog)  # nothing in it yet
    db = catalog.store("prose")
    real = postings._lengths

    def meanwhile(*args):
        read = real(*args)
        write(folder / "first.md", "# First\n\nzebra quagga\n")
        indexing.index(catalog)  # its update finds the build under way, and leaves it
        return read

    monkeypatch.setattr(postings, "_lengths", meanwhile)
    postings.build(db, path_of(catalog))
    monkeypatch.setattr(postings, "_lengths", real)
    assert postings.behind(db) > 0
    assert live(db) == {}
    postings.update(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == fts5(db)


class Stop(Exception):
    pass


def stop(*args, **kwargs):
    raise Stop


@pytest.mark.parametrize("where", ["starting", "finishing"])
def test_a_build_stopped_leaves_the_postings_to_be_built(prose, monkeypatch, where):
    # Stopped as it starts, the postings are as they were; stopped later, unbuilt, which searches see (FTS5 answers),
    # and the next update builds them.
    catalog, _ = prose
    db = catalog.store("prose")
    before = live(db)
    module, name = {"starting": (store, "set_meta"), "finishing": (postings, "_write_chunks")}[where]
    real = getattr(module, name)
    monkeypatch.setattr(module, name, stop)
    with pytest.raises(Stop):
        postings.build(db, path_of(catalog))
    monkeypatch.setattr(module, name, real)  # not undo(): that would undo the fixtures' too
    if where == "starting":
        assert postings.ready(db)
        assert live(db) == before
    else:
        assert not postings.built(db)
    postings.update(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == fts5(db)


def test_a_delta_stopped_leaves_the_chunks_dirty(prose, monkeypatch):
    monkeypatch.setattr(postings, "SMALL", 0)
    catalog, _ = prose
    db = catalog.store("prose")
    store.delete_docs(db, [db.execute("SELECT min(id) FROM docs").fetchone()[0]])
    db.commit()
    behind, real = postings.behind(db), postings._write_terms
    monkeypatch.setattr(postings, "_write_terms", stop)
    with pytest.raises(Stop):
        postings.update(db, path_of(catalog))
    assert postings.behind(db) == behind
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (1,)
    monkeypatch.setattr(postings, "_write_terms", real)
    postings.update(db, path_of(catalog))
    assert postings.ready(db)
    assert live(db) == fts5(db)


def test_a_delta_of_deletions_alone(prose, monkeypatch):
    monkeypatch.setattr(postings, "SMALL", 0)
    catalog, _ = prose
    db = catalog.store("prose")
    store.delete_docs(db, [d for (d,) in db.execute("SELECT id FROM docs LIMIT 3")])
    db.commit()
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT chunks FROM kw_segments WHERE id = 2").fetchone() == (0,)
    assert live(db) == fts5(db)


def test_a_build_elsewhere_is_left_to_finish(prose, capsys):
    # Two sifts building at once would write over each other: the second leaves the work to the first.
    catalog, _ = prose
    db = catalog.store("prose")
    unbuilt(db)
    other = sqlite3.connect(f"{path_of(catalog)}.postings-lock", isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")
    postings.build(db, path_of(catalog), "prose")
    assert "another sift is building" in capsys.readouterr().err
    assert not postings.built(db)
    other.execute("ROLLBACK")
    other.close()  # as the OS does when that sift ends, however it ends
    postings.build(db, path_of(catalog), "prose")
    assert postings.ready(db)


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
