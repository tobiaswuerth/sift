"""Keyword search's own postings (sift.postings): their encoding, built out of FTS5's index posting for posting, kept
in step with it through changes (deltas, masks, rebuilds), and read back for searches."""

import itertools
import math
import random
import sqlite3
import time
import tracemalloc
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import pytest

from sift import embed, indexing, postings, store
from sift.catalog import Catalog

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
            places, tf = t.everything()
            alive = reader.alive(t.segment, places)
            for chunk, n in zip(reader.chunks_at(places[alive]).tolist(), tf[alive].tolist()):
                assert chunk not in found.get(term, {}), "a chunk counted twice"
                found.setdefault(term, {})[chunk] = n
    return {term: chunks for term, chunks in found.items() if chunks}


def _varints(blob):
    """SQLite's varints read one byte at a time: the reference."""
    numbers, value = [], 0
    for byte in blob:
        value = value << 7 | byte & 0x7F
        if not byte & 0x80:
            numbers.append(value)
            value = 0
    return numbers


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


def test_lengths_round_trip_in_blocks_of_consecutive_ids():
    first = (2 << 40) + 5 * (1 << postings.LENGTHS_SHIFT)
    ids = np.array([first + 10, first + 11, first + 1023, first + 1024, first + 9000], np.int64)
    sizes = np.array([5, 300, 70000, 1, 2])
    blocks = postings.encode_lengths(ids, sizes)
    assert sorted(blocks) == [(first >> postings.LENGTHS_SHIFT) + b for b in (0, 1, 8)]
    for i, size in zip(ids, sizes):
        held = postings.decode_lengths(blocks[int(i) >> postings.LENGTHS_SHIFT])
        assert held[int(i) & ((1 << postings.LENGTHS_SHIFT) - 1)] == size
    assert postings.decode_lengths(blocks[first >> postings.LENGTHS_SHIFT]).sum() == 305 + 70000  # 0 for no chunk


def test_decoded_blocks_are_kept_to_a_budget_of_bytes():
    made = []

    def block(n):
        made.append(n)
        return np.zeros(100, np.int64), np.zeros(100, np.uint8)  # 900 bytes

    cache = postings._Cache(2000)
    for n in [1, 2, 1, 3, 1, 2]:
        cache.get(n, lambda n=n: block(n))
    assert made == [1, 2, 3, 2]  # 1 used again before 3 came: 2 went, the least recently used
    assert cache.size <= 2000


def test_a_decoded_block_holds_no_more_than_its_arrays():
    # The cache counts a block's arrays: counts read straight off the decompressed bytes kept all of those alive, the
    # gaps too, and the cache held half as much again as its budget.
    ids = np.arange(postings.BLOCK, dtype=np.int64) * 3
    data = postings.encode_block(ids, np.ones(postings.BLOCK, np.int64))
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        block = postings.decode_block(data)
        held = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert held <= sum(part.nbytes for part in block) + 1024  # and the arrays' own few hundred bytes


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
    # bm25's term score, idf aside, as FTS5's bm25() computes it
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
    assert reader.lengths(1, ids).tolist() == [lengths(db)[i] for i in ids.tolist()]
    owner = dict(db.execute("SELECT id, doc FROM chunks"))
    assert reader.documents(ids[::-1]).tolist() == [owner[i] for i in ids[::-1].tolist()]


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
        scores = postings.term_scores(tf, reader.lengths(1, ids), average)
        for r, s, e, b in zip(rid, starts, ends, best):
            assert np.all(ids[s:e] >> postings.RANGE_SHIFT == r)
            assert scores[s:e].max() <= b
        picked = [1, len(rid) - 1]  # some ranges, across blocks
        got_ids, got_tf = t.postings(starts[picked], ends[picked])
        want = np.r_[np.arange(starts[1], ends[1]), np.arange(starts[-1], ends[-1])]
        assert np.array_equal(got_ids, ids[want])
        assert np.array_equal(got_tf, tf[want])


@pytest.mark.parametrize("topical", [False, True])
def test_a_build_in_worker_processes_is_the_same(prose, monkeypatch, request, topical):
    catalog, _ = prose
    db = catalog.store("prose")
    if topical:  # the workers number the chunks by their places
        request.getfixturevalue("fake_models")
        embed.embed_pending(catalog)
    postings.build(db, path_of(catalog))
    assert bool(db.execute("SELECT count(*) FROM kw_order").fetchone()[0]) == topical
    serial = sorted(db.execute("SELECT term, segment, df, data FROM kw_terms WHERE list IS NULL"))
    monkeypatch.setattr(postings, "PARALLEL", 0)
    monkeypatch.setattr(postings, "_shards", lambda: [("", "m"), ("m", "w3"), ("w3", "{"), ("{", None)])  # a few
    monkeypatch.setattr(postings, "PROGRESS_LINES", 2)  # shown every other part
    postings.build(db, path_of(catalog), workers=2)
    assert sorted(db.execute("SELECT term, segment, df, data FROM kw_terms WHERE list IS NULL")) == serial
    assert live(db) == fts5(db)


def test_a_build_leaves_no_write_ahead_log_behind(prose):
    # A build writes the postings through the log while its own readers hold it: as large as the postings, it would
    # stay so as long as another sift (a server) has the store open.
    catalog, _ = prose
    server = catalog.store("prose")
    server.execute("SELECT count(*) FROM chunks").fetchone()
    path = path_of(catalog)
    db = store.connect(path, catalog.get("prose").id)
    postings.build(db, path)
    assert Path(f"{path}-wal").stat().st_size == 0
    assert postings.ready(server)
    db.close()


# --- in topical order ---------------------------------------------------------------
# In the order chunks are indexed, a range of postings mixes every topic, and a question of common words bounded most
# ranges high: the postings number chunks by their place in a topical order (see order), from their codes.


@pytest.fixture
def ordered(prose, fake_models):
    """The prose collection embedded, its postings built anew in topical order, as `sift index` builds them."""
    catalog, root = prose
    embed.embed_pending(catalog)
    postings.update(catalog.store("prose"), path_of(catalog))
    return catalog, root


def chunk_ids(db):
    return np.array(sorted(c for (c,) in db.execute("SELECT id FROM chunks")), np.int64)


def test_postings_number_chunks_in_topical_order(ordered):
    catalog, _ = ordered
    db = catalog.store("prose")
    reader = postings.Reader(db)
    ids = chunk_ids(db)
    places = np.arange(len(ids), dtype=np.int64) + ids[0]  # dense, from the collection's first id
    assert sorted(reader.chunks_at(places).tolist()) == ids.tolist()  # each chunk once
    assert reader.chunks_at(places).tolist() != ids.tolist()  # in another order than indexed
    assert live(db) == fts5(db)


def test_codes_of_chunks_the_build_did_not_read_are_left_out(prose, fake_models, monkeypatch):
    # A chunk added, and embedded, after the build read the lengths: its code is there, its postings aren't yet.
    catalog, _ = prose
    embed.embed_pending(catalog)
    db = catalog.store("prose")
    ids, real = chunk_ids(db), postings._lengths
    monkeypatch.setattr(postings, "_lengths", lambda *args: tuple(part[:-1] for part in real(*args)))
    postings.build(db, path_of(catalog))
    places = np.arange(len(ids) - 1, dtype=np.int64) + ids[0]
    assert sorted(postings.Reader(db).chunks_at(places).tolist()) == ids[:-1].tolist()


def test_ordering_says_so(prose, fake_models, capsys):
    # Minutes over tens of millions of passages, between the build's other steps: without a word, it looked hung.
    catalog, _ = prose
    embed.embed_pending(catalog)
    capsys.readouterr()
    db = catalog.store("prose")
    postings.build(db, path_of(catalog), "prose")
    assert f"prose: ordering {len(chunk_ids(db))} passages by topic" in capsys.readouterr().err


def test_a_collection_without_codes_keeps_its_chunks_order(prose):
    catalog, _ = prose
    db = catalog.store("prose")
    ids = chunk_ids(db)
    assert postings.Reader(db).chunks_at(ids).tolist() == ids.tolist()
    assert db.execute("SELECT count(*) FROM kw_order").fetchone() == (0,)
    assert store.get_meta(db, "kw_built").isdigit()  # as sift 1.10 reads them: chunk ids, as it takes them


def test_postings_in_topical_order_are_closed_to_sift_1_10(ordered):
    # sift 1.10 takes a place for the chunk of that id: one still running when sift 1.11 ordered the postings found
    # other documents than its scores were of. It reads postings only where kw_built is a number, FTS5 elsewhere.
    catalog, _ = ordered
    db = catalog.store("prose")
    assert not store.get_meta(db, "kw_built").isdigit()
    assert postings.ready(db)


def built_by_sift_1_10(db, path, monkeypatch):
    """The postings built anew as a sift 1.10 still running builds them: in the chunks' order, with kw_order and its
    meta left as they were, which it knows nothing of."""
    order = db.execute("SELECT block, data FROM kw_order").fetchall()
    meta = {key: store.get_meta(db, key) for key in ("kw_order", "kw_ordered")}
    with monkeypatch.context() as patched:
        patched.setattr(postings, "_topical", lambda *args: None)
        postings.build(db, path)
    db.executemany("INSERT INTO kw_order(block, data) VALUES(?, ?)", order)
    for key, value in meta.items():
        store.set_meta(db, key, value)
    db.commit()


def test_postings_sift_1_10_built_anew_over_an_order_are_read_in_theirs(ordered, monkeypatch):
    # Read through the order sift 1.10 left behind, keyword searches found other documents than they scored.
    catalog, root = ordered
    db = catalog.store("prose")
    built_by_sift_1_10(db, path_of(catalog), monkeypatch)
    assert live(db) == fts5(db)
    write(root / "n000.md", "# Changed\n\nw1 w2\n")
    indexing.reindex(catalog, ["prose/n000.md"])  # a delta, its chunks at their ids too
    assert live(db) == fts5(db)
    indexing.index(catalog)  # orders them anew, though no more chunks got codes
    assert store.get_meta(db, "kw_built").startswith(postings.ORDERED)
    assert live(db) == fts5(db)


@pytest.mark.usefixtures("small_lists")
def test_merged_deltas_keep_their_chunks_places(ordered, monkeypatch):
    # A merge gave every chunk of the deltas a place anew: an agent reindexing note after note, which never builds the
    # postings anew, grew the order with the square of its changes.
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    monkeypatch.setattr(postings, "MOST_SEGMENTS", 2)
    catalog, root = ordered
    db = catalog.store("prose")
    rng = random.Random(14)
    for _ in range(4):
        change_notes(root, rng, 3)
        indexing.index(catalog)
    places = np.arange(int(store.get_meta(db, "kw_order")), dtype=np.int64) + postings._first(db) + 1
    held = postings._chunks_at(db, places, postings._first(db)).tolist()
    assert len(held) == len(set(held))  # each chunk at one place
    assert live(db) == fts5(db)


def test_vectors_embedded_anew_order_the_postings_anew(ordered):
    # `sift embed --reset`: the codes the postings were ordered by are gone, and as many come back.
    catalog, _ = ordered
    db = catalog.store("prose")
    state = postings.Reader.state(db)
    embed.embed_pending(catalog, reset=True)
    postings.update(db, path_of(catalog))
    assert postings.Reader.state(db) != state


def test_postings_are_ordered_anew_once_many_more_chunks_have_codes(prose, fake_models):
    catalog, _ = prose
    db = catalog.store("prose")
    embed.embed_pending(catalog)
    postings.update(db, path_of(catalog), rebuild=False)  # a reindex builds nothing
    assert db.execute("SELECT count(*) FROM kw_order").fetchone() == (0,)
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT count(*) FROM kw_order").fetchone()[0] > 0
    state = postings.Reader.state(db)
    postings.update(db, path_of(catalog))  # ordered with these codes already
    assert postings.Reader.state(db) == state


def test_vectors_without_codes_order_nothing_and_build_once(prose):
    # Vectors of another length than the store's get no code (store.add_codes): nothing to order them by, and each
    # `sift index` building the postings anew to find that out again.
    catalog, _ = prose
    db = catalog.store("prose")
    db.executemany("INSERT INTO vectors(chunk, vec) VALUES(?, ?)", [(int(c), b"\0" * 6) for c in chunk_ids(db)])
    db.commit()
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT count(*) FROM kw_order").fetchone() == (0,)
    state = postings.Reader.state(db)
    postings.update(db, path_of(catalog))
    assert postings.Reader.state(db) == state


def far(monkeypatch):
    """Have the chunk ids handed out pass 2**32 from the collection's first, as ids are never handed out twice."""
    real = postings._first
    monkeypatch.setattr(postings, "_first", lambda db: real(db) - (1 << 32))


def test_chunk_ids_too_far_from_the_first_are_refused_not_wrapped():
    # The order keeps each chunk as its id from the collection's first, in 32 bits: wrapped, keyword searches would
    # find other chunks.
    with closing(order_table()) as db, pytest.raises(ValueError, match="2\\*\\*32"):
        postings._write_order(db, 1, np.array([1 << 32]), 0)


def test_chunk_ids_too_far_for_an_order_keep_the_chunks_order(prose, fake_models, monkeypatch, capsys):
    # The build refused them after deleting the postings: every `sift index` failed again, and searches read FTS5.
    catalog, _ = prose
    embed.embed_pending(catalog)
    far(monkeypatch)
    db = catalog.store("prose")
    postings.build(db, path_of(catalog), "prose")
    assert store.get_meta(db, "kw_built").isdigit()
    assert "prose: too many chunk ids for a topical order" in capsys.readouterr().err
    assert live(db) == fts5(db)
    state = postings.Reader.state(db)
    postings.update(db, path_of(catalog))  # tried once
    assert postings.Reader.state(db) == state


def test_ordered_postings_whose_ids_grew_too_far_are_built_anew_in_the_chunks_order(ordered, monkeypatch):
    # A delta can't place chunks past 2**32 in the order: a reindex leaves them to `sift index`, which builds anew.
    catalog, root = ordered
    db = catalog.store("prose")
    far(monkeypatch)
    write(root / "n000.md", "# Changed\n\nw1 w2\n")
    indexing.reindex(catalog, ["prose/n000.md"])
    assert not postings.ready(db)  # searches read FTS5 meanwhile
    indexing.index(catalog)
    assert store.get_meta(db, "kw_built").isdigit()
    assert postings.ready(db)
    assert live(db) == fts5(db)


def order_table():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE kw_order(block INTEGER PRIMARY KEY, data BLOB NOT NULL)")
    return db


def test_the_order_reads_back_at_any_places():
    first = 5 << 40
    chunks = first + 1 + np.random.default_rng(0).permutation(5000)
    places = first + 1 + np.random.default_rng(1).permutation(5000)
    with closing(order_table()) as db:
        postings._write_order(db, first + 1, chunks[:3000], first)
        postings._write_order(db, first + 3001, chunks[3000:], first)  # into the block the first left half full
        postings._write_order(db, first + 5001, chunks[:0], first)
        assert postings._chunks_at(db, places, first).tolist() == chunks[places - first - 1].tolist()
        assert postings._chunks_at(db, places[:0], first).tolist() == []


def test_the_order_is_written_and_read_in_time_linear_in_its_chunks():
    # Each block of places masked the whole order: 12 minutes for PubMed's 29.9 million chunks, in the build's write
    # transaction, past which a `sift embed` beside it failed on the lock. Four times the chunks, four times the time.
    first = 5 << 40

    def took(n):
        chunks, places = first + 1 + np.random.default_rng(0).permutation(n), first + 1 + np.arange(n)
        best = math.inf
        for _ in range(3):
            db, start = order_table(), time.perf_counter()
            postings._write_order(db, first + 1, chunks, first)
            postings._chunks_at(db, places, first)
            best = min(best, time.perf_counter() - start)
            db.close()
        return best

    assert took(1_000_000) < 8 * took(250_000)


def test_a_build_anew_is_a_new_state_for_readers(prose):
    # A server keeps a store's reader for its state: a build that changes no chunk, as one in a new order, must
    # still make another.
    catalog, _ = prose
    db = catalog.store("prose")
    state = postings.Reader.state(db)
    postings.build(db, path_of(catalog))
    assert postings.Reader.state(db) != state


@pytest.mark.usefixtures("small_lists")
def test_changes_after_ordering_take_places_after_all_others(ordered, monkeypatch):
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    catalog, root = ordered
    db = catalog.store("prose")
    last = int(store.get_meta(db, "kw_order"))
    change_notes(root, random.Random(12))
    indexing.index(catalog)
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (2,)
    assert int(store.get_meta(db, "kw_order")) > last  # the changed and new chunks, placed after the rest
    assert live(db) == fts5(db)


@pytest.mark.usefixtures("small_lists")
def test_ordered_deltas_grown_too_many_are_merged(ordered, monkeypatch):
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 1.0)
    monkeypatch.setattr(postings, "MOST_SEGMENTS", 2)
    catalog, root = ordered
    db = catalog.store("prose")
    rng = random.Random(13)
    for _ in range(3):
        change_notes(root, rng, 3)
        indexing.index(catalog)
        assert live(db) == fts5(db)
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone()[0] <= 2


def test_a_phrase_is_counted_as_fts5s_bm25_counts_it(tmp_path, home):
    # Overlapping occurrences each: "a a" is twice in "a a a", where highlight() marks it once.
    root, catalog = tmp_path / "phrases", Catalog(home)
    write(root / "a.md", "# Repeat\n\nw1 w1 w1 w2\n")
    write(root / "b.md", "# Once\n\nw1 w1\n")
    write(root / "c.md", "# None\n\nw1 w2 w1\n")
    catalog.add(root, "phrases")
    indexing.index(catalog)
    db = catalog.store("phrases")
    reader = postings.Reader(db)
    chunks = dict(db.execute("SELECT d.path, c.id FROM chunks c JOIN docs d ON d.id = c.doc"))
    ids = [chunks["a.md"], chunks["b.md"], chunks["c.md"]]
    counts, sizes = reader.phrase_counts(("w1", "w1"), ids)
    assert counts.tolist() == [2, 1, 0]
    assert sizes.tolist() == [lengths(db)[ids[0]], lengths(db)[ids[1]], 0]  # 0 where it isn't: no score
    assert reader.phrase_df(("w1", "w1")) == 2
    # as FTS5's queries of several collections score their phrases (search._as_one)
    assert postings.matches(db, '"w1 w1"', ids) == {ids[0]: 2, ids[1]: 1}
    assert postings.matches(db, '"w1 w2"', ids) == {ids[0]: 1, ids[2]: 1}
    assert postings.matches(db, "w*", ids) == {ids[0]: 4, ids[1]: 2, ids[2]: 3}  # a prefix: each word it begins


@pytest.mark.parametrize("phrase", [("w1", "w2"), ("w1", "w1")])  # counted by highlight(), or from the tokens
def test_a_phrase_is_counted_in_a_chunk_once_for_these_postings(tmp_path, home, phrase):
    # A question with a hyphenated word over 45 million chunks counted its phrase in the same chunks on every call,
    # with FTS5: 62% of its 2.2 seconds.
    root, catalog = tmp_path / "phrases", Catalog(home)
    for name, words in (("a", "w1 w1 w1 w2"), ("b", "w1 w2 w9 w1 w2"), ("c", "w1 w1 w2")):
        write(root / f"{name}.md", f"# {name}\n\n{words}\n")
    catalog.add(root, "phrases")
    indexing.index(catalog)
    db = catalog.store("phrases")
    chunks = dict(db.execute("SELECT d.path, c.id FROM chunks c JOIN docs d ON d.id = c.doc"))
    reader, asked = postings.Reader(db), []
    first = reader.phrase_counts(phrase, [chunks["a.md"], chunks["b.md"]])
    db.set_trace_callback(asked.append)
    try:
        again = reader.phrase_counts(phrase, [chunks["b.md"], chunks["a.md"]])
        assert asked == []  # nothing asked of the store
        more = reader.phrase_counts(phrase, [chunks["a.md"], chunks["c.md"]])
    finally:
        db.set_trace_callback(None)
    assert [list(part) for part in again] == [list(part)[::-1] for part in first]
    assert [list(part) for part in more] == [
        list(part) for part in postings.Reader(db).phrase_counts(phrase, [chunks["a.md"], chunks["c.md"]])
    ]
    assert all(str(chunks["a.md"]) not in s for s in asked)  # only the chunk not counted yet


def test_highlights_mark_already_in_a_text_counts_for_nothing(tmp_path, home):
    # Control characters come with PDFs and mails; highlight()'s mark among them counted as phrases found.
    root, catalog = tmp_path / "marked", Catalog(home)
    write(root / "a.md", f"# Marked {postings.MARK}\n\nw1 w2 {postings.MARK * 3} w3 w1 w2\n")
    catalog.add(root, "marked")
    indexing.index(catalog)
    db = catalog.store("marked")
    (chunk,) = [c for (c,) in db.execute("SELECT id FROM chunks")]
    assert postings.matches(db, '"w1 w2"', [chunk]) == {chunk: 2}
    assert postings.matches(db, "w*", [chunk]) == {chunk: 5}
    assert postings.Reader(db).phrase_counts(("w1", "w2"), [chunk])[0].tolist() == [2]


def test_the_chunks_of_phrases_are_kept_to_a_budget_of_bytes(tmp_path, home, monkeypatch):
    # A server asked ever new phrases kept the chunks of each for as long as the postings stood: 80 MB for a common
    # phrase over 30 million chunks.
    root, catalog = tmp_path / "phrases", Catalog(home)
    for i in range(6):
        write(root / f"n{i}.md", f"# Note\n\nw1 w2 w{i + 3}\n")
    catalog.add(root, "phrases")
    indexing.index(catalog)
    db = catalog.store("phrases")
    budget = 16 * 8  # chunk ids, counts and lengths: the six of "w1 w2" and one each of two other phrases
    monkeypatch.setattr(postings, "_PHRASES", postings._Cache(budget))
    reader, asked = postings.Reader(db), []
    db.set_trace_callback(lambda sql: "MATCH" in sql and asked.append(sql))
    try:
        for phrase in [("w1", "w2"), ("w2", "w3"), ("w1", "w2"), ("w2", "w4"), ("w2", "w5"), ("w2", "w6")]:
            assert len(reader.phrase_chunks(phrase)) == (6 if phrase == ("w1", "w2") else 1)
        assert postings._PHRASES.size <= budget
        assert len(asked) == 5  # ("w1", "w2") once while it was kept
        reader.phrase_chunks(("w1", "w2"))
        assert len(asked) == 6  # let go since, and found again
    finally:
        db.set_trace_callback(None)


def test_the_parts_of_a_build_cover_every_term_once():
    shards = postings._shards()
    assert shards[0][0] == ""
    assert shards[-1][1] is None
    assert all(hi == next_lo for (_, hi), (next_lo, _) in itertools.pairwise(shards))
    assert all(lo < hi for lo, hi in shards[:-1])

    def part(term):
        return next(i for i, (lo, hi) in enumerate(shards) if lo <= term and (hi is None or term < hi))

    assert len({part("that"), part("the"), part("this")}) == 3  # common words in parts of their own


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
    db.execute("DELETE FROM meta WHERE key = 'kw_built'")
    for table in ("kw_terms", "kw_blocks", "kw_lengths", "kw_segments", "kw_masked", "kw_dirty"):
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
    module, name = {"starting": (store, "set_meta"), "finishing": (postings, "_write_lengths")}[where]
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
    other = sqlite3.connect(store.postings_lock(path_of(catalog)), isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")
    postings.build(db, path_of(catalog), "prose")
    assert "another sift is updating" in capsys.readouterr().err
    assert not postings.built(db)
    other.execute("ROLLBACK")
    other.close()  # as the OS does when that sift ends, however it ends
    postings.build(db, path_of(catalog), "prose")
    assert postings.ready(db)


def test_a_delta_waits_for_a_build_elsewhere(prose, monkeypatch):
    # A build elsewhere rewrites every table: a delta written meanwhile would be lost, or mix with it.
    monkeypatch.setattr(postings, "SMALL", 0)
    catalog, _ = prose
    db = catalog.store("prose")
    store.delete_docs(db, [db.execute("SELECT min(id) FROM docs").fetchone()[0]])
    db.commit()
    other = sqlite3.connect(store.postings_lock(path_of(catalog)), isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")
    postings.update(db, path_of(catalog))
    assert postings.behind(db) > 0
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (1,)
    other.close()
    postings.update(db, path_of(catalog))
    assert postings.ready(db)


def test_an_update_another_sift_did_meanwhile_is_not_done_again(prose, monkeypatch):
    # Behind when asked, up to date once this sift has the lock: another sift updated them in between.
    monkeypatch.setattr(postings, "SMALL", 0)
    catalog, _ = prose
    db = catalog.store("prose")
    store.delete_docs(db, [db.execute("SELECT min(id) FROM docs").fetchone()[0]])
    db.commit()
    real = postings._alone

    @contextmanager
    def after_another(path):
        with real(path) as alone:
            postings._delta(db)  # what the other sift did, before this one got the lock
            yield alone

    monkeypatch.setattr(postings, "_alone", after_another)
    postings.update(db, path_of(catalog))
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (2,)  # its delta, no second one
    assert postings.ready(db)


def test_a_reindex_never_builds(prose, monkeypatch):
    # An agent's reindex waits for its answer: it adds a delta of what changed, however much, and builds nothing,
    # which reads the whole store; `sift index` does that.
    monkeypatch.setattr(postings, "SMALL", 0)
    monkeypatch.setattr(postings, "REBUILD_SHARE", 0.0)
    catalog, root = prose
    db = catalog.store("prose")
    write(root / "n000.md", "# Changed\n\nw1 w2\n")
    indexing.reindex(catalog, ["prose/n000.md"])
    assert postings.ready(db)
    assert db.execute("SELECT count(*) FROM kw_segments").fetchone() == (2,)  # a delta, though over the share
    assert live(db) == fts5(db)
    unbuilt(db)
    write(root / "n001.md", "# Changed too\n")
    indexing.reindex(catalog, ["prose/n001.md"])
    assert not postings.built(db)  # left to `sift index`; searches read FTS5 meanwhile
    indexing.index(catalog)
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
