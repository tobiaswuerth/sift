import contextlib
import tempfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sift.api.errors import StoreUnavailableError
from sift.api.ids import CollectionId
from sift.domain.arrays import ints
from sift.domain.codec.postings import blocked, positions_of
from sift.domain.codec.segments import Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.segments import Held, Policy, merges
from sift.storage.database import Batches, Connection, Reading, WritingDocuments, reading, rows, run, writing
from sift.storage.keyword_lists import blocks, dead, expansions, frequencies, parts, positions, segments
from sift.storage.keywords import REMAKE_REMEDY, current, remade, remake_started, totals
from sift.storage.store import LAYOUT, create_store, open_store
from tests.keyword_stores import COLUMNS, SMALL, Content, Instance, Round, applied, changed, instances, opened_store

WORDS = ("alpha", "beta", "Gamma", "δέλτα", "x-ray", "naïve")
FIELDS = ("", "title: Front\n", "tags: [alpha, 2]\nscore: 1e20\nratio: 0.1\nok: true\n", "nested: {title: beta}\n")
text = st.lists(st.sampled_from(WORDS), max_size=40).map(" ".join)
content = st.tuples(st.sampled_from(FIELDS), text, text)
rounds = st.lists(
    st.dictionaries(st.sampled_from(["a.md", "b.md", "c.md", "d.md"]), st.one_of(st.none(), content), max_size=4),
    min_size=1,
    max_size=4,
)


class Term(NamedTuple):
    term: str


type Places = dict[str, dict[int, list[int]]]


def expected(found: list[Instance]) -> Places:
    sizes = Counter((each.chunk, each.column) for each in found)
    held: Places = {}
    for each in found:
        before = sum(sizes[each.chunk, column] + 1 for column in COLUMNS[: COLUMNS.index(each.column)])
        held.setdefault(each.term, {}).setdefault(each.chunk, []).append(before + each.offset)
    return {term: {chunk: sorted(places) for chunk, places in chunks.items()} for term, chunks in held.items()}


def stored(transaction: Reading) -> Places:
    gone = set(ints(dead(transaction)))
    every = sorted({row.term for row in rows(transaction, Term, "SELECT DISTINCT term FROM keyword_postings")})
    held: Places = {}
    for part in parts(transaction, every):
        found = (
            [(part.data, positions(transaction, part.segment, part.term, None)[0].positions)]
            if not part.ranged
            else [
                (block.postings, place.positions)
                for block, place in zip(
                    blocks(transaction, part.segment, part.term, range(part.chunks)),
                    positions(transaction, part.segment, part.term, range(part.chunks)),
                    strict=True,
                )
            ]
        )
        for postings, places in found:
            ids, counts = blocked(postings)
            flat = positions_of(places, counts)
            for chunk, own in zip(ints(ids), np.split(flat, np.cumsum(counts)[:-1]), strict=True):
                if chunk not in gone:
                    held.setdefault(part.term, {})[chunk] = own.tolist()
    return held


class Kept(NamedTuple):
    first: int
    last: int
    chunks: int
    dead: int


def checked(connection: Connection) -> None:
    with reading(connection) as transaction:
        found = instances(transaction)
        want = expected(found)
        assert stored(transaction) == want
        assert frequencies(transaction, sorted(want)) == {term: len(chunks) for term, chunks in want.items()}
        assert sorted(row.term for row in rows(transaction, Term, "SELECT term FROM keyword_terms")) == sorted(want)
        assert totals(transaction).tokens == len(found)
        held = segments(transaction)
        live = {chunk for chunks in want.values() for chunk in chunks}
        assert all(sum(s.first <= chunk <= s.last for s in held) == 1 for chunk in live)
        gone = ints(dead(transaction))
        for each in rows(transaction, Kept, "SELECT first, last, chunks, dead FROM keyword_segments"):
            assert each.dead == sum(each.first <= chunk <= each.last for chunk in gone) < each.chunks


@settings(max_examples=40, deadline=None)
@given(rounds, st.sampled_from([SMALL, Shape()]))
def test_the_keyword_index_holds_what_fts5_makes_of_the_live_chunks_however_they_changed(
    each: list[Round], shape: Shape
) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, shape) as connection:
        applied(connection, each)
        checked(connection)


@settings(max_examples=20, deadline=None)
@given(rounds)
def test_segments_of_few_chunks_and_documents_forgotten_a_few_at_a_time_index_the_same(each: list[Round]) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, each, Batches(chunks_a_segment=2, documents_a_forget=1))
        checked(connection)


TWO = Batches(chunks_a_segment=2)
NOTES: Round = {f"n{n}.md": ("", f"Note {n}", "one two three " * n) for n in range(7)}


class Average(NamedTuple):
    chunks: int
    tokens: int
    average: float


def staled(connection: Connection) -> None:
    with writing(connection) as transaction:
        run(transaction, "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
        run(transaction, "DELETE FROM keyword_postings")


def test_an_index_of_another_version_takes_no_writes_until_remade_a_batch_at_a_time() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        staled(connection)
        with pytest.raises(StoreUnavailableError, match="another version") as refused:
            applied(connection, [{}])
        assert refused.value.remedy == REMAKE_REMEDY
        assert remake_started(connection) == len(NOTES)
        assert list(remade(connection, TWO)) == [2, 4, 6, 7]
        with reading(connection) as transaction:
            assert current(transaction)
            made = rows(transaction, Average, "SELECT chunks, tokens, average FROM keyword_segments ORDER BY first")
        assert [each.average for each in made] == pytest.approx(
            [sum(m.tokens for m in made[: n + 1]) / sum(m.chunks for m in made[: n + 1]) for n in range(len(made))]
        )
        assert remake_started(connection) == 0
        applied(connection, [{"x.md": ("", "X", "four")}])
        checked(connection)


def test_a_remake_cut_short_goes_on_where_it_stopped_and_one_of_another_version_starts_over() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        staled(connection)
        assert remake_started(connection) == len(NOTES)
        assert next(iter(remade(connection, TWO))) == 2
        assert remake_started(connection) == len(NOTES) - 2
        with writing(connection) as transaction:
            run(transaction, "UPDATE meta SET value = 'remaking other' WHERE key = 'keyword_fingerprint'")
        with pytest.raises(StoreUnavailableError, match="another sift"):
            list(remade(connection, TWO))
        assert remake_started(connection) == len(NOTES)
        assert list(remade(connection))[-1] == len(NOTES)
        checked(connection)


def test_a_store_of_the_layout_before_the_keyword_index_gets_one_when_remade() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "3.sqlite"
        create_store(path, CollectionId(3), LAYOUT[:1])
        with (
            contextlib.closing(open_store(path, CollectionId(3), LAYOUT[:1])) as connection,
            writing(connection) as transaction,
        ):
            run(transaction, "DELETE FROM meta WHERE key LIKE 'keyword%'")
            changed(WritingDocuments(connection), "a.md", ("tags: [old]\n", "Old", "kept from before"))
        with contextlib.closing(open_store(path, CollectionId(3))) as connection:
            assert remake_started(connection) == 1
            assert list(remade(connection)) == [1]
            applied(connection, [{"b.md": ("", "New", "added after")}])
            checked(connection)
            with reading(connection) as transaction:
                assert frequencies(transaction, ["old", "added"]) == {"old": 1, "added": 1}


@pytest.mark.parametrize(
    "key, value, problem",
    [
        ("keyword_shape", "{bad", "keyword shape"),
        ("keyword_shape", '{"range_shift": 99}', "keyword shape"),
        ("keyword_shape", '{"colour": 1}', "keyword shape"),
        ("keywords_through", "-5", "no number"),
        ("keywords_through", None, "holds no keywords_through"),
    ],
)
def test_keyword_meta_that_is_damaged_or_missing_is_damage(key: str, value: str | None, problem: str) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        with writing(connection) as transaction:
            run(transaction, "DELETE FROM meta WHERE key = ?", (key,))
            if value is not None:
                run(transaction, "INSERT INTO meta(key, value) VALUES (?, ?)", (key, value))
        with pytest.raises(DamagedError, match=problem):
            applied(connection, [{"a.md": ("", "a", "b")}])


def test_a_prefix_expands_to_the_terms_it_begins() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "x", "super superb superconduct supra s")}])
        with reading(connection) as transaction:
            assert expansions(transaction, "super") == ["super", "superb", "superconduct"]
            assert expansions(transaction, "s") == ["s", "super", "superb", "superconduct", "supra"]
            assert expansions(transaction, "zz") == []
            assert expansions(transaction, chr(0x10FFFF)) == []
            assert expansions(transaction, "a" + chr(0x10FFFF)) == []
            assert expansions(transaction, chr(0xD7FF)) == []


def test_a_segment_of_chunks_all_gone_goes_with_them() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}, {"b.md": ("", "b", "three")}, {"a.md": None}])
        with reading(connection) as transaction:
            (left,) = segments(transaction)
            assert dead(transaction).tolist() == []
            assert frequencies(transaction, ["one", "three"]) == {"three": 1}
            assert left.chunks == 1


def test_a_chunk_deleted_outside_every_segment_is_damage() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}])
        with writing(connection) as transaction:
            run(transaction, "DELETE FROM keyword_segments")
        with pytest.raises(DamagedError, match="no segment of a chunk"):
            applied(connection, [{"a.md": None}])


def test_a_remake_with_nothing_left_to_index_is_done_at_once() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        staled(connection)
        assert remake_started(connection) == 0
        with reading(connection) as transaction:
            assert current(transaction)
        applied(connection, [NOTES])
        checked(connection)


def test_a_remake_cut_right_after_its_last_batch_is_done() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        staled(connection)
        remake_started(connection)
        batches = iter(remade(connection, Batches(chunks_a_segment=len(NOTES))))
        assert next(batches) == len(NOTES)
        with reading(connection) as transaction:
            assert current(transaction)
        assert remake_started(connection) == 0
        checked(connection)


def test_a_chunk_deleted_past_the_end_of_its_segment_is_damage() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two " * 5000)}])
        with writing(connection) as transaction:
            run(transaction, "UPDATE keyword_segments SET last = first")
        with pytest.raises(DamagedError, match="no segment of a chunk"):
            applied(connection, [{"a.md": None}])


def test_a_segment_written_after_deletions_averages_the_live_chunks() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        keeping = Batches(merging=Policy(dead_share=0.99))
        applied(
            connection, [{"a.md": ("", "a", "one " * 50), "b.md": ("", "b", "two " * 300)}, {"b.md": None}], keeping
        )
        with reading(connection) as transaction:
            tokens = totals(transaction).tokens
            (live,) = rows(transaction, Sized, "SELECT sum(chunks - dead), 0 FROM keyword_segments")
            assert rows(transaction, Sized, "SELECT sum(chunks), sum(dead) FROM keyword_segments")[0].dead > 0
        applied(connection, [{"c.md": ("", "c", "three")}], Batches(merging=Policy(dead_share=0.99)))
        with reading(connection) as transaction:
            made = rows(transaction, Average, "SELECT chunks, tokens, average FROM keyword_segments ORDER BY first")
        assert made[-1].average == pytest.approx((tokens + made[-1].tokens) / (live.chunks + made[-1].chunks))


MERGING = Batches(chunks_a_segment=2, documents_a_forget=1, postings_a_merge=3, merging=Policy(factor=2, floor=1))


class Sized(NamedTuple):
    chunks: int
    dead: int


def settled(connection: Connection, policy: Policy) -> list[Sized]:
    with reading(connection) as transaction:
        held = rows(transaction, Sized, "SELECT chunks, dead FROM keyword_segments ORDER BY first")
    assert merges([Held(each.chunks, each.dead) for each in held], policy) == ()
    return held


@settings(max_examples=30, deadline=None)
@given(rounds, st.sampled_from([SMALL, Shape()]))
def test_segments_merged_as_they_come_index_what_fts5_makes_of_the_live_chunks(each: list[Round], shape: Shape) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, shape) as connection:
        applied(connection, each, MERGING)
        checked(connection)
        settled(connection, MERGING.merging)


def test_one_chunk_written_at_a_time_merges_like_a_binary_counter() -> None:
    batches = Batches(merging=Policy(factor=2, floor=1, span=0.0))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        for n in range(8):
            applied(connection, [{f"n{n}.md": ("", f"Note {n}", "one two")}], batches)
        assert settled(connection, batches.merging) == [Sized(8, 0)]
        applied(connection, [{"n0.md": None, "n1.md": None, "n2.md": None}], batches)
        assert settled(connection, batches.merging) == [Sized(5, 0)]
        checked(connection)


def test_a_remake_merges_its_segments_as_it_goes() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        staled(connection)
        remake_started(connection)
        assert list(remade(connection, MERGING))[-1] == len(NOTES)
        checked(connection)
        settled(connection, MERGING.merging)


PAIRS = Batches(merging=Policy(factor=2, floor=1, span=0.0))
LONG: Round = {"a.md": ("", "a", "one two " * 5000)}


@pytest.mark.parametrize(
    "sql, problem",
    [
        ("DELETE FROM keyword_lengths", "lengths of some chunks are missing"),
        ("DELETE FROM chunks WHERE id = (SELECT min(id) FROM chunks)", "chunks the store does not"),
    ],
)
def test_segments_damaged_before_they_merge_are_damage(sql: str, problem: str) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}], PAIRS)
        with writing(connection) as transaction:
            run(transaction, sql)
        with pytest.raises(DamagedError, match=problem):
            applied(connection, [{"b.md": ("", "b", "three")}], PAIRS)


def test_a_segment_to_rewrite_whose_chunks_are_all_gone_is_damage() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [LONG])
        with writing(connection) as transaction:
            run(transaction, "DELETE FROM chunks")
            run(transaction, "UPDATE keyword_segments SET dead = chunks - 1")
        with pytest.raises(DamagedError, match="hold no chunk the store holds"):
            applied(connection, [{}])


def test_a_term_only_dead_chunks_held_is_gone_after_a_merge() -> None:
    batches = Batches(postings_a_merge=1, merging=Policy(factor=2, floor=1, span=0.0, dead_share=0.99))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        common: Content = ("", "c", "common")
        applied(connection, [{"a.md": ("", "a", "solo"), "b.md": common, "c.md": common}, {"a.md": None}], batches)
        applied(connection, [{"d.md": common, "e.md": common}], batches)
        assert settled(connection, batches.merging) == [Sized(4, 0)]
        with reading(connection) as transaction:
            assert rows(transaction, Term, "SELECT term FROM keyword_postings WHERE term = 'solo'") == []
        checked(connection)
