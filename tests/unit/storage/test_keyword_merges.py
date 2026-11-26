import tempfile
import tracemalloc
from pathlib import Path
from typing import NamedTuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.api.errors import StoreUnavailableError
from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.segments import Policy
from sift.storage.database import Batches, reading, rows, run, writing
from sift.storage.keyword_merges import MergedTerm, batched
from sift.storage.keywords import TAKEN_OVER_REMEDY, remade, remake_started, settle
from tests.examples import few
from tests.keyword_indexes import (
    LONG,
    MERGING,
    NOTES,
    PAIRS,
    Sized,
    Term,
    checked,
    content,
    rounds,
    settled,
    staled,
)
from tests.keyword_stores import SMALL, Content, Round, applied, opened_store


@few(30)
@given(rounds, st.sampled_from([SMALL, Shape()]))
def test_segments_merged_as_they_come_index_what_fts5_makes_of_the_live_chunks(each: list[Round], shape: Shape) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, shape) as connection:
        applied(connection, each, MERGING)
        checked(connection)
        settled(connection, MERGING.merging)


PIECES = Batches(chunks_a_segment=3, postings_a_merge=1, bytes_a_merge=1, merging=Policy(factor=2, floor=1))
DENSE = Batches(chunks_a_segment=3, tokens_a_segment=1, merging=Policy(factor=2, floor=1))


@few(30)
@given(rounds, st.sampled_from([SMALL, Shape()]), st.sampled_from([PIECES, DENSE]))
def test_terms_merged_a_row_at_a_time_index_what_fts5_makes_of_the_live_chunks(
    each: list[Round], shape: Shape, batches: Batches
) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, shape) as connection:
        applied(connection, each, batches)
        checked(connection)
        settled(connection, batches.merging)


class Stored(NamedTuple):
    first: int
    term: str
    number: int
    chunks: int
    postings: bytes | None
    positions: bytes | None
    ranges: bytes | None


STORED = """SELECT s.first, p.term, -1, p.chunks, p.postings, p.positions, p.ranges
            FROM keyword_postings p JOIN keyword_segments s ON s.id = p.segment
            UNION ALL
            SELECT s.first, b.term, b.number, 0, b.postings, b.positions, NULL
            FROM keyword_blocks b JOIN keyword_segments s ON s.id = b.segment
            ORDER BY 1, 2, 3"""
KEPT_APART = Batches(chunks_a_segment=3, merging=Policy(factor=10**6))
many_rounds = st.lists(
    st.dictionaries(st.sampled_from([f"{n}.md" for n in range(12)]), st.one_of(st.none(), content), max_size=12),
    min_size=1,
    max_size=4,
)


def merged_with(folder: Path, each: list[Round], shape: Shape, batches: Batches) -> list[Stored]:
    with opened_store(folder, 3, shape) as connection:
        applied(connection, each, KEPT_APART)
        list(settle(connection, batches))
        with reading(connection) as transaction:
            return rows(transaction, Stored, STORED)


@few(30)
@given(many_rounds, st.sampled_from([SMALL, Shape()]), st.sampled_from([PIECES, DENSE]))
def test_terms_merged_in_pieces_are_stored_byte_for_byte_as_merged_whole(
    each: list[Round], shape: Shape, batches: Batches
) -> None:
    with tempfile.TemporaryDirectory() as pieces, tempfile.TemporaryDirectory() as whole:
        cut = merged_with(Path(pieces), each, shape, batches)
        assert cut == merged_with(Path(whole), each, shape, Batches(merging=batches.merging))


@pytest.mark.parametrize("batches", [PIECES, DENSE])
def test_the_ranges_pieces_of_a_term_share_are_joined_as_a_whole_merge_joins_them(batches: Batches) -> None:
    each: list[Round] = [{f"{n:02}.md": ("", f"Note {n}", "common " * (n % 3 + 1)) for n in range(12)}]
    with tempfile.TemporaryDirectory() as pieces, tempfile.TemporaryDirectory() as whole:
        cut = merged_with(Path(pieces), each, SMALL, batches)
        assert cut == merged_with(Path(whole), each, SMALL, Batches(merging=batches.merging))
        assert any(row.ranges for row in cut)


@pytest.mark.parametrize(
    "halves",
    [
        Batches(postings_a_merge=8, merging=Policy(factor=2, floor=1, span=0.0)),
        Batches(tokens_a_segment=4096, merging=Policy(factor=2, floor=1, span=0.0)),
    ],
)
def test_a_term_too_large_for_a_merge_s_batch_is_merged_in_pieces_never_whole(halves: Batches) -> None:
    first_run, once_the_code_has_run = (peak_merging_two_segments_of_one_word(halves) for _ in range(2))
    assert (first_run > 0, once_the_code_has_run < 2 * 60_000 * 8) == (True, True)


def peak_merging_two_segments_of_one_word(halves: Batches) -> int:
    word: Content = ("", "w", "word " * 60_000)
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": word}, {"b.md": word}], Batches(merging=Policy(factor=10, floor=10**6)))
        with reading(connection) as transaction:
            assert len(rows(transaction, Sized, "SELECT chunks, dead FROM keyword_segments")) == 2
        tracemalloc.start()
        try:
            assert list(settle(connection, halves))[-1] > 0
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert len(settled(connection, halves.merging)) == 1
        checked(connection)
        return peak


def test_a_merge_reads_a_few_rows_of_each_segment_at_a_time_however_many_it_holds() -> None:
    small = Batches(postings_a_merge=8, tokens_a_segment=4096, merging=Policy(factor=2, floor=1, span=0.0))
    words = {name: ("", name, " ".join(f"{name}{n}" for n in range(5_000))) for name in ("a", "b")}
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(
            connection, [{"a.md": words["a"]}, {"b.md": words["b"]}], Batches(merging=Policy(factor=10, floor=10**6))
        )
        tracemalloc.start()
        try:
            assert list(settle(connection, small))[-1] > 0
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 1 << 20
        checked(connection)


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


@pytest.mark.parametrize("postings", [1, 2])
def test_a_term_only_dead_chunks_held_is_gone_after_a_merge(postings: int) -> None:
    batches = Batches(postings_a_merge=postings, merging=Policy(factor=2, floor=1, span=0.0, dead_share=0.99))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        common: Content = ("", "c", "common")
        applied(connection, [{"a.md": ("", "s", "solo"), "b.md": common, "c.md": common}, {"a.md": None}], batches)
        applied(connection, [{"d.md": common, "e.md": common}], batches)
        assert settled(connection, batches.merging) == [Sized(4, 0)]
        with reading(connection) as transaction:
            assert rows(transaction, Term, "SELECT term FROM keyword_postings WHERE term = 'solo'") == []
        checked(connection)


def test_merges_too_large_for_a_write_wait_for_settling_a_merge_a_transaction() -> None:
    batches = Batches(chunks_a_settle=1, merging=Policy(factor=2, floor=1, span=0.0))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        for n in range(4):
            applied(connection, [{f"n{n}.md": ("", f"Note {n}", "one two")}], batches)
        with reading(connection) as transaction:
            held = rows(transaction, Sized, "SELECT chunks, dead FROM keyword_segments ORDER BY first")
        assert len(held) == 4
        assert list(settle(connection, batches)) == [0, 2, 4, 8]
        assert settled(connection, batches.merging) == [Sized(4, 0)]
        checked(connection)


def test_settling_an_index_of_another_version_is_refused() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        staled(connection)
        with pytest.raises(StoreUnavailableError, match="another version"):
            list(settle(connection))


def test_merges_that_fit_a_write_run_though_a_larger_one_waits() -> None:
    batches = Batches(chunks_a_settle=2, merging=Policy(factor=2, floor=1, span=0.0))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        for n in range(12):
            applied(connection, [{f"n{n}.md": ("", f"Note {n}", "one two")}], batches)
        with reading(connection) as transaction:
            sized = rows(transaction, Sized, "SELECT chunks, dead FROM keyword_segments ORDER BY first")
            held = [each.chunks for each in sized]
        assert held == [2, 2, 2, 2, 2, 2]
        assert list(settle(connection, batches))[-1] > 0
        checked(connection)


def _shown(batches: Batches) -> list[tuple[str, list[str]]]:
    listed = [MergedTerm(0, term, 0, 3, 10, b"", b"") for term in ("a", "a", "b", "c")]
    return [(type(each).__name__, [row.term for row in each.rows]) for each in batched(iter(listed), batches)]


def test_merges_batch_terms_by_bytes_as_well_as_by_postings_and_a_term_past_either_in_pieces() -> None:
    assert _shown(Batches(bytes_a_merge=1)) == [("LargeTerm", ["a", "a"]), ("LargeTerm", ["b"]), ("LargeTerm", ["c"])]
    assert _shown(Batches(postings_a_merge=4)) == [("LargeTerm", ["a", "a"]), ("TermsBatch", ["b", "c"])]
    assert _shown(Batches(bytes_a_merge=30, postings_a_merge=10**9)) == [
        ("TermsBatch", ["a", "a", "b"]),
        ("TermsBatch", ["c"]),
    ]
    assert _shown(Batches()) == [("TermsBatch", ["a", "a", "b", "c"])]


def test_settling_or_writing_a_half_remade_index_says_to_finish_the_remake() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}])
        with writing(connection) as transaction:
            run(transaction, "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
        remake_started(connection)
        with pytest.raises(StoreUnavailableError, match="not yet wholly remade") as refused:
            list(settle(connection))
        assert refused.value.remedy == "index the collection to remake it"
        with pytest.raises(StoreUnavailableError, match="not yet wholly remade"):
            applied(connection, [{"b.md": ("", "b", "three")}])


def test_settling_or_writing_while_another_sift_remakes_the_index_for_itself_says_so() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}])
        with writing(connection) as transaction:
            run(transaction, "UPDATE meta SET value = 'remaking other' WHERE key = 'keyword_fingerprint'")
        with pytest.raises(StoreUnavailableError, match="another sift began remaking") as refused:
            list(settle(connection))
        assert refused.value.remedy == TAKEN_OVER_REMEDY
        with pytest.raises(StoreUnavailableError, match="another sift began remaking"):
            applied(connection, [{"b.md": ("", "b", "three")}])


def test_a_write_merges_no_more_than_its_budget_however_many_would_fit_alone() -> None:
    batches = Batches(chunks_a_segment=1, chunks_a_settle=2, merging=Policy(factor=2, floor=1, span=0.0))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{f"n{n}.md": ("", f"Note {n}", "one") for n in range(4)}], batches)
        with reading(connection) as transaction:
            sized = rows(transaction, Sized, "SELECT chunks, dead FROM keyword_segments ORDER BY first")
        assert [each.chunks for each in sized] == [2, 1, 1]
        checked(connection)
