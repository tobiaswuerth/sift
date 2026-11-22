import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sift.domain.codec.segments import Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.segments import Policy
from sift.storage.database import Batches, reading, rows, run, writing
from sift.storage.keyword_lists import dead, expansions, frequencies, segments
from sift.storage.keywords import totals
from tests.keyword_indexes import (
    Average,
    Sized,
    checked,
    rounds,
)
from tests.keyword_stores import SMALL, Round, applied, opened_store


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


@pytest.mark.parametrize(
    "key, value, problem",
    [
        ("keyword_shape", "{bad", "keyword shape"),
        ("keyword_shape", '{"range_shift": 99}', "keyword shape"),
        ("keyword_shape", '{"colour": 1}', "keyword shape"),
        ("keywords_through", "-5", "no number"),
        ("keywords_through", str(1 << 63), "no number"),
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


def test_a_store_of_the_highest_collections_takes_writes() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 8_388_607, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}, {"a.md": None}])
        checked(connection)
