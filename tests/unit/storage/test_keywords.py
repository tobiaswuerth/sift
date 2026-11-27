import dataclasses
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.segments import Policy
from sift.domain.text.notes import note
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS
from sift.storage.database import BATCHES, Batches, Count, reading, rows, run, writing
from sift.storage.documents import add_documents, put_source
from sift.storage.keyword_lists import dead_within, gone, prefixed, segments
from sift.storage.keywords import totals, writing_documents
from tests.examples import few
from tests.keyword_indexes import (
    Average,
    Sized,
    checked,
    frequencies,
    generation_of,
    rounds,
)
from tests.keyword_stores import SMALL, Round, applied, opened_store, written


@few(40)
@given(rounds, st.sampled_from([SMALL, Shape()]))
def test_the_keyword_index_holds_what_fts5_makes_of_the_live_chunks_however_they_changed(
    each: list[Round], shape: Shape
) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, shape) as connection:
        applied(connection, each)
        checked(connection)


@few(20)
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
            assert prefixed(transaction, "super", EVERY) == {"super": 1, "superb": 1, "superconduct": 1}
            assert sorted(prefixed(transaction, "s", EVERY)) == ["s", "super", "superb", "superconduct", "supra"]
            for none in ("zz", chr(0x10FFFF), "a" + chr(0x10FFFF), chr(0xD7FF)):
                assert prefixed(transaction, none, EVERY) == {}


KEEPING = Batches(merging=Policy(dead_share=0.99))
EVERY = 1 << 62


def test_a_term_counts_its_live_passages_alone_and_what_a_segment_counted_gone_leaves_with_it() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        both: Round = {"a.md": ("", "a", "super supra"), "b.md": ("", "b", "super")}
        applied(connection, [both, {"c.md": ("", "c", "supra")}, {"a.md": None}], KEEPING)
        with reading(connection) as transaction:
            assert [each.chunks for each in segments(transaction)] == [2, 1]
            assert prefixed(transaction, "su", EVERY) == {"super": 1, "supra": 1}
            assert frequencies(transaction, ["a", "super", "supra"]) == {"super": 1, "supra": 1}
            assert gone(transaction, ["b", "super", "supra"]) == {"super": 1, "supra": 1}
        applied(connection, [{"b.md": None}], KEEPING)
        with reading(connection) as transaction:
            assert prefixed(transaction, "", EVERY) == {"c": 1, "supra": 1}
            assert rows(transaction, Count, "SELECT count(*) FROM keyword_gone") == [Count(0)]


def test_a_pack_that_holds_other_than_its_head_says_is_damage_to_a_prefix() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "super supra")}])
        with writing(connection) as transaction:
            run(transaction, "UPDATE keyword_packs SET lists = substr(lists, 1, 6)")
        with reading(connection) as transaction, pytest.raises(DamagedError, match="a pack of lists"):
            prefixed(transaction, "su", EVERY)


def test_a_prefix_is_counted_no_further_than_makes_it_certain_to_begin_more_words_than_asked() -> None:
    ticks: list[int] = []

    def tick() -> int:
        ticks.append(1)
        return 0

    words = " ".join(f"w{number:05}" for number in range(30_000))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, Shape()) as connection:
        applied(connection, [{"a.md": ("", "a", words)}, {"b.md": ("", "b", "w1 w2 w3")}, {"b.md": None}], KEEPING)
        with reading(connection) as transaction:
            connection.set_progress_handler(tick, 100)
            certain = prefixed(transaction, "w", 10)
            cut = len(ticks)
            whole = prefixed(transaction, "w", EVERY)
            connection.set_progress_handler(None, 0)
    assert 10 < len(certain) < len(whole) > 20_000
    assert 3 * cut < len(ticks) - cut


def test_a_segment_is_written_with_no_statement_for_each_of_its_terms() -> None:
    words = " ".join(f"w{number}" for number in range(400))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, Shape()) as connection:
        applied(connection, [{"a.md": ("", "a", words)}])
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        applied(connection, [{"b.md": ("", "b", words)}])
        connection.set_trace_callback(None)
    assert len(statements) < 100


def test_a_segment_of_chunks_all_gone_goes_with_them() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}, {"b.md": ("", "b", "three")}, {"a.md": None}])
        with reading(connection) as transaction:
            (left,) = segments(transaction)
            assert dead_within(transaction, 0, 2**63 - 1).tolist() == []
            assert frequencies(transaction, ["one", "three"]) == {"three": 1}
            assert left.chunks == 1


def test_the_keyword_generation_moves_with_each_segment_made_or_gone_and_only_then() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        seen = [generation_of(connection)]
        changes: list[Round] = [{"a.md": ("", "a", "one two")}, {"b.md": ("", "b", "three")}, {"a.md": None}]
        for each in changes:
            applied(connection, [each])
            seen.append(generation_of(connection))
        seen.append(generation_of(connection))
        with reading(connection) as transaction:
            assert len(segments(transaction)) == 1
    assert seen == [0, 1, 2, 3, 3]


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
            made = rows(
                transaction, Average, "SELECT first, last, chunks, average FROM keyword_segments ORDER BY first"
            )
            added = totals(transaction).tokens - tokens
        assert made[-1].average == pytest.approx((tokens + added) / (live.chunks + made[-1].chunks))


def test_a_store_of_the_highest_collections_takes_writes() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 8_388_607, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}, {"a.md": None}])
        checked(connection)


def _tokens_with(title: str, about: str) -> int:
    made = note(written("", "a", "ordinary words " * 400), "a.md")
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        with writing_documents(connection, BATCHES) as transaction:
            source = put_source(transaction, "a.md", 0, 0)
            add_documents(
                transaction, source, "a.md", [dataclasses.replace(made, title=title, fields={"about": about})]
            )
        with reading(connection) as transaction:
            return totals(transaction).tokens


def test_a_document_s_title_and_fields_are_keywords_up_to_their_first_million_characters() -> None:
    filler = "filler " * (MOST_FIELD_CHARACTERS // len("filler ") + 1)
    longer = filler + "zebra " * 200_000
    assert _tokens_with(filler, filler) == _tokens_with(longer, longer)
    assert _tokens_with("a", "filler") < _tokens_with(filler, "filler") < _tokens_with(filler, filler)
