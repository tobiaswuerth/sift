import contextlib
import tempfile
import threading
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path

import pytest

from sift.api.errors import StoreUnavailableError
from sift.domain.errors import DamagedError
from sift.storage.database import Batches, Connection, Reading, rows, run, writing
from sift.storage.keyword_beside import built
from sift.storage.keyword_reads import STUCK_REMEDY, Named, Passages, Span, passages_at_once, read_ahead, read_in
from sift.storage.keywords import remade, remake_started, writing_keywords
from tests.keyword_indexes import NOTES, TWO, checked, staled
from tests.keyword_orders import placed, placed_in
from tests.keyword_stores import SMALL, applied, opened_store
from tests.waits import until, waited

SPANS = "SELECT place AS id, document, start, length FROM chunks ORDER BY place DESC"
PARTS = Batches(chunks_a_read=3, chunks_a_read_beside=1, readers=2)
READ_BY_THE_WRITER = ("keyword_documents", "JOIN texts")


@contextlib.contextmanager
def noted() -> Generator[Connection]:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        yield connection


def spans_of(transaction: Reading) -> list[Span]:
    return rows(transaction, Span, SPANS)


def spans_in(passages: Iterable[Passages]) -> list[Span]:
    return [span for each in passages for span in each.spans]


def passages_by_span(passages: Iterable[Passages]) -> list[tuple[Span, str, Named]]:
    return [
        (span, text, next(held for held in each.documents if held.id == span.document))
        for each in passages
        for span, text in zip(each.spans, each.texts, strict=True)
    ]


def readers_left() -> list[str]:
    return [each.name for each in threading.enumerate() if each.name.startswith("sift-read")]


@contextlib.contextmanager
def traced(connection: Connection) -> Generator[list[str]]:
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    try:
        yield statements
    finally:
        connection.set_trace_callback(None)


def reads_of(statements: Sequence[str]) -> list[str]:
    return [each for each in statements if any(read in each for read in READ_BY_THE_WRITER)]


def test_passages_read_ahead_beside_the_writer_are_those_it_reads_itself_part_by_part_in_order() -> None:
    with noted() as connection, writing_keywords(connection, PARTS) as transaction:
        spans = spans_of(transaction)
        with read_in(transaction, spans) as reads:
            inside = list(reads)
        with traced(connection) as statements, read_ahead(transaction, spans) as reads:
            ahead = list(reads)
    assert passages_by_span(ahead) == passages_by_span(inside)
    assert spans_in(ahead) == spans_in(inside) == spans
    assert [len(each.spans) for each in inside] == [len(spans[at : at + 3]) for at in range(0, len(spans), 3)]
    assert [len(each.spans) for each in ahead] == [1] * len(spans)
    assert all({span.document for span in each.spans} == {held.id for held in each.documents} for each in ahead)
    assert reads_of(statements) == []
    assert readers_left() == []


def test_reading_ahead_left_early_stops_its_readers() -> None:
    with noted() as connection, writing_keywords(connection, PARTS) as transaction:
        spans = spans_of(transaction)
        with read_ahead(transaction, spans) as reads:
            first = next(reads)
    assert (list(first.spans), readers_left()) == (spans[:1], [])


def test_no_spans_are_read_as_no_parts() -> None:
    with (
        noted() as connection,
        writing_keywords(connection, PARTS) as transaction,
        read_ahead(transaction, []) as reads,
    ):
        assert list(reads) == []


def test_a_passage_a_reader_cannot_cut_is_damage_where_the_passages_are_taken() -> None:
    with noted() as connection:
        with writing(connection) as transaction:
            spans = spans_of(transaction)
            run(transaction, "DELETE FROM texts WHERE document = ?", (spans[-1].document,))
        with (
            writing_keywords(connection, PARTS) as transaction,
            read_ahead(transaction, spans) as reads,
            pytest.raises(DamagedError, match="past the text of its document"),
        ):
            list(reads)
    assert readers_left() == []


def test_a_part_not_read_in_time_says_reading_the_store_is_stuck_without_waiting_for_it() -> None:
    release = threading.Event()

    def stuck(transaction: Reading, spans: Sequence[Span]) -> Passages:
        waited(release)
        return passages_at_once(transaction, spans)

    with noted() as connection, writing_keywords(connection, PARTS) as transaction:
        spans = spans_of(transaction)
        with (
            pytest.raises(
                StoreUnavailableError, match="passages were not read within 0 s: reading the store is stuck"
            ) as told,
            read_ahead(transaction, spans, read=stuck, waited=0.01) as reads,
        ):
            list(reads)
        left = readers_left()
        release.set()
    assert (left != [], told.value.remedy, until(lambda: readers_left() == [])) == (True, STUCK_REMEDY, True)


def test_an_order_and_a_remake_read_their_passages_beside_the_writer_and_new_documents_where_they_are_written() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        with traced(connection) as statements:
            applied(connection, [NOTES])
        assert reads_of(statements) != []
        ids = [each.id for each in placed(connection)]
        placed_in(connection, ids[::-1])
        with traced(connection) as statements:
            assert max(built(connection, TWO)) == len(ids)
        assert reads_of(statements) == []
        staled(connection)
        remake_started(connection)
        with traced(connection) as statements:
            assert max(remade(connection, TWO)) == len(ids)
        assert reads_of(statements) == []
        checked(connection)
