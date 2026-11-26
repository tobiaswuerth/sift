import tempfile
from pathlib import Path

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.domain.codec.shape import FORMAT
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS, TABLE_DIGEST
from sift.storage.database import Batches, Connection, Value, reading, rows, run, writing
from sift.storage.keyword_lists import frequencies
from sift.storage.keywords import REMAKE_REMEDY, current, refuse_stale, remade, remake_started, settle
from tests.keyword_indexes import (
    NOTES,
    TWO,
    Average,
    checked,
    chunk_tokens,
    staled,
)
from tests.keyword_orders import placed, placed_in
from tests.keyword_stores import SMALL, applied, opened_store

FORMAT_HELD = "SELECT value FROM meta WHERE key = 'keyword_format'"
MADE_OF = {1: ("32642b01ffce29e28e6f6e9e4b9fd5e8", 1_048_576)}


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
            made = rows(
                transaction, Average, "SELECT first, last, chunks, average FROM keyword_segments ORDER BY first"
            )
            counted = chunk_tokens(transaction)
        held = [sum(n for chunk, n in counted.items() if m.first <= chunk <= m.last) for m in made]
        assert [each.average for each in made] == pytest.approx(
            [sum(held[: n + 1]) / sum(m.chunks for m in made[: n + 1]) for n in range(len(made))]
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


def of_a_newer_format(connection: Connection) -> None:
    with writing(connection) as transaction:
        run(transaction, "UPDATE meta SET value = ? WHERE key = 'keyword_format'", (str(FORMAT + 1),))
        run(transaction, "UPDATE meta SET value = 'of a newer sift' WHERE key = 'keyword_fingerprint'")


def test_an_index_of_a_newer_format_is_refused_saying_to_upgrade_and_is_never_made_anew() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        with reading(connection) as transaction:
            held = frequencies(transaction, ["one", "note"])
        of_a_newer_format(connection)
        ways = (
            lambda: applied(connection, [{"x.md": ("", "X", "four")}]),
            lambda: remake_started(connection),
            lambda: list(settle(connection)),
            lambda: placed_in(connection, [each.id for each in placed(connection)][::-1]),
        )
        for way in ways:
            with pytest.raises(StoreUnavailableError) as refused:
                way()
            assert refused.value.message == "the store's keyword index was made by a newer sift"
            assert (refused.value.remedy, refused.value.code) == ("upgrade sift", Problem.STORE_NEWER)
        with reading(connection) as transaction:
            with pytest.raises(StoreUnavailableError, match="made by a newer sift"):
                refuse_stale(transaction)
            assert frequencies(transaction, ["one", "note"]) == held


def test_a_new_store_says_the_format_of_its_keyword_index_and_a_remake_says_it_anew() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        with reading(connection) as transaction:
            assert rows(transaction, Value, FORMAT_HELD) == [Value(str(FORMAT))]
        with writing(connection) as transaction:
            run(transaction, "UPDATE meta SET value = '0' WHERE key = 'keyword_format'")
        staled(connection)
        assert remake_started(connection) == len(NOTES)
        list(remade(connection))
        with reading(connection) as transaction:
            assert rows(transaction, Value, FORMAT_HELD) == [Value(str(FORMAT))]
        checked(connection)


def test_what_the_terms_of_a_keyword_index_are_made_of_changes_only_with_its_format_number() -> None:
    assert MADE_OF[FORMAT] == (TABLE_DIGEST, MOST_FIELD_CHARACTERS)


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


def test_a_remake_another_sift_of_the_same_version_finished_is_done() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        staled(connection)
        remake_started(connection)
        ours = iter(remade(connection, TWO))
        assert next(ours) == 2
        assert list(remade(connection))[-1] == len(NOTES) - 2
        assert list(ours) == []
        with reading(connection) as transaction:
            assert current(transaction)
        checked(connection)
