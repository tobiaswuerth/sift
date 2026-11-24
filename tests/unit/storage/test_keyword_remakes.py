import contextlib
import tempfile
from pathlib import Path

import pytest

from sift.api.errors import StoreUnavailableError
from sift.api.ids import CollectionId
from sift.storage.database import Batches, reading, rows, run, writing
from sift.storage.keyword_lists import frequencies
from sift.storage.keywords import REMAKE_REMEDY, current, remade, remake_started
from sift.storage.layout import LAYOUT
from sift.storage.store import create_store, open_store
from tests.keyword_indexes import (
    NOTES,
    TWO,
    Average,
    checked,
    chunk_tokens,
    staled,
)
from tests.keyword_stores import SMALL, applied, opened_store


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


def test_a_store_of_the_layout_before_the_keyword_index_gets_one_when_remade() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "3.sqlite"
        create_store(path, CollectionId(3), LAYOUT[:1])
        with (
            contextlib.closing(open_store(path, CollectionId(3), LAYOUT[:1])) as connection,
            writing(connection) as transaction,
        ):
            run(transaction, "DELETE FROM meta WHERE key LIKE 'keyword%'")
            run(transaction, "INSERT INTO sources(id, path, modified_ns, size) VALUES (1, 'a.md', 0, 0)")
            body = "# Old\n\nkept from before\n"
            run(
                transaction,
                (
                    "INSERT INTO documents(source, key, digest, title, fields, text) "
                    "VALUES (1, '', 'd', 'Old', '{\"tags\": [\"old\"]}', ?)"
                ),
                (body,),
            )
            run(
                transaction,
                (
                    "INSERT INTO chunks(document, first_line, last_line, start, length) "
                    "SELECT id, 1, 3, 0, ? FROM documents"
                ),
                (len(body.strip()),),
            )
        with contextlib.closing(open_store(path, CollectionId(3))) as connection:
            assert remake_started(connection) == 1
            assert list(remade(connection)) == [1]
            applied(connection, [{"b.md": ("", "New", "added after")}])
            checked(connection)
            with reading(connection) as transaction:
                assert frequencies(transaction, ["old", "added"]) == {"old": 1, "added": 1}


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
