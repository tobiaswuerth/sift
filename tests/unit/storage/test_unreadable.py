import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from sift.api.ids import CollectionId
from sift.domain.codec.shape import SHAPE
from sift.storage.database import Connection, reading, run, writing
from sift.storage.documents import Mark, Unreadable, clear_unreadable, mark_unreadable, unreadable
from sift.storage.store import LAYOUT, create_store, open_store
from tests.keyword_stores import opened_store

COLLECTION = CollectionId(2)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Connection]:
    with opened_store(tmp_path, COLLECTION, SHAPE) as connection:
        yield connection


def test_unreadable_files_are_listed_with_their_problem_until_cleared(store: Connection) -> None:
    with writing(store) as transaction:
        mark_unreadable(transaction, "b.md", Mark("denied", read_again=True))
        mark_unreadable(transaction, "a.md", Mark("denied", read_again=True))
        mark_unreadable(transaction, "a.md", Mark("too large", read_again=False))
        mark_unreadable(transaction, "c.pdf", Mark("a scan; SIFT_OCR is off", read_again=False, skipped_under="off"))
        mark_unreadable(transaction, "d.pdf", Mark("its reader hung", read_again=False, stopped_under="2.0.0"))
    with reading(store) as transaction:
        assert unreadable(transaction) == [
            Unreadable("a.md", "too large", 0, None, None),
            Unreadable("b.md", "denied", 1, None, None),
            Unreadable("c.pdf", "a scan; SIFT_OCR is off", 0, "off", None),
            Unreadable("d.pdf", "its reader hung", 0, None, "2.0.0"),
        ]
    with writing(store) as transaction:
        clear_unreadable(transaction, ["a.md", "gone.md", "c.pdf", "d.pdf"])
    with reading(store) as transaction:
        assert unreadable(transaction) == [Unreadable("b.md", "denied", 1, None, None)]


@pytest.mark.parametrize("setting", ["on", "sometimes"])
def test_a_file_is_left_unread_only_under_a_setting_that_can_read_more(store: Connection, setting: str) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"), writing(store) as transaction:
        mark_unreadable(transaction, "c.pdf", Mark("a scan", read_again=False, skipped_under=setting))


def test_the_unreadable_files_of_an_older_store_were_left_unread_under_no_setting(tmp_path: Path) -> None:
    path = tmp_path / "older.sqlite"
    create_store(path, COLLECTION, LAYOUT[:6])
    with contextlib.closing(open_store(path, COLLECTION, LAYOUT[:6])) as connection, writing(connection) as transaction:
        run(transaction, "INSERT INTO unreadable(path, problem, read_again) VALUES ('a.pdf', 'no text in it', 0)")
    with contextlib.closing(open_store(path, COLLECTION)) as connection, reading(connection) as transaction:
        assert unreadable(transaction) == [Unreadable("a.pdf", "no text in it", 0, None, None)]
