import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from sift.api.filters import Filter, Operator
from sift.api.ids import CollectionId, DocumentId
from sift.domain.query.filters import condition
from sift.domain.text.notes import note
from sift.storage.database import reading
from sift.storage.documents import add_documents, chunks_of, put_source
from sift.storage.keywords import writing_documents
from sift.storage.passing import Passing, reads_quotes
from sift.storage.store import create_store, open_store

COLLECTION = CollectionId(2)
LONG = "\n\n".join(["Ranges are scored best bound first. " * 20] * 6)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[tuple[sqlite3.Connection, dict[str, DocumentId]]]:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    with contextlib.closing(open_store(tmp_path / "s.sqlite", COLLECTION)) as connection:
        ids: dict[str, DocumentId] = {}
        for path, text in (
            ("papers/long.md", f"---\nyear: 2024\n---\n# Long\n\n{LONG}\n"),
            ("papers/short.md", "---\nyear: 2023\n---\n# Short\n\nBrief.\n"),
            ("notes/plain.md", "# Plain\n\nNo front matter.\n"),
        ):
            with writing_documents(connection) as transaction:
                source = put_source(transaction, path, 1, 2)
                (ids[path],) = add_documents(transaction, source, path, (note(text.encode(), path.rsplit("/", 1)[-1]),))
        yield connection, ids


def test_the_chunks_a_condition_lets_through_are_checked_once_and_listed_in_order(
    store: tuple[sqlite3.Connection, dict[str, DocumentId]],
) -> None:
    connection, ids = store
    asked = condition("papers/*", [Filter("year", Operator.ANY, (2024,))], reads_quotes=True)
    assert asked is not None
    with reading(connection) as transaction:
        long = [each.id for each in chunks_of(transaction, ids["papers/long.md"])]
        short = [each.id for each in chunks_of(transaction, ids["papers/short.md"])]
        plain = [each.id for each in chunks_of(transaction, ids["notes/plain.md"])]
        allowed = Passing(transaction, asked)
        asked_ids = np.array([*short, long[1], *plain, long[0], long[1]], np.int64)
        assert allowed.checked(asked_ids).tolist() == [False] * len(short) + [True] + [False] * len(plain) + [True] * 2
        assert allowed.checked(np.zeros(0, np.int64)).tolist() == []
        assert len(allowed.seen) == 3
        assert allowed.checked(np.array([long[0], 10**15], np.int64)).tolist() == [True, False]
        assert allowed.listed().tolist() == sorted(long)
        documents = np.array([ids["notes/plain.md"], ids["papers/long.md"], ids["papers/long.md"]], np.int64)
        assert allowed.documents(documents).tolist() == [False, True, True]
        assert allowed.listed() is allowed.listed()
    assert len(long) > 1


def test_whether_sqlite_reads_a_quote_in_a_field_s_name_is_asked_of_it() -> None:
    assert reads_quotes() == (sqlite3.sqlite_version_info >= (3, 47))
