import contextlib
import json
import sqlite3
from collections.abc import Generator
from pathlib import Path

from sift.api.fields import Fields
from sift.api.ids import CollectionId
from sift.storage.database import reading, run, run_each, writing
from sift.storage.documents import fields_sampled
from sift.storage.store import create_store, open_store

COLLECTION = CollectionId(3)


@contextlib.contextmanager
def stored(folder: Path, held: list[Fields]) -> Generator[sqlite3.Connection]:
    path = folder / "s.sqlite"
    create_store(path, COLLECTION)
    with contextlib.closing(open_store(path, COLLECTION)) as connection:
        with writing(connection) as transaction:
            run(transaction, "INSERT INTO sources(id, path, modified_ns, size, folded) VALUES (1, 'a.jsonl', 0, 0, '')")
            run_each(
                transaction,
                "INSERT INTO documents(source, key, digest, title, fields) VALUES (1, ?, 'd', 't', ?)",
                [(str(at), json.dumps(fields)) for at, fields in enumerate(held)],
            )
        yield connection


def test_the_fields_a_store_holds_are_sampled_from_its_first_document_to_its_last(tmp_path: Path) -> None:
    with stored(tmp_path, [{"first": 1}, *[{}] * 998, {"last": 1}]) as connection, reading(connection) as transaction:
        assert fields_sampled(transaction, 2) == ("first", "last")
        assert fields_sampled(transaction) == ("first", "last")


def test_a_store_without_documents_holds_no_fields(tmp_path: Path) -> None:
    with stored(tmp_path, []) as connection, reading(connection) as transaction:
        assert fields_sampled(transaction) == ()


def test_the_fields_of_a_large_store_are_read_from_the_documents_sampled_alone(tmp_path: Path) -> None:
    ticks: list[int] = []

    def tick() -> int:
        ticks.append(1)
        return 0

    with stored(tmp_path, [{"year": 2024, "Straße": "x"}] * 20_000) as connection:
        connection.set_progress_handler(tick, 100)
        with reading(connection) as transaction:
            found = fields_sampled(transaction, 10)
        connection.set_progress_handler(None, 0)
    assert (found, len(ticks) < 100) == (("Straße", "year"), True)
