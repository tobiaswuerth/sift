import contextlib
import dataclasses
from pathlib import Path
from typing import NamedTuple

from sift.api.ids import CollectionId, DocumentId, first_row
from sift.domain.text.notes import note
from sift.storage.database import Count, reading, rows
from sift.storage.documents import Counts, add_documents, counts, kept, put_source
from sift.storage.keywords import writing_documents
from sift.storage.store import create_store, open_store

COLLECTION = CollectionId(2)
MANY = 100


class Placed(NamedTuple):
    id: int
    document: int
    place: int


def test_documents_added_together_are_inserted_many_a_statement_and_numbered_as_one_by_one(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    records = [dataclasses.replace(note(b"x", "data.jsonl"), key=str(n)) for n in range(MANY)]
    statements: list[str] = []
    with contextlib.closing(open_store(tmp_path / "s.sqlite", COLLECTION)) as store:
        store.set_trace_callback(statements.append)
        with writing_documents(store) as transaction:
            source = put_source(transaction, "data.jsonl", 10, 20)
            ids = add_documents(transaction, source, "data.jsonl", records)
        store.set_trace_callback(None)
        with reading(store) as transaction:
            placed = rows(transaction, Placed, "SELECT id, document, place FROM chunks ORDER BY id")
            (unembedded,) = rows(transaction, Count, "SELECT count(*) FROM unembedded")
            keys = [each.key for each in kept(transaction, source)]
            held = counts(transaction)
    for table in ("documents", "chunks"):
        assert len({each for each in statements if each.startswith(f"INSERT INTO {table}(")}) == 1, table
    start = first_row(COLLECTION)
    assert ids == tuple(DocumentId(start + n) for n in range(1, MANY + 1))
    assert keys == [str(n) for n in range(MANY)]
    assert placed == [Placed(start + n, start + n, start + n) for n in range(1, MANY + 1)]
    assert (held, unembedded.n) == (Counts(MANY, MANY), MANY)
