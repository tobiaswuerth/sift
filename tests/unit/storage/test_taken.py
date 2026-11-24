import contextlib
from pathlib import Path
from typing import NamedTuple

from sift.api.ids import CollectionId
from sift.storage.database import Parameters, Reading, rows, writing
from sift.storage.store import create_store, open_store
from sift.storage.taken import KEYED, UNTAKEN, nothing_taken

COLLECTION = CollectionId(3)


class Step(NamedTuple):
    id: int
    parent: int
    unused: int
    detail: str


def planned(transaction: Reading, sql: str, parameters: Parameters) -> list[str]:
    return [step.detail for step in rows(transaction, Step, f"EXPLAIN QUERY PLAN {sql}", parameters)]


def test_the_documents_of_a_file_stored_again_are_looked_up_by_its_source_and_their_keys_never_scanned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with contextlib.closing(open_store(path, COLLECTION)) as connection, writing(connection) as transaction:
        nothing_taken(transaction)
        keyed = planned(transaction, KEYED, {"source": 1, "keys": "[]"})
        untaken = planned(transaction, UNTAKEN, (1,))
    assert any(step.startswith("SEARCH documents USING") and "(source=? AND key=?)" in step for step in keyed)
    assert any(step.startswith("SEARCH documents USING") and "(source=?)" in step for step in untaken)
    assert not [step for step in (*keyed, *untaken) if step.startswith("SCAN documents")]
