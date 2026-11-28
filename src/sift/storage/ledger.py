from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from sift.api.ids import CollectionId
from sift.storage.database import Reading, Writing, rows, run
from sift.storage.store import of_another_collection
from sift.storage.store_files import store_files_left, unlinked

LEFTOVERS = """CREATE TABLE leftovers(
    store TEXT PRIMARY KEY,
    collection INTEGER NOT NULL
) STRICT, WITHOUT ROWID"""


class Owed(NamedTuple):
    store: str
    collection: int


def owe(transaction: Writing, store: Path, collection: CollectionId) -> None:
    run(transaction, "INSERT OR REPLACE INTO leftovers(store, collection) VALUES (?, ?)", (str(store), collection))


def owe_the_store_of(transaction: Writing, collection: CollectionId) -> None:
    sql = "INSERT OR REPLACE INTO leftovers(store, collection) SELECT store, id FROM collections WHERE id = ?"
    run(transaction, sql, (collection,))


def forgive(transaction: Writing, store: Path) -> None:
    run(transaction, "DELETE FROM leftovers WHERE store = ?", (str(store),))


def owed(transaction: Reading) -> list[Owed]:
    return rows(transaction, Owed, "SELECT store, collection FROM leftovers ORDER BY store")


def reclaimed(transaction: Writing, owing: Owed, unlink: Callable[[Path], None] = unlinked) -> tuple[Path, ...]:
    store = Path(owing.store)
    if of_another_collection(store, CollectionId(owing.collection)):
        forgive(transaction, store)
        return ()
    left = store_files_left(store, unlink)
    if not left:
        forgive(transaction, store)
    return left
