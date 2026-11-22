from contextlib import closing
from pathlib import Path

from sift.api.collections import (
    AddCollection,
    CollectionAdded,
    CollectionStatus,
    Counted,
    Status,
    TitleFrom,
    Unavailable,
)
from sift.api.errors import StoreUnavailableError
from sift.storage.catalog import CATALOG, Choices, Collection, NewCollection, add_collection, collections, open_catalog
from sift.storage.database import reading
from sift.storage.documents import counts
from sift.storage.store import open_store, stored_bytes

OBSIDIAN = ".obsidian"


def add(home: Path, request: AddCollection) -> CollectionAdded:
    folder = Path(request.folder)
    vault = (folder / OBSIDIAN).is_dir()
    title_from = request.title_from or (TitleFrom.NAME if vault else TitleFrom.HEADING)
    new = NewCollection(
        folder,
        request.name,
        Path(request.store) if request.store else None,
        nested=request.nested,
        choices=Choices(title_from),
    )
    with closing(open_catalog(home)) as catalog:
        added = add_collection(catalog, home, new)
    return CollectionAdded(added.name, str(added.folder), str(added.store), title_from, vault)


def _contents(collection: Collection) -> Counted | Unavailable:
    try:
        connection = open_store(collection.store, collection.id)
    except StoreUnavailableError as error:
        return Unavailable(error.message, error.remedy)
    with closing(connection), reading(connection) as transaction:
        kept = counts(transaction)
    return Counted(kept.documents, kept.chunks, stored_bytes(collection.store))


def _status(collection: Collection) -> CollectionStatus:
    return CollectionStatus(collection.name, str(collection.folder), str(collection.store), _contents(collection))


def status(home: Path) -> Status:
    if not (home / CATALOG).is_file():
        return Status(str(home), ())
    with closing(open_catalog(home)) as catalog, reading(catalog) as transaction:
        known = collections(transaction)
    return Status(str(home), tuple(map(_status, known)))
