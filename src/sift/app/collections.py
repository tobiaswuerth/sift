from contextlib import closing
from pathlib import Path

from sift.api.collections import (
    AddCollection,
    CollectionAdded,
    CollectionStatus,
    Counted,
    ShowStatus,
    Status,
    TitleFrom,
    Unavailable,
    UnreadableFile,
)
from sift.api.errors import StoreUnavailableError
from sift.storage.catalog import (
    CATALOG,
    Choices,
    Collection,
    NewCollection,
    add_collection,
    collection,
    collections,
    open_catalog,
)
from sift.storage.database import reading
from sift.storage.documents import counts, unreadable
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
        choices=Choices(title_from, follow_links=request.follow_links),
    )
    with closing(open_catalog(home)) as catalog:
        added = add_collection(catalog, home, new)
    return CollectionAdded(
        added.name, str(added.folder), str(added.store), title_from, vault, follow_links=request.follow_links
    )


def _contents(chosen: Collection, *, listed: bool) -> Counted | Unavailable:
    try:
        connection = open_store(chosen.store, chosen.id)
    except StoreUnavailableError as error:
        return Unavailable(error.message, error.remedy)
    with closing(connection), reading(connection) as transaction:
        kept, failed = counts(transaction), unreadable(transaction)
    files = tuple(UnreadableFile(entry.path, entry.problem) for entry in failed) if listed else ()
    return Counted(kept.documents, kept.chunks, stored_bytes(chosen.store), len(failed), files)


def _status(chosen: Collection, *, listed: bool) -> CollectionStatus:
    return CollectionStatus(chosen.name, str(chosen.folder), str(chosen.store), _contents(chosen, listed=listed))


def _chosen(home: Path, names: tuple[str, ...]) -> tuple[Collection, ...]:
    with closing(open_catalog(home)) as catalog, reading(catalog) as transaction:
        if not names:
            return collections(transaction)
        return tuple(dict.fromkeys(collection(transaction, name) for name in names))


def status(home: Path, request: ShowStatus) -> Status:
    if not (home / CATALOG).is_file() and not request.names:
        return Status(str(home), ())
    chosen = _chosen(home, request.names)
    return Status(str(home), tuple(_status(each, listed=request.unreadable) for each in chosen))
