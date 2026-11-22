from collections.abc import Sequence
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
    Choices,
    Collection,
    NewCollection,
    add_collection,
    known_collections,
    named,
    open_catalog,
)
from sift.storage.database import reading
from sift.storage.documents import counts, unreadable
from sift.storage.store import open_store, stored_bytes, unavailable_if_damaged

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


def chosen(home: Path, names: Sequence[str]) -> tuple[Collection, ...]:
    known = known_collections(home)
    return tuple(dict.fromkeys(named(known, name) for name in names)) if names else known


def _contents(one: Collection, *, listed: bool) -> Counted | Unavailable:
    try:
        with (
            unavailable_if_damaged(one.store),
            closing(open_store(one.store, one.id)) as connection,
            reading(connection) as transaction,
        ):
            kept, failed = counts(transaction), unreadable(transaction)
    except StoreUnavailableError as error:
        return Unavailable(error.message, error.remedy)
    files = tuple(UnreadableFile(entry.path, entry.problem) for entry in failed) if listed else ()
    return Counted(kept.documents, kept.chunks, stored_bytes(one.store), len(failed), files)


def _status(one: Collection, *, listed: bool) -> CollectionStatus:
    return CollectionStatus(one.name, str(one.folder), str(one.store), _contents(one, listed=listed))


def status(home: Path, request: ShowStatus) -> Status:
    found = chosen(home, request.names)
    return Status(str(home), tuple(_status(one, listed=request.unreadable) for one in found))
