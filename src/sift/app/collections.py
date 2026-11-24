from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
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
from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.app.notes import unresolved_links
from sift.domain.text.records import Chosen
from sift.storage.catalog import (
    Choices,
    Collection,
    NewCollection,
    add_collection,
    known_collections,
    named,
    open_catalog,
)
from sift.storage.documents import counts, fields_sampled, unreadable
from sift.storage.links import links_held
from sift.storage.store import read_store, stored_bytes
from sift.storage.vector_meta import left, model_of

OBSIDIAN = ".obsidian"
FIELD_REMEDY = "name the field, e.g. abstract"


def _field(name: str | None, field: str) -> str | None:
    if name is not None and not name.strip():
        raise InvalidRequestError("names no field", FIELD_REMEDY, field)
    return name


def add(home: Path, request: AddCollection) -> CollectionAdded:
    folder = Path(request.folder)
    vault = (folder / OBSIDIAN).is_dir()
    title_from = request.title_from or (TitleFrom.NAME if vault else TitleFrom.HEADING)
    records = Chosen(_field(request.id_field, "id_field"), _field(request.title_field, "title_field"))
    records = replace(records, text=tuple(_field(each, "text_fields") or "" for each in request.text_fields))
    new = NewCollection(
        folder,
        request.name,
        Path(request.store) if request.store else None,
        nested=request.nested,
        choices=Choices(title_from, follow_links=request.follow_links, records=records),
    )
    with closing(open_catalog(home)) as catalog:
        added = add_collection(catalog, home, new)
    return CollectionAdded(
        added.name, str(added.folder), str(added.store), title_from, vault, follow_links=request.follow_links
    )


def chosen(home: Path, names: Sequence[str]) -> tuple[Collection, ...]:
    known = known_collections(home)
    return tuple(dict.fromkeys(named(known, name) for name in names)) if names else known


def chosen_or_active(home: Path, names: Sequence[str]) -> tuple[Collection, ...]:
    return chosen(home, names) if names else tuple(known for known in chosen(home, ()) if known.active)


def _contents(one: Collection, *, listed: bool) -> Counted | Unavailable:
    try:
        with read_store(one.store, one.id) as transaction:
            kept, failed = counts(transaction), unreadable(transaction)
            linked, gone = links_held(transaction), len(unresolved_links(transaction))
            waiting, model, fields = left(transaction), model_of(transaction), fields_sampled(transaction)
    except StoreUnavailableError as error:
        return Unavailable(error.message, error.remedy)
    files = tuple(UnreadableFile(entry.path, entry.problem) for entry in failed) if listed else ()
    size, embedded = stored_bytes(one.store), kept.chunks - waiting
    held = Counted(kept.documents, kept.chunks, size, len(failed), files, linked, gone)
    return replace(held, embedded=embedded, model=model, fields=fields)


def _status(one: Collection, *, listed: bool) -> CollectionStatus:
    return CollectionStatus(one.name, str(one.folder), str(one.store), _contents(one, listed=listed), one.active)


def status(home: Path, request: ShowStatus) -> Status:
    found = chosen(home, request.names)
    return Status(str(home), tuple(_status(one, listed=request.unreadable) for one in found))
