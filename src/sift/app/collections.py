from collections.abc import Sequence
from contextlib import closing, nullcontext
from dataclasses import replace
from pathlib import Path

from sift.api.collections import (
    ActivateCollections,
    AddCollection,
    CollectionAdded,
    CollectionRemoved,
    CollectionsSwitched,
    CollectionStatus,
    Counted,
    DeactivateCollections,
    RemoveCollection,
    ShowStatus,
    Status,
    TitleFrom,
    Unavailable,
    UnreadableFile,
)
from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.app.notes import unresolved_links
from sift.domain.text.records import Chosen
from sift.storage.catalog import (
    Choices,
    Collection,
    NewCollection,
    add_collection,
    forget_collection,
    known_collections,
    named,
    open_catalog,
    refuse_inactive,
    set_active,
)
from sift.storage.documents import counts, fields_sampled, unreadable
from sift.storage.links import links_held
from sift.storage.store import read_store
from sift.storage.store_files import store_files_left, stored_bytes
from sift.storage.vector_meta import left, model_of
from sift.storage.writer import sole_writer

OBSIDIAN = ".obsidian"
FIELD_REMEDY = "name the field, e.g. abstract"
NONE_ACTIVE_REMEDY = "run sift activate with a collection's name"


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


def searchable(home: Path, names: Sequence[str]) -> tuple[Collection, ...]:
    if names:
        return tuple(map(refuse_inactive, chosen(home, names)))
    known = chosen(home, ())
    active = tuple(one for one in known if one.active)
    if known and not active:
        raise InvalidRequestError("no collection is active", NONE_ACTIVE_REMEDY, code=Problem.INACTIVE_COLLECTION)
    return active


def _held(one: Collection) -> tuple[int | None, str | None]:
    try:
        with read_store(one.store, one.id) as transaction:
            return counts(transaction).documents, None
    except StoreUnavailableError as error:
        return None, error.message


def remove(home: Path, request: RemoveCollection) -> CollectionRemoved:
    one = named(known_collections(home), request.name)
    with sole_writer(one.store, "sift rm") if one.store.parent.is_dir() else nullcontext():
        documents, problem = _held(one)
        forget_collection(home, one.id)
    left = tuple(str(file) for file in store_files_left(one.store))
    return CollectionRemoved(one.name, str(one.store), documents, problem, left)


def _switched(home: Path, names: Sequence[str], *, active: bool) -> CollectionsSwitched:
    switched = set_active(home, names, active=active)
    changed = tuple(one.name for one in switched if one.active != active)
    return CollectionsSwitched(active, changed, tuple(one.name for one in switched if one.active == active))


def activate(home: Path, request: ActivateCollections) -> CollectionsSwitched:
    return _switched(home, request.names, active=True)


def deactivate(home: Path, request: DeactivateCollections) -> CollectionsSwitched:
    return _switched(home, request.names, active=False)


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
