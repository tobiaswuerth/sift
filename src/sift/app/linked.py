from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import InvalidRequestError, NotFoundError, StoreUnavailableError
from sift.api.filters import Filter
from sift.api.search import Options, Unsearched
from sift.app.notes import Note, Resolver, at_path, by_id, linking, matches, passed, read_in, target_of, written
from sift.domain.query.filters import Condition, Linked, Linking, condition, linkings
from sift.domain.text.resolving import LEADING
from sift.storage.catalog import Collection
from sift.storage.database import Reading
from sift.storage.links import links_out
from sift.storage.passing import reads_quotes

NOTE_REMEDY = 'name a note by its id, as sift get takes it, or as links name it, e.g. "Acme AG"'


@dataclass(frozen=True, slots=True)
class Pointed:
    note: Note | None
    target: str
    where: frozenset[str]


@dataclass(frozen=True, slots=True)
class LeftOut:
    note: Note
    linking: bool


@dataclass(frozen=True, slots=True)
class Narrowing:
    collections: tuple[Collection, ...]
    ruled_out: int
    path: str | None
    filters: tuple[Filter, ...]
    reads_quotes: bool
    linkings: tuple[Linking, ...]
    links_to: tuple[Pointed, ...] | None
    linked_from: tuple[Pointed, ...] | None
    left_out: LeftOut | None


@dataclass(frozen=True, slots=True)
class Looked:
    note: Note | None
    unsearched: tuple[Unsearched, ...]


def _checked(spec: str, option: str) -> str:
    if not spec.strip():
        raise InvalidRequestError(f"{option} takes a note's id or name, not {spec!r}", NOTE_REMEDY)
    return spec


def _held(one: Collection, target: str) -> bool:
    try:
        return read_in(one, lambda transaction: bool(written(transaction, target, None)))
    except StoreUnavailableError:
        return True


def _there(note: Note) -> bool:
    try:
        read_in(note.collection, note.find)
    except NotFoundError:
        return False
    except StoreUnavailableError:
        return True
    return True


def _by_id(home: Path, spec: str) -> Note | None:
    try:
        note = by_id(home, spec)
    except NotFoundError:
        return None
    return note if _there(note) else None


def _note(home: Path, scope: Sequence[Collection], spec: str) -> Looked:
    note = _by_id(home, spec)
    if note is not None:
        return Looked(note, ())
    found = matches(scope, spec)
    return Looked(at_path(found.found[0]) if found.found else None, found.unsearched)


def _unread(looked: Looked) -> frozenset[str]:
    return frozenset(each.collection for each in looked.unsearched)


def _linked_to(home: Path, scope: Sequence[Collection], spec: str) -> Pointed:
    target, looked = target_of(spec), _note(home, scope, _checked(spec, "--links-to"))
    if looked.note is not None:
        return Pointed(looked.note, target, frozenset({looked.note.collection.name}) | _unread(looked))
    held = frozenset(one.name for one in scope if _held(one, target))
    if not held:
        raise NotFoundError(f"no note {spec!r} in the index, and no link to one", NOTE_REMEDY)
    return Pointed(None, target, held)


def _linking_from(home: Path, scope: Sequence[Collection], spec: str) -> Pointed:
    looked = _note(home, scope, _checked(spec, "--linked-from"))
    if looked.note is None:
        said = passed(f"no note {spec!r} in the index", looked.unsearched)
        raise NotFoundError(said, NOTE_REMEDY)
    return Pointed(looked.note, target_of(spec), frozenset({looked.note.collection.name}) | _unread(looked))


def _kept(
    scope: tuple[Collection, ...], links_to: tuple[Pointed, ...] | None, linked_from: tuple[Pointed, ...] | None
) -> tuple[Collection, ...]:
    names = {one.name for one in scope}
    if links_to is not None:
        names &= {name for each in links_to for name in each.where}
    if linked_from is not None:
        names &= {name for each in linked_from for name in each.where}
    return tuple(one for one in scope if one.name in names)


def narrowing(
    home: Path, request: Options, scope: tuple[Collection, ...], left_out: LeftOut | None = None
) -> Narrowing:
    quotes = reads_quotes()
    condition(request.path, request.filters, reads_quotes=quotes)
    links_to = tuple(_linked_to(home, scope, each) for each in request.links_to) if request.links_to else None
    linked_from = (
        tuple(_linking_from(home, scope, each) for each in request.linked_from) if request.linked_from else None
    )
    kept = _kept(scope, links_to, linked_from)
    return Narrowing(
        kept,
        len(scope) - len(kept),
        request.path,
        request.filters,
        quotes,
        linkings(request.filters),
        links_to,
        linked_from,
        left_out,
    )


def _following(resolver: Resolver, each: Linking) -> set[int]:
    how = resolver.of_name(each.target)
    if how.via in LEADING:
        return set(linking(resolver, how.documents[0], each.field))
    return written(resolver.transaction, each.target, each.field)


def _document(note: Note | None, one: Collection, transaction: Reading) -> int | None:
    if note is None or note.collection.id != one.id:
        return None
    try:
        return note.find(transaction)
    except NotFoundError:
        return None


def _into(resolver: Resolver, one: Collection, pointed: Pointed) -> set[int]:
    if pointed.note is None:
        return written(resolver.transaction, pointed.target, None)
    document = _document(pointed.note, one, resolver.transaction)
    return set() if document is None else set(linking(resolver, document, None))


def _out_of(resolver: Resolver, one: Collection, pointed: Pointed) -> set[int]:
    document = _document(pointed.note, one, resolver.transaction)
    if document is None:
        return set()
    rows = links_out(resolver.transaction, document, None)
    return {how.documents[0] for how in map(resolver.of, rows) if how.via in LEADING}


def _within(found: Narrowing, one: Collection, resolver: Resolver) -> list[int] | None:
    kept: set[int] | None = None
    if found.links_to is not None:
        kept = {document for each in found.links_to for document in _into(resolver, one, each)}
    if found.linked_from is not None:
        to = {document for each in found.linked_from for document in _out_of(resolver, one, each)}
        kept = to if kept is None else kept & to
    return None if kept is None else sorted(kept)


def _left_out(left_out: LeftOut | None, one: Collection, resolver: Resolver) -> list[int]:
    document = None if left_out is None else _document(left_out.note, one, resolver.transaction)
    if left_out is None or document is None:
        return []
    return [document, *linking(resolver, document, None)] if left_out.linking else [document]


def condition_in(found: Narrowing, one: Collection, transaction: Reading) -> Condition | None:
    resolver = Resolver(transaction)
    values = {each: sorted(_following(resolver, each)) for each in found.linkings}
    linked = Linked(values, _within(found, one, resolver), _left_out(found.left_out, one, resolver))
    return condition(found.path, found.filters, reads_quotes=found.reads_quotes, linked=linked)
