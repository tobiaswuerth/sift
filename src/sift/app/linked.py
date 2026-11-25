from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from sift.api.errors import InvalidRequestError, NotFoundError, StoreUnavailableError
from sift.api.filters import Filter
from sift.api.ids import DocumentId
from sift.api.search import Options
from sift.app.notes import NOTE_REMEDY, Note, Resolver, linking, note_of, read_in, target_of, written
from sift.domain.query.filters import Condition, Linked, LinkFilter, condition, linkings, readable
from sift.domain.text.resolving import LEADING
from sift.storage.catalog import Collection
from sift.storage.database import Reading
from sift.storage.links import links_out
from sift.storage.passing import reads_quotes


@dataclass(frozen=True, slots=True)
class Pointed:
    spec: str
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
    linkings: Mapping[LinkFilter, Pointed]
    links_to: tuple[Pointed, ...] | None
    linked_from: tuple[Pointed, ...] | None
    left_out: LeftOut | None


def _checked(spec: str, field: str) -> str:
    if not spec.strip():
        raise InvalidRequestError(f"takes a note's id or name, not {spec!r}", NOTE_REMEDY, field)
    return spec


def _held(one: Collection, target: str) -> bool:
    try:
        return read_in(one, lambda transaction: bool(written(transaction, target, None)))
    except StoreUnavailableError:
        return True


def _found(home: Path, spec: str, field: str) -> Pointed:
    named = note_of(home, spec, field)
    where = frozenset({named.note.collection.name, *(each.collection for each in named.unsearched if each.collection)})
    return Pointed(spec, named.note, target_of(spec), where)


def _linked_to(home: Path, scope: Sequence[Collection], spec: str) -> Pointed:
    try:
        return _found(home, _checked(spec, "links_to"), "links_to")
    except NotFoundError:
        target = target_of(spec)
        held = frozenset(one.name for one in scope if _held(one, target))
        if not held:
            raise
        return Pointed(spec, None, target, held)


def _linking_from(home: Path, spec: str) -> Pointed:
    return _found(home, _checked(spec, "linked_from"), "linked_from")


def _filtered(home: Path, each: LinkFilter) -> Pointed:
    try:
        return _found(home, each.target, "filters")
    except NotFoundError:
        return Pointed(each.target, None, each.target, frozenset())


def _reachable(scope: tuple[Collection, ...], pointed: Sequence[Pointed]) -> None:
    names = {one.name for one in scope}
    for each in pointed:
        if each.note is not None and not each.where & names:
            name = each.note.collection.name
            said = f"{each.spec!r} is a note of {name}, and links stay within their collection"
            raise InvalidRequestError(said, f"search the collection {name}, or every collection", "collections")


def _kept(
    scope: tuple[Collection, ...], links_to: tuple[Pointed, ...] | None, linked_from: tuple[Pointed, ...] | None
) -> tuple[Collection, ...]:
    _reachable(scope, (*(links_to or ()), *(linked_from or ())))
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
    readable(request.filters, reads_quotes=quotes)
    links_to = tuple(_linked_to(home, scope, each) for each in request.links_to) if request.links_to else None
    linked_from = tuple(_linking_from(home, each) for each in request.linked_from) if request.linked_from else None
    kept = _kept(scope, links_to, linked_from)
    filtered = MappingProxyType({each: _filtered(home, each) for each in linkings(request.filters)})
    return Narrowing(
        kept, len(scope) - len(kept), request.path, request.filters, quotes, filtered, links_to, linked_from, left_out
    )


def _document(note: Note | None, one: Collection, transaction: Reading) -> DocumentId | None:
    if note is None or note.collection.id != one.id:
        return None
    try:
        return note.find(transaction)
    except NotFoundError:
        return None


def _into(resolver: Resolver, one: Collection, pointed: Pointed, field: str | None) -> set[DocumentId]:
    if pointed.note is None:
        return written(resolver.transaction, pointed.target, field)
    document = _document(pointed.note, one, resolver.transaction)
    return set() if document is None else set(linking(resolver, document, field))


def _out_of(resolver: Resolver, one: Collection, pointed: Pointed) -> set[DocumentId]:
    document = _document(pointed.note, one, resolver.transaction)
    if document is None:
        return set()
    rows = links_out(resolver.transaction, document, None)
    return {how.documents[0] for how in map(resolver.of, rows) if how.via in LEADING} - {document}


def _within(found: Narrowing, one: Collection, resolver: Resolver) -> list[DocumentId] | None:
    kept: set[DocumentId] | None = None
    if found.links_to is not None:
        kept = {document for each in found.links_to for document in _into(resolver, one, each, None)}
    if found.linked_from is not None:
        to = {document for each in found.linked_from for document in _out_of(resolver, one, each)}
        kept = to if kept is None else kept & to
    return None if kept is None else sorted(kept)


def _left_out(left_out: LeftOut | None, one: Collection, resolver: Resolver) -> list[DocumentId]:
    if left_out is None:
        return []
    document = _document(left_out.note, one, resolver.transaction)
    if document is None:
        return sorted(written(resolver.transaction, left_out.note.name, None)) if left_out.linking else []
    return [document, *linking(resolver, document, None)] if left_out.linking else [document]


def condition_in(found: Narrowing, one: Collection, transaction: Reading) -> Condition | None:
    resolver = Resolver(transaction)
    values = {each: sorted(_into(resolver, one, pointed, each.field)) for each, pointed in found.linkings.items()}
    linked = Linked(values, _within(found, one, resolver), _left_out(found.left_out, one, resolver))
    return condition(found.path, found.filters, reads_quotes=found.reads_quotes, linked=linked)
