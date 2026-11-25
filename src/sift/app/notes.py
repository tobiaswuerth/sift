from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from sift.api.errors import InvalidRequestError, NotFoundError, Problem, StoreUnavailableError, field_at_fault
from sift.api.ids import DocumentId
from sift.api.notices import Notice, notice_of
from sift.app.documents import place, resolved_at
from sift.app.documents import unknown as documents_unknown
from sift.domain.text.kinds import public_id
from sift.domain.text.links import LinkKind, NamedTarget, name_key, named_target, target_key
from sift.domain.text.resolving import LEADING, Candidate, Resolution, Via, link_path, resolved
from sift.storage.catalog import Collection, known_collections
from sift.storage.database import Reading
from sift.storage.documents import stored
from sift.storage.links import LinkRow, doubtful, keys_of, links_into, links_keyed, links_out, named, titled
from sift.storage.store import read_store

AMBIGUOUS_SHOWN = 5
AMBIGUOUS_REMEDY = "give the note's whole id"
NOTE_REMEDY = 'name a note by its id, or as links name it, e.g. "Acme AG"'

type NameCandidates = tuple[list[Candidate], list[Candidate]]


@dataclass(slots=True)
class Resolver:
    transaction: Reading
    seen: dict[str, NameCandidates] = field(default_factory=dict[str, NameCandidates])

    def _named(self, key: str) -> NameCandidates:
        if key not in self.seen:
            self.seen[key] = named(self.transaction, key)
        return self.seen[key]

    def of(self, row: LinkRow) -> Resolution:
        return resolved(row.path, row.kind, row.target, *self._named(row.key))

    def of_name(self, name: NamedTarget) -> Resolution:
        return resolved(None, LinkKind.WIKILINK, name.target, *self._named(name.key))


@dataclass(frozen=True, slots=True)
class Note:
    collection: Collection
    path: str
    key: str = ""

    @property
    def id(self) -> str:
        return public_id(self.collection.name, self.path, self.key)

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).stem

    def find(self, transaction: Reading) -> DocumentId:
        found = stored(transaction, self.path, self.key)
        if found is None:
            raise documents_unknown(self.id)
        return found.id


@dataclass(frozen=True, slots=True)
class NamedNote:
    note: Note
    unsearched: tuple[Notice, ...]


@dataclass(frozen=True, slots=True)
class Matches:
    found: tuple[Note, ...]
    unsearched: tuple[Notice, ...]


def read_in[T](one: Collection, read: Callable[[Reading], T]) -> T:
    with read_store(one.store, one.id) as transaction:
        return read(transaction)


def _named(transaction: Reading, name: NamedTarget) -> list[int]:
    found = Resolver(transaction).of_name(name)
    return list(found.documents) if found.via is not Via.UNRESOLVED else []


def _matches(one: Collection, name: NamedTarget) -> list[Note]:
    def read(transaction: Reading) -> list[Note]:
        found = _named(transaction, name)
        held = sorted(titled(transaction, found).values(), key=lambda each: (each.path, each.key))
        return [Note(one, each.path, each.key) for each in held]

    return read_in(one, read)


def passed(message: str, unsearched: Sequence[Notice]) -> str:
    return "; ".join([message, *(f"not searched: {each.collection}: {each.message}" for each in unsearched)])


def _remedy(remedy: str, unsearched: Sequence[Notice]) -> str:
    found = [each.remedy for each in unsearched if each.remedy is not None]
    return "; ".join(dict.fromkeys([*found, remedy]))


def matches(collections: Sequence[Collection], id_: str) -> Matches:
    name, unread = named_target(id_), list[Notice]()
    found: list[Note] = []
    for one in collections:
        try:
            found += _matches(one, name)
        except StoreUnavailableError as error:
            unread.append(notice_of(error, one.name))
    if len(found) > 1:
        listed = ", ".join(sorted(each.id for each in found)[:AMBIGUOUS_SHOWN])
        more = ", ..." if len(found) > AMBIGUOUS_SHOWN else ""
        said = passed(f"{id_!r} names several notes: {listed}{more}", unread)
        raise InvalidRequestError(said, _remedy(AMBIGUOUS_REMEDY, unread), code=Problem.AMBIGUOUS)
    return Matches(tuple(found), tuple(unread))


def unknown_note(spec: str, unsearched: Sequence[Notice]) -> NotFoundError:
    said = passed(f"no note or document {spec!r}", unsearched)
    return NotFoundError(said, _remedy(NOTE_REMEDY, unsearched), code=Problem.UNKNOWN_NOTE)


def _by_id(home: Path, spec: str) -> Note:
    chosen, relative = place(known_collections(home), spec)

    def read(transaction: Reading) -> Note:
        found = resolved_at(transaction, chosen.name, spec, relative, absolute=Path(spec).is_absolute())
        return Note(chosen, found.path, found.key)

    return read_in(chosen, read)


def linking(resolver: Resolver, document: DocumentId, wanted: str | None) -> dict[DocumentId, list[LinkRow]]:
    keys = keys_of(resolver.transaction, document)
    found: dict[DocumentId, list[LinkRow]] = {}
    for row in links_into(resolver.transaction, keys, document, wanted):
        how = resolver.of(row)
        if how.via in LEADING and how.documents == (document,):
            found.setdefault(row.document, []).append(row)
    return found


def linked_from(resolver: Resolver, document: DocumentId, wanted: str | None) -> Iterator[tuple[LinkRow, Resolution]]:
    for row in links_out(resolver.transaction, document, wanted):
        how = resolver.of(row)
        if how.via in LEADING and how.documents[0] != document:
            yield row, how


def written(transaction: Reading, target: str, wanted: str | None) -> set[DocumentId]:
    want = name_key(target.lstrip("/"))
    rows = links_keyed(transaction, target_key(target), wanted)
    return {row.document for row in rows if name_key(link_path(row.path, row.kind, row.target)) == want}


def unresolved_links(transaction: Reading) -> list[LinkRow]:
    resolver = Resolver(transaction)
    return [row for row in doubtful(transaction) if resolver.of(row).via is Via.UNRESOLVED]


def note_of(home: Path, spec: str, field: str) -> NamedNote:
    with field_at_fault(field):
        return _note_of(home, spec)


def _note_of(home: Path, spec: str) -> NamedNote:
    try:
        return NamedNote(_by_id(home, spec), ())
    except NotFoundError:
        found = matches(tuple(one for one in known_collections(home) if one.active), spec)
    if not found.found:
        raise unknown_note(spec, found.unsearched)
    (note,) = found.found
    return NamedNote(note, found.unsearched)
