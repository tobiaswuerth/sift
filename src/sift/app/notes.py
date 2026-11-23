from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from sift.api.errors import InvalidRequestError, NotFoundError, StoreUnavailableError
from sift.api.search import Unsearched
from sift.app.documents import place, resolved_at
from sift.app.documents import unknown as documents_unknown
from sift.domain.text.links import LinkKind, link_target, name_key
from sift.domain.text.resolving import LEADING, Candidate, Resolution, Via, link_path, resolved, stem
from sift.storage.catalog import Collection, known_collections
from sift.storage.database import Reading, reading
from sift.storage.documents import stored
from sift.storage.links import LinkRow, doubtful, keys_of, links_into, links_keyed, named, titled
from sift.storage.store import open_store, unavailable_if_unusable

AMBIGUOUS_SHOWN = 5
AMBIGUOUS_REMEDY = "give the note's whole id, as sift get takes it"

type Candidates = tuple[list[Candidate], list[Candidate]]
type Find = Callable[[Reading], int]


@dataclass(slots=True)
class Resolver:
    transaction: Reading
    seen: dict[str, Candidates] = field(default_factory=dict[str, Candidates])

    def _named(self, key: str) -> Candidates:
        if key not in self.seen:
            self.seen[key] = named(self.transaction, key)
        return self.seen[key]

    def of(self, row: LinkRow) -> Resolution:
        return resolved(row.path, row.kind, row.target, *self._named(row.key))

    def of_name(self, target: str) -> Resolution:
        return resolved(None, LinkKind.WIKILINK, target, *self._named(name_key(target.rsplit("/", 1)[-1])))


@dataclass(frozen=True, slots=True)
class Note:
    collection: Collection
    find: Find


@dataclass(frozen=True, slots=True)
class Named:
    note: Note
    unsearched: tuple[Unsearched, ...]


@dataclass(frozen=True, slots=True)
class Match:
    collection: Collection
    path: str


@dataclass(frozen=True, slots=True)
class Matches:
    found: tuple[Match, ...]
    unsearched: tuple[Unsearched, ...]


def read_in[T](one: Collection, read: Callable[[Reading], T]) -> T:
    with (
        unavailable_if_unusable(one.store),
        closing(open_store(one.store, one.id)) as connection,
        reading(connection) as transaction,
    ):
        return read(transaction)


def target_of(id_: str) -> str:
    return link_target(id_) or id_.strip()


def _named(transaction: Reading, name: str) -> list[int]:
    found = Resolver(transaction).of_name(name)
    return list(found.documents) if found.via is not Via.UNRESOLVED else []


def _matches(one: Collection, name: str) -> list[Match]:
    def read(transaction: Reading) -> list[Match]:
        found = _named(transaction, name)
        return [Match(one, path) for path in sorted(each.path for each in titled(transaction, found).values())]

    return read_in(one, read)


def whole(match: Match) -> str:
    return f"{match.collection.name}/{match.path}"


def at_path(match: Match) -> Note:
    def find(transaction: Reading) -> int:
        found = stored(transaction, match.path)
        if found is None:
            raise documents_unknown(whole(match))
        return found.id

    return Note(match.collection, find)


def passed(message: str, unsearched: Sequence[Unsearched]) -> str:
    return "; ".join([message, *(f"not searched: {each.collection}: {each.message}" for each in unsearched)])


def _remedy(remedy: str | None, unsearched: Sequence[Unsearched]) -> str | None:
    found = [each.remedy for each in unsearched if each.remedy is not None]
    return "; ".join(dict.fromkeys([*found, remedy] if remedy is not None else found)) or None


def matches(collections: Sequence[Collection], id_: str) -> Matches:
    name, unread = target_of(id_), list[Unsearched]()
    found: list[Match] = []
    for one in collections:
        try:
            found += _matches(one, name)
        except StoreUnavailableError as error:
            unread.append(Unsearched(one.name, error.message, error.remedy))
    if len(found) > 1:
        listed = ", ".join(sorted(map(whole, found))[:AMBIGUOUS_SHOWN])
        more = ", ..." if len(found) > AMBIGUOUS_SHOWN else ""
        said = passed(f"{id_!r} names several notes: {listed}{more}", unread)
        raise InvalidRequestError(said, _remedy(AMBIGUOUS_REMEDY, unread))
    return Matches(tuple(found), tuple(unread))


def by_name(collections: Sequence[Collection], id_: str, unknown: NotFoundError) -> Named:
    found = matches(collections, id_)
    if not found.found:
        raise NotFoundError(passed(unknown.message, found.unsearched), _remedy(unknown.remedy, found.unsearched))
    (match,) = found.found
    return Named(at_path(match), found.unsearched)


def by_id(home: Path, id_: str) -> Note:
    chosen, relative = place(known_collections(home), id_)

    def find(transaction: Reading) -> int:
        return resolved_at(transaction, id_, relative, absolute=Path(id_).is_absolute()).id

    return Note(chosen, find)


def linking(resolver: Resolver, document: int, wanted: str | None) -> dict[int, list[LinkRow]]:
    keys = keys_of(resolver.transaction, document)
    found: dict[int, list[LinkRow]] = {}
    for row in links_into(resolver.transaction, keys, document, wanted):
        how = resolver.of(row)
        if how.via in LEADING and how.documents == (document,):
            found.setdefault(row.document, []).append(row)
    return found


def written(transaction: Reading, target: str, wanted: str | None) -> set[int]:
    want = stem(target.lstrip("/"))
    rows = links_keyed(transaction, name_key(target.rsplit("/", 1)[-1]), wanted)
    return {row.document for row in rows if stem(link_path(row.path, row.kind, row.target)) == want}


def unresolved_links(transaction: Reading) -> list[LinkRow]:
    resolver = Resolver(transaction)
    return [row for row in doubtful(transaction) if resolver.of(row).via is Via.UNRESOLVED]


def note_of(home: Path, collections: Sequence[Collection], id_: str) -> Named:
    try:
        note = by_id(home, id_)
        read_in(note.collection, note.find)
    except NotFoundError as unknown:
        return by_name(collections, id_, unknown)
    return Named(note, ())
