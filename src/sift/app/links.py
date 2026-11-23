from collections import Counter
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from sift.api.errors import InvalidRequestError, NotFoundError, StoreUnavailableError
from sift.api.ids import DocumentId
from sift.api.links import (
    Direction,
    LinkIn,
    LinkingLine,
    LinkOut,
    Links,
    ListUnresolved,
    ShowLinks,
    Target,
    Unresolved,
)
from sift.api.search import Unsearched
from sift.app.collections import chosen_or_active
from sift.app.documents import place, resolved_at
from sift.app.searching import checked_k
from sift.domain.text.links import LinkKind, in_line, name_key
from sift.domain.text.resolving import (
    LEADING,
    Candidate,
    Linking,
    Resolution,
    Via,
    link_path,
    linking_lines,
    resolved,
    stem,
)
from sift.storage.catalog import Collection, known_collections
from sift.storage.database import Reading, reading
from sift.storage.documents import texts
from sift.storage.links import LinkRow, Titled, doubtful, keys_of, links_into, links_out, named, titled
from sift.storage.store import open_store, unavailable_if_damaged

AMBIGUOUS_SHOWN = 5
LINKING_SHOWN = 5
DEPTHS = (1, 2)
FURTHER = 2
DEPTH_REMEDY = "follow links one step, or two: --depth 1 or --depth 2"
AMBIGUOUS_REMEDY = "give the note's whole id, as sift get takes it"

type Candidates = tuple[list[Candidate], list[Candidate]]


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


type Find = Callable[[Reading], int]


@dataclass(frozen=True, slots=True)
class Note:
    collection: Collection
    find: Find


def _reading[T](one: Collection, read: Callable[[Reading], T]) -> T:
    with (
        unavailable_if_damaged(one.store),
        closing(open_store(one.store, one.id)) as connection,
        reading(connection) as transaction,
    ):
        return read(transaction)


def target_of(id_: str) -> str:
    written = id_.strip()
    found = in_line(written, 1)
    whole = len(found) == 1 and written.startswith(("[[", "![[")) and written.endswith("]]")
    return found[0].target if whole else written


def _named(transaction: Reading, name: str) -> list[int]:
    found = Resolver(transaction).of_name(name)
    return list(found.documents) if found.via is not Via.UNRESOLVED else []


def _matches(one: Collection, name: str) -> list[str]:
    def read(transaction: Reading) -> list[str]:
        found = _named(transaction, name)
        return sorted(f"{one.name}/{each.path}" for each in titled(transaction, found).values())

    return _reading(one, read)


def _at(whole: str) -> Find:
    path = whole.split("/", 1)[1]

    def find(transaction: Reading) -> int:
        return resolved_at(transaction, whole, path, absolute=False).id

    return find


def _by_name(home: Path, id_: str, unknown: NotFoundError) -> Note:
    name, passed = target_of(id_), list[str]()
    found: list[tuple[Collection, str]] = []
    for one in chosen_or_active(home, ()):
        try:
            found += [(one, whole) for whole in _matches(one, name)]
        except StoreUnavailableError:
            passed.append(one.name)
    if not found:
        said = f"{unknown.message}; passed over: {', '.join(passed)}" if passed else unknown.message
        raise NotFoundError(said, unknown.remedy)
    if len(found) > 1:
        listed = ", ".join(sorted(whole for _, whole in found)[:AMBIGUOUS_SHOWN])
        more = ", ..." if len(found) > AMBIGUOUS_SHOWN else ""
        raise InvalidRequestError(f"{id_!r} names several notes: {listed}{more}", AMBIGUOUS_REMEDY)
    return Note(found[0][0], _at(found[0][1]))


def _by_id(home: Path, id_: str) -> Note:
    chosen, relative = place(known_collections(home), id_)

    def find(transaction: Reading) -> int:
        return resolved_at(transaction, id_, relative, absolute=Path(id_).is_absolute()).id

    return Note(chosen, find)


@dataclass(frozen=True, slots=True)
class Out:
    row: LinkRow
    how: Resolution
    through: int | None


def _out(resolver: Resolver, document: int, request: ShowLinks) -> list[Out]:
    first = [Out(row, resolver.of(row), None) for row in links_out(resolver.transaction, document, request.field)]
    found = list(first)
    if request.depth == FURTHER:
        reached = [each.how.documents[0] for each in first if each.how.via in LEADING]
        seen = {document, *reached}
        for step in dict.fromkeys(reached):
            for row in links_out(resolver.transaction, step, request.field):
                how = resolver.of(row)
                if how.via in LEADING and how.documents[0] not in seen:
                    seen.add(how.documents[0])
                    found.append(Out(row, how, step))
    return found[: request.k]


@dataclass(frozen=True, slots=True)
class In:
    document: int
    rows: list[LinkRow]
    through: int | None


def _linking(resolver: Resolver, document: int, wanted: str | None) -> dict[int, list[LinkRow]]:
    keys = keys_of(resolver.transaction, document)
    found: dict[int, list[LinkRow]] = {}
    for row in links_into(resolver.transaction, keys, document, wanted):
        how = resolver.of(row)
        if how.via in LEADING and how.documents == (document,):
            found.setdefault(row.document, []).append(row)
    return found


def _in(resolver: Resolver, document: int, request: ShowLinks) -> list[In]:
    first = _linking(resolver, document, request.field)
    found = [In(source, rows, None) for source, rows in first.items()]
    if request.depth == FURTHER:
        seen = {document, *first}
        for step in first:
            for source, rows in _linking(resolver, step, request.field).items():
                if source not in seen:
                    seen.add(source)
                    found.append(In(source, rows, step))
    return found[: request.k]


def _shown_out(each: Out, names: dict[int, Titled], collection: str) -> LinkOut:
    row, how = each.row, each.how
    leads = how.via in LEADING
    first = names[how.documents[0]] if leads else None
    ambiguous = tuple(f"{collection}/{names[at].path}" for at in how.documents[:AMBIGUOUS_SHOWN]) if not leads else ()
    return LinkOut(
        row.line,
        row.target,
        row.field,
        row.fragment,
        str(how.via),
        None if first is None else f"{collection}/{first.path}",
        None if first is None else first.title,
        ambiguous,
        None if each.through is None else f"{collection}/{names[each.through].path}",
    )


def _shown_in(each: In, names: dict[int, Titled], text: str, collection: str) -> LinkIn:
    lines = linking_lines([Linking(row.line, row.field, row.target) for row in each.rows], text)
    return LinkIn(
        f"{collection}/{names[each.document].path}",
        names[each.document].title,
        tuple(LinkingLine(*line) for line in lines),
        None if each.through is None else f"{collection}/{names[each.through].path}",
    )


def _links(transaction: Reading, found: Note, request: ShowLinks) -> Links:
    resolver, collection, document = Resolver(transaction), found.collection.name, found.find(transaction)
    out = _out(resolver, document, request) if request.direction is not Direction.IN else None
    into = _in(resolver, document, request) if request.direction is not Direction.OUT else None
    documents = [document]
    documents += [at for each in out or () for at in (*each.how.documents, each.through) if at is not None]
    documents += [at for each in into or () for at in (each.document, each.through) if at is not None]
    names = titled(transaction, documents)
    whole = texts(transaction, [DocumentId(each.document) for each in into or ()])
    return Links(
        f"{collection}/{names[document].path}",
        names[document].title,
        None if out is None else tuple(_shown_out(each, names, collection) for each in out),
        None
        if into is None
        else tuple(_shown_in(each, names, whole[DocumentId(each.document)], collection) for each in into),
    )


def links(home: Path, request: ShowLinks) -> Links:
    checked_k(request.k)
    if request.depth not in DEPTHS:
        raise InvalidRequestError(f"--depth {request.depth} is neither 1 nor 2", DEPTH_REMEDY)
    try:
        found = _by_id(home, request.id)
        return _reading(found.collection, lambda transaction: _links(transaction, found, request))
    except NotFoundError as unknown:
        named = _by_name(home, request.id, unknown)
        return _reading(named.collection, lambda transaction: _links(transaction, named, request))


@dataclass(frozen=True, slots=True)
class Gone:
    written: Counter[str]
    documents: dict[int, None]


def _unresolved(transaction: Reading) -> list[tuple[str, list[str]]]:
    resolver = Resolver(transaction)
    targets: dict[str, Gone] = {}
    for row in doubtful(transaction):
        if resolver.of(row).via is Via.UNRESOLVED:
            gone = targets.setdefault(stem(link_path(row.path, row.kind, row.target)), Gone(Counter(), {}))
            gone.written[row.target] += 1
            gone.documents[row.document] = None
    names = titled(transaction, [document for gone in targets.values() for document in gone.documents])
    return [
        (
            min(gone.written.items(), key=lambda each: (-each[1], each[0]))[0],
            sorted(names[document].path for document in gone.documents),
        )
        for gone in targets.values()
    ]


def unresolved(home: Path, request: ListUnresolved) -> Unresolved:
    checked_k(request.k)
    found: list[tuple[str, list[str]]] = []
    passed: list[Unsearched] = []
    for one in chosen_or_active(home, () if request.collection is None else (request.collection,)):
        try:
            found += [
                (target, [f"{one.name}/{path}" for path in paths]) for target, paths in _reading(one, _unresolved)
            ]
        except StoreUnavailableError as error:
            passed.append(Unsearched(one.name, error.message, error.remedy))
    ranked = sorted(found, key=lambda each: (-len(each[1]), each[0]))[: request.k]
    targets = tuple(Target(target, len(ids), tuple(sorted(ids)[:LINKING_SHOWN])) for target, ids in ranked)
    return Unresolved(targets, tuple(passed))
