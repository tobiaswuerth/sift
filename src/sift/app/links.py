from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from sift.api.errors import StoreUnavailableError
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
from sift.api.validated import validated
from sift.app.collections import chosen_or_active
from sift.app.notes import AMBIGUOUS_SHOWN, Note, Resolver, linking, note_of, read_in, unresolved_links
from sift.domain.text.kinds import public_id
from sift.domain.text.links import name_key
from sift.domain.text.resolving import LEADING, Linking, Resolution, link_path, linking_lines
from sift.storage.database import Reading
from sift.storage.documents import texts
from sift.storage.links import LinkRow, Titled, links_out, titled

LINKING_SHOWN = 5
FURTHER = 2


@dataclass(frozen=True, slots=True)
class Out:
    row: LinkRow
    how: Resolution
    through: DocumentId | None


def _out(resolver: Resolver, document: DocumentId, request: ShowLinks) -> list[Out]:
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
    document: DocumentId
    rows: list[LinkRow]
    through: DocumentId | None


def _in(resolver: Resolver, document: DocumentId, request: ShowLinks) -> list[In]:
    first = linking(resolver, document, request.field)
    found = [In(source, rows, None) for source, rows in first.items()]
    if request.depth == FURTHER:
        seen = {document, *first}
        for step in first:
            for source, rows in linking(resolver, step, request.field).items():
                if source not in seen:
                    seen.add(source)
                    found.append(In(source, rows, step))
    return found[: request.k]


def _id(collection: str, row: Titled) -> str:
    return public_id(collection, row.path, row.key)


def _shown_out(each: Out, names: dict[int, Titled], collection: str) -> LinkOut:
    row, how = each.row, each.how
    leads = how.via in LEADING
    first = names[how.documents[0]] if leads else None
    ambiguous = tuple(_id(collection, names[at]) for at in how.documents[:AMBIGUOUS_SHOWN]) if not leads else ()
    return LinkOut(
        row.line,
        row.target,
        row.field,
        row.fragment,
        str(how.via),
        None if first is None else _id(collection, first),
        None if first is None else first.title,
        ambiguous,
        None if each.through is None else _id(collection, names[each.through]),
    )


def _shown_in(each: In, names: dict[int, Titled], text: str, collection: str) -> LinkIn:
    lines = linking_lines([Linking(row.line, row.field, row.target) for row in each.rows], text)
    return LinkIn(
        _id(collection, names[each.document]),
        names[each.document].title,
        tuple(LinkingLine(*line) for line in lines),
        None if each.through is None else _id(collection, names[each.through]),
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
        _id(collection, names[document]),
        names[document].title,
        None if out is None else tuple(_shown_out(each, names, collection) for each in out),
        None
        if into is None
        else tuple(_shown_in(each, names, whole[DocumentId(each.document)], collection) for each in into),
    )


def links(home: Path, request: ShowLinks) -> Links:
    validated(request)
    named = note_of(home, request.id)
    shown = read_in(named.note.collection, lambda transaction: _links(transaction, named.note, request))
    return replace(shown, unsearched=named.unsearched)


@dataclass(frozen=True, slots=True)
class Gone:
    written: Counter[str]
    documents: dict[int, None]


def _unresolved(collection: str, transaction: Reading) -> list[tuple[str, list[str]]]:
    targets: dict[str, Gone] = {}
    for row in unresolved_links(transaction):
        gone = targets.setdefault(name_key(link_path(row.path, row.kind, row.target)), Gone(Counter(), {}))
        gone.written[row.target] += 1
        gone.documents[row.document] = None
    names = titled(transaction, [document for gone in targets.values() for document in gone.documents])
    return [
        (
            min(gone.written.items(), key=lambda each: (-each[1], each[0]))[0],
            sorted(_id(collection, names[document]) for document in gone.documents),
        )
        for gone in targets.values()
    ]


def unresolved(home: Path, request: ListUnresolved) -> Unresolved:
    validated(request)
    found: list[tuple[str, list[str]]] = []
    passed: list[Unsearched] = []
    for one in chosen_or_active(home, request.collections):
        try:
            found += read_in(one, partial(_unresolved, one.name))
        except StoreUnavailableError as error:
            passed.append(Unsearched(one.name, error.message, error.remedy))
    ranked = sorted(found, key=lambda each: (-len(each[1]), each[0]))[: request.k]
    targets = tuple(Target(target, len(ids), tuple(sorted(ids)[:LINKING_SHOWN])) for target, ids in ranked)
    return Unresolved(targets, tuple(passed))
