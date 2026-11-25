import os
from collections.abc import Sequence
from pathlib import Path, PurePath

from sift.api.documents import MOST_LINE, Documents, DocumentText, GetDocuments, Unfound, WantedDocument
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, SiftError
from sift.domain.query import fields as shown
from sift.domain.text.kinds import public_id, split_keyed_path
from sift.domain.text.lines import refuse_past_end
from sift.storage.catalog import Collection, known_collections, refuse_inactive
from sift.storage.database import Reading
from sift.storage.documents import Stored, fields_of, paths_named, stored
from sift.storage.store import read_store
from sift.storage.texts import LinesRead, lines_between, text_of

MOST_LISTED = 5
SEPARATOR = "/"


def unknown(id_: str) -> NotFoundError:
    return NotFoundError(f"unknown document {id_!r}", "take the id from a search result", code=Problem.UNKNOWN_DOCUMENT)


def holding(known: Sequence[Collection], full: Path) -> tuple[Collection, str] | None:
    nearest = sorted(known, key=lambda chosen: -len(chosen.folder.parts))

    def held(path: Path) -> tuple[Collection, str] | None:
        return next(
            (
                (chosen, path.relative_to(chosen.folder).as_posix())
                for chosen in nearest
                if path.is_relative_to(chosen.folder) and path != chosen.folder
            ),
            None,
        )

    anchored = any(PurePath(chosen.folder.anchor) == PurePath(full.anchor) for chosen in known)
    return held(full) or (held(full.parent.resolve() / full.name) if anchored else None)


def located(known: Sequence[Collection], id_: str) -> tuple[Collection, str] | None:
    if Path(id_).is_absolute():
        found = holding(known, Path(os.path.normpath(id_)))
    else:
        name, _, relative = id_.partition(SEPARATOR)
        chosen = next((one for one in known if one.name.casefold() == name.casefold()), None)
        found = (chosen, relative) if chosen else None
    return found if found is not None and found[1] else None


def place(known: Sequence[Collection], id_: str) -> tuple[Collection, str]:
    found = located(known, id_)
    if found is None:
        raise unknown(id_)
    return refuse_inactive(found[0]), found[1]


def resolved_at(transaction: Reading, id_: str, relative: str, *, absolute: bool) -> Stored:
    exact = stored(transaction, relative) or stored(transaction, *split_keyed_path(relative))
    if exact is not None:
        return exact
    candidates = paths_named(transaction, relative.rsplit(SEPARATOR, 1)[-1])
    if absolute:
        matches = [path for path in candidates if path.casefold() == relative.casefold()]
    else:
        matches = [path for path in candidates if path.endswith(SEPARATOR + relative)]
    if len(matches) > 1 and not absolute:
        listed = ", ".join(matches[:MOST_LISTED]) + (", ..." if len(matches) > MOST_LISTED else "")
        raise InvalidRequestError(f"{id_!r} matches {listed}", "give the whole id")
    found = stored(transaction, matches[0]) if len(matches) == 1 else None
    if found is None:
        raise unknown(id_)
    return found


def _lines(wanted: WantedDocument) -> tuple[int, int] | None:
    if wanted.first_line is None and wanted.last_line is None:
        return None
    first, last = wanted.first_line or 1, wanted.last_line or MOST_LINE
    if last < first:
        said = f"{wanted.id}: line {last} comes before line {first}"
        raise InvalidRequestError(said, "give the first line before the last", "documents")
    return first, last


def _document(known: Sequence[Collection], wanted: WantedDocument, fields: tuple[str, ...]) -> DocumentText:
    span = _lines(wanted)
    chosen, relative = place(known, wanted.id)
    with read_store(chosen.store, chosen.id) as transaction:
        found = resolved_at(transaction, wanted.id, relative, absolute=Path(wanted.id).is_absolute())
        read = lines_between(transaction, found.id, *span) if span else LinesRead(text_of(transaction, found.id), 0)
    whole = public_id(chosen.name, found.path, found.key)
    if span:
        refuse_past_end(whole, read.total, first=span[0])
    picked = shown.picked(fields_of(found.fields), fields) if fields else None
    return DocumentText(whole, found.title, read.text, picked)


def documents(home: Path, request: GetDocuments) -> Documents:
    shown.checked(request.fields, None)
    for wanted in request.documents:
        _lines(wanted)
    known = known_collections(home)
    found: list[DocumentText] = []
    unfound: list[Unfound] = []
    for wanted in request.documents:
        try:
            found.append(_document(known, wanted, request.fields))
        except SiftError as error:
            unfound.append(Unfound(wanted.id, error.message, error.remedy, error.code))
    return Documents(tuple(found), tuple(unfound))
