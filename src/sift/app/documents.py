import os
from collections.abc import Sequence
from pathlib import Path, PurePath

from sift.api.documents import Documents, DocumentText, GetDocuments, Unfound
from sift.api.errors import InvalidRequestError, NotFoundError, SiftError
from sift.domain.query import fields as shown
from sift.domain.text.kinds import public_id, split_keyed_path
from sift.domain.text.lines import line_range, lines_of
from sift.storage.catalog import Collection, known_collections
from sift.storage.database import Reading
from sift.storage.documents import Stored, fields_of, paths_named, stored
from sift.storage.store import read_store

MOST_LISTED = 5
SEPARATOR = "/"


def unknown(id_: str) -> NotFoundError:
    return NotFoundError(f"unknown document {id_!r}", "take the id from a search result")


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


def place(known: Sequence[Collection], id_: str) -> tuple[Collection, str]:
    if Path(id_).is_absolute():
        found = holding(known, Path(os.path.normpath(id_)))
    else:
        name, _, relative = id_.partition(SEPARATOR)
        chosen = next((one for one in known if one.name.casefold() == name.casefold()), None)
        found = (chosen, relative) if chosen else None
    if found is None or not found[1]:
        raise unknown(id_)
    return found


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


def _document(known: Sequence[Collection], id_: str, request: GetDocuments) -> DocumentText:
    span = line_range(request.lines) if request.lines is not None else None
    chosen, relative = place(known, id_)
    with read_store(chosen.store, chosen.id) as transaction:
        found = resolved_at(transaction, id_, relative, absolute=Path(id_).is_absolute())
    whole = public_id(chosen.name, found.path, found.key)
    text = lines_of(found.text, *span, whole) if span else found.text
    fields = shown.picked(fields_of(found.fields), request.fields) if request.fields else None
    return DocumentText(whole, found.title, text, fields)


def documents(home: Path, request: GetDocuments) -> Documents:
    shown.checked(request.fields, None)
    if request.lines is not None:
        line_range(request.lines)
    known = known_collections(home)
    found: list[DocumentText] = []
    unfound: list[Unfound] = []
    for id_ in request.ids:
        try:
            found.append(_document(known, id_, request))
        except SiftError as error:
            unfound.append(Unfound(id_, error.message, error.remedy))
    return Documents(tuple(found), tuple(unfound))
