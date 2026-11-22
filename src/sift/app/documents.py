import os
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from sift.api.documents import Documents, DocumentText, GetDocuments, Unfound
from sift.api.errors import InvalidRequestError, NotFoundError, SiftError
from sift.domain.text.lines import line_range, lines_of
from sift.storage.catalog import CATALOG, Collection, collections, open_catalog
from sift.storage.database import Reading, reading
from sift.storage.documents import Stored, paths_named, stored
from sift.storage.store import open_store

MOST_LISTED = 5
SEPARATOR = "/"


def _unknown(id_: str) -> NotFoundError:
    return NotFoundError(f"unknown document {id_!r}", "take the id from a search result, or `sift status`")


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

    return held(full) or held(full.parent.resolve() / full.name)


def _place(known: Sequence[Collection], id_: str) -> tuple[Collection, str]:
    if Path(id_).is_absolute():
        found = holding(known, Path(os.path.normpath(id_)))
    else:
        name, _, relative = id_.partition(SEPARATOR)
        chosen = next((one for one in known if one.name.casefold() == name.casefold()), None)
        found = (chosen, relative) if chosen else None
    if found is None or not found[1]:
        raise _unknown(id_)
    return found


def _resolved(transaction: Reading, id_: str, relative: str, *, absolute: bool) -> Stored:
    exact = stored(transaction, relative)
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
        raise _unknown(id_)
    return found


def _document(known: Sequence[Collection], id_: str, span: tuple[int, int] | None) -> DocumentText:
    chosen, relative = _place(known, id_)
    with closing(open_store(chosen.store, chosen.id)) as connection, reading(connection) as transaction:
        found = _resolved(transaction, id_, relative, absolute=Path(id_).is_absolute())
    whole = f"{chosen.name}/{found.path}"
    return DocumentText(whole, found.title, lines_of(found.text, *span, whole) if span else found.text)


def _known(home: Path) -> tuple[Collection, ...]:
    if not (home / CATALOG).is_file():
        return ()
    with closing(open_catalog(home)) as catalog, reading(catalog) as transaction:
        return collections(transaction)


def documents(home: Path, request: GetDocuments) -> Documents:
    span = line_range(request.lines) if request.lines is not None else None
    known = _known(home)
    found: list[DocumentText] = []
    unfound: list[Unfound] = []
    for id_ in request.ids:
        try:
            found.append(_document(known, id_, span))
        except SiftError as error:
            unfound.append(Unfound(id_, error.message, error.remedy))
    return Documents(tuple(found), tuple(unfound))
