import os
import unicodedata
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePath

from sift.api.documents import MOST_LINE, Documents, DocumentText, GetDocuments, WantedDocument
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, SiftError
from sift.api.ids import CollectionId
from sift.api.notices import Notice, notice_of
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
BACKSLASH = "\\"
PARENT = ".."
UNKNOWN_REMEDY = "take the id from a search result"
SEPARATED_REMEDY = "write an id's folders with / between them, as search results give it"


def _written_oddly(id_: str) -> bool:
    folders = id_.split(SEPARATOR)
    return not Path(id_).is_absolute() and (BACKSLASH in id_ or not folders[-1] or PARENT in folders)


def unknown(id_: str, remedy: str = UNKNOWN_REMEDY) -> NotFoundError:
    told = SEPARATED_REMEDY if _written_oddly(id_) else remedy
    return NotFoundError(f"unknown document {id_!r}", told, code=Problem.UNKNOWN_DOCUMENT)


def _unknown_in(name: str, id_: str, relative: str, candidates: Sequence[str]) -> NotFoundError:
    folded = relative.casefold()
    alike = [path for path in candidates if path.casefold() == folded or path.casefold().endswith(SEPARATOR + folded)]
    listed = ", ".join(public_id(name, path, "") for path in alike[:MOST_LISTED])
    return unknown(id_, f"ids keep their case: give {listed}" if alike else UNKNOWN_REMEDY)


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


def _composed(path: str) -> str:
    return unicodedata.normalize("NFC", path)


def resolved_at(transaction: Reading, name: str, id_: str, relative: str, *, absolute: bool) -> Stored:
    exact = stored(transaction, relative) or stored(transaction, *split_keyed_path(relative))
    if exact is not None:
        return exact
    candidates = paths_named(transaction, relative.rsplit(SEPARATOR, 1)[-1])
    composed = _composed(relative)
    if absolute:
        matches = [path for path in candidates if _composed(path).casefold() == composed.casefold()]
    else:
        matches = [
            path for path in candidates if _composed(path) == composed or _composed(path).endswith(SEPARATOR + composed)
        ]
    if len(matches) > 1 and not absolute:
        shown = [public_id(name, path, "") for path in matches[:MOST_LISTED]]
        listed = ", ".join(shown) + (", ..." if len(matches) > MOST_LISTED else "")
        raise InvalidRequestError(f"{id_!r} matches {listed}", "give the whole id", code=Problem.AMBIGUOUS)
    found = stored(transaction, matches[0]) if len(matches) == 1 else None
    if found is None:
        raise _unknown_in(name, id_, relative, candidates)
    return found


def _lines(wanted: WantedDocument, at: int) -> tuple[int, int] | None:
    if wanted.first_line is None and wanted.last_line is None:
        return None
    first, last = wanted.first_line or 1, wanted.last_line or MOST_LINE
    if last < first:
        said = f"{wanted.id}: line {last} comes before line {first}"
        raise InvalidRequestError(said, "give the first line before the last", f"documents[{at}].last_line")
    return first, last


type Read = DocumentText | Notice
type StoreReading = Callable[[Path, CollectionId], AbstractContextManager[Reading]]


@dataclass(frozen=True, slots=True)
class _Placed:
    at: int
    wanted: WantedDocument
    relative: str


def _unread(wanted: WantedDocument, error: SiftError) -> Notice:
    return notice_of(error, id_=wanted.id)


def _document(transaction: Reading, chosen: Collection, placed: _Placed, fields: tuple[str, ...]) -> DocumentText:
    wanted, span = placed.wanted, _lines(placed.wanted, placed.at)
    found = resolved_at(transaction, chosen.name, wanted.id, placed.relative, absolute=Path(wanted.id).is_absolute())
    read = lines_between(transaction, found.id, *span) if span else LinesRead(text_of(transaction, found.id), 0)
    whole = public_id(chosen.name, found.path, found.key)
    if span:
        refuse_past_end(whole, read.total, first=span[0])
    picked = shown.picked(fields_of(found.fields), fields) if fields else None
    return DocumentText(whole, found.title, read.text, picked)


def _answered(transaction: Reading, chosen: Collection, placed: _Placed, fields: tuple[str, ...]) -> Read:
    try:
        return _document(transaction, chosen, placed, fields)
    except SiftError as error:
        return _unread(placed.wanted, error)


def _read_from(
    chosen: Collection, asked: list[_Placed], fields: tuple[str, ...], reading: StoreReading
) -> dict[int, Read]:
    answers: dict[int, Read] = {}
    try:
        with reading(chosen.store, chosen.id) as transaction:
            for placed in asked:
                answers[placed.at] = _answered(transaction, chosen, placed, fields)
    except SiftError as error:
        return {placed.at: answers.get(placed.at) or _unread(placed.wanted, error) for placed in asked}
    return answers


def documents(home: Path, request: GetDocuments, reading: StoreReading = read_store) -> Documents:
    for at, wanted in enumerate(request.documents):
        _lines(wanted, at)
    known = known_collections(home)
    answers: dict[int, Read] = {}
    held: dict[CollectionId, tuple[Collection, list[_Placed]]] = {}
    for at, wanted in enumerate(request.documents):
        try:
            chosen, relative = place(known, wanted.id)
        except SiftError as error:
            answers[at] = _unread(wanted, error)
            continue
        held.setdefault(chosen.id, (chosen, []))[1].append(_Placed(at, wanted, relative))
    for chosen, asked in held.values():
        answers |= _read_from(chosen, asked, request.fields, reading)
    ordered = [answers[at] for at in range(len(request.documents))]
    found = tuple(each for each in ordered if isinstance(each, DocumentText))
    return Documents(found, tuple(each for each in ordered if isinstance(each, Notice)))
