from typing import NamedTuple

from sift.api.collections import CollectionAdded, CollectionStatus, Counted, Status, TitleFrom
from sift.api.documents import Documents
from sift.api.errors import SiftError
from sift.api.indexing import IndexResult
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, table
from sift.interfaces.console.units import size

STATUS_HEADER = ("collection", "folder", "documents", "chunks", "size")
UNREADABLE_HEADER = ("collection", "file", "problem")
NONE = "-"


class Shown(NamedTuple):
    results: tuple[str, ...]
    messages: tuple[str, ...]
    failed: bool = False


def _fix(remedy: str | None, style: Style) -> tuple[str, ...]:
    return (labelled("fix", remedy, Tone.SECONDARY, style),) if remedy else ()


def _next(words: tuple[str, ...], platform: str, style: Style) -> str:
    return labelled("next", command(words, platform), Tone.SECONDARY, style)


def _empty(platform: str, style: Style) -> Shown:
    return Shown((), (labelled("empty", "no collections yet"), _next(("sift", "add", "FOLDER"), platform, style)))


def added(result: CollectionAdded, platform: str, style: Style) -> Shown:
    vault = ", as in an Obsidian vault" if result.vault else ""
    titles = (labelled("title", f"from file names{vault}"),) if result.title_from is TitleFrom.NAME else ()
    links = (labelled("links", "followed out of the folder"),) if result.follow_links else ()
    return Shown(
        (),
        (
            labelled("added", f"{result.name}  {result.folder}", Tone.DONE, style),
            labelled("store", result.store),
            *titles,
            *links,
            _next(("sift", "index", result.name), platform, style),
        ),
    )


def _row(collection: CollectionStatus, *, unreadable: bool) -> tuple[str, ...]:
    held = collection.contents
    numbers = (
        (f"{held.documents:,}", f"{held.chunks:,}", size(held.stored_bytes), *((f"{held.unreadable:,}",) * unreadable))
        if isinstance(held, Counted)
        else (NONE,) * (len(STATUS_HEADER) - 2 + unreadable)
    )
    return (collection.name, collection.folder, *numbers)


def _problems(collection: CollectionStatus, style: Style) -> tuple[str, ...]:
    held = collection.contents
    if isinstance(held, Counted):
        return ()
    return (labelled("error", f"{collection.name}: {held.message}", Tone.ERROR, style), *_fix(held.remedy, style))


def _unreadable_files(result: Status) -> tuple[str, ...]:
    rows = [
        (collection.name, entry.path, entry.problem)
        for collection in result.collections
        if isinstance(collection.contents, Counted)
        for entry in collection.contents.unreadable_files
    ]
    return ("", *table((UNREADABLE_HEADER, *rows), frozenset())) if rows else ()


def status(result: Status, platform: str, style: Style) -> Shown:
    if not result.collections:
        return _empty(platform, style)
    unreadable = any(isinstance(each.contents, Counted) and each.contents.unreadable for each in result.collections)
    header = (*STATUS_HEADER, *(("unreadable",) * unreadable))
    numbers = frozenset(range(2, len(header)))
    rows = (header, *(_row(each, unreadable=unreadable) for each in result.collections))
    messages = tuple(line for each in result.collections for line in _problems(each, style))
    return Shown((*table(rows, numbers), *_unreadable_files(result)), messages)


def indexed(result: IndexResult, platform: str, style: Style) -> Shown:
    return _empty(platform, style) if not result.collections else Shown((), ())


def _text(text: str) -> str:
    return text.removesuffix("\n")


def documents(result: Documents, _platform: str, style: Style) -> Shown:
    if len(result.found) == 1 and not result.unfound:
        return Shown((_text(result.found[0].text),), ())
    shown = [line for found in result.found for line in (f"==> {found.id} <==", _text(found.text), "")]
    messages = [
        line
        for unfound in result.unfound
        for line in (
            labelled("error", f"{unfound.id}: {unfound.message}", Tone.ERROR, style),
            *_fix(unfound.remedy, style),
        )
    ]
    return Shown(tuple(shown[:-1]), tuple(messages), failed=bool(result.unfound))


def failure(error: SiftError | OSError, style: Style) -> Shown:
    if isinstance(error, OSError):
        return Shown((), (labelled("error", str(error), Tone.ERROR, style),), failed=True)
    return Shown((), (labelled("error", error.message, Tone.ERROR, style), *_fix(error.remedy, style)), failed=True)
