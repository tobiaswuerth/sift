from typing import NamedTuple

from sift.api.collections import CollectionAdded, CollectionStatus, Counted, Status, TitleFrom
from sift.api.errors import SiftError
from sift.interfaces.console.shell import command
from sift.interfaces.console.text import labelled, table
from sift.interfaces.console.units import size

STATUS_HEADER = ("collection", "folder", "documents", "chunks", "size")
NUMBERS = frozenset({2, 3, 4})
NONE = "-"


class Shown(NamedTuple):
    results: tuple[str, ...]
    messages: tuple[str, ...]


def added(result: CollectionAdded, platform: str) -> Shown:
    vault = ", as in an Obsidian vault" if result.vault else ""
    titles = (labelled("title", f"from file names{vault}"),) if result.title_from is TitleFrom.NAME else ()
    return Shown(
        (),
        (
            labelled("added", f"{result.name}  {result.folder}"),
            labelled("store", result.store),
            *titles,
            labelled("next", command(("sift", "index", result.name), platform)),
        ),
    )


def _row(collection: CollectionStatus) -> tuple[str, ...]:
    held = collection.contents
    numbers = (
        (f"{held.documents:,}", f"{held.chunks:,}", size(held.stored_bytes))
        if isinstance(held, Counted)
        else (NONE,) * len(NUMBERS)
    )
    return (collection.name, collection.folder, *numbers)


def _problems(collection: CollectionStatus) -> tuple[str, ...]:
    held = collection.contents
    if isinstance(held, Counted):
        return ()
    fix = (labelled("fix", held.remedy),) if held.remedy else ()
    return (labelled("error", f"{collection.name}: {held.message}"), *fix)


def status(result: Status) -> Shown:
    if not result.collections:
        return Shown((), (labelled("empty", "no collections yet"), labelled("next", "sift add FOLDER")))
    rows = (STATUS_HEADER, *map(_row, result.collections))
    return Shown(table(rows, NUMBERS), tuple(line for each in result.collections for line in _problems(each)))


def failure(error: SiftError | OSError) -> Shown:
    if isinstance(error, OSError):
        return Shown((), (labelled("error", str(error)),))
    fix = (labelled("fix", error.remedy),) if error.remedy else ()
    return Shown((), (labelled("error", error.message), *fix))
