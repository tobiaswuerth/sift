from dataclasses import dataclass

from sift.api.collections import CollectionAdded, CollectionStatus, Counted, Status, TitleFrom, Unavailable
from sift.interfaces.cli.shown.outcome import Shown, empty, fix, next_step
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, table
from sift.interfaces.console.units import size

UNREADABLE_HEADER = ("collection", "file", "problem")
NONE = "-"


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
            next_step(("sift", "index", result.name), platform, style),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Columns:
    embedded: bool
    linked: bool
    unreadable: bool
    resting: bool


def _columns(result: Status) -> _Columns:
    held = [each.contents for each in result.collections if isinstance(each.contents, Counted)]
    return _Columns(
        embedded=any(each.embedded for each in held),
        linked=any(each.links for each in held),
        unreadable=any(each.unreadable for each in held),
        resting=not all(each.active for each in result.collections),
    )


def _header(columns: _Columns) -> tuple[str, ...]:
    return (
        "collection",
        "folder",
        "documents",
        "chunks",
        *(("embedded",) * columns.embedded),
        "size",
        *(("links", "unresolved") if columns.linked else ()),
        *(("unreadable",) * columns.unreadable),
        *(("active",) * columns.resting),
    )


def _counts(held: Counted, columns: _Columns) -> tuple[str, ...]:
    return (
        f"{held.documents:,}",
        f"{held.chunks:,}",
        *((f"{held.embedded:,}",) * columns.embedded),
        size(held.stored_bytes),
        *((f"{held.links:,}", f"{held.unresolved:,}") if columns.linked else ()),
        *((f"{held.unreadable:,}",) * columns.unreadable),
    )


def _row(collection: CollectionStatus, columns: _Columns) -> tuple[str, ...]:
    active = (("yes" if collection.active else "no"),) * columns.resting
    match collection.contents:
        case Counted() as held:
            counted = _counts(held, columns)
        case _:
            counted = (NONE,) * (len(_header(columns)) - 2 - len(active))
    return (collection.name, collection.folder, *counted, *active)


def _models(result: Status) -> tuple[str, ...]:
    counted = [each.contents for each in result.collections if isinstance(each.contents, Counted)]
    held = sorted({each.model for each in counted if each.model is not None})
    return ("", labelled("model", ", ".join(held))) if held else ()


def _unavailable(name: str, held: Unavailable, style: Style) -> tuple[str, ...]:
    return (labelled("error", f"{name}: {held.message}", Tone.ERROR, style), *fix(held.remedy, style))


def _problems(collection: CollectionStatus, style: Style) -> tuple[str, ...]:
    match collection.contents:
        case Counted():
            return ()
        case _ as held:
            return _unavailable(collection.name, held, style)


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
        return empty(platform, style)
    columns = _columns(result)
    header = _header(columns)
    rows = (header, *(_row(each, columns) for each in result.collections))
    messages = tuple(line for each in result.collections for line in _problems(each, style))
    shown = (*table(rows, frozenset(range(2, len(header)))), *_models(result), *_unreadable_files(result))
    return Shown(shown, messages)
