from dataclasses import dataclass
from types import MappingProxyType

from sift.api.collections import (
    CollectionAdded,
    CollectionChanged,
    CollectionRemoved,
    CollectionsSwitched,
    CollectionStatus,
    Counted,
    Status,
    StoresMoved,
    TitleFrom,
    Unavailable,
)
from sift.domain.text.wording import counted
from sift.interfaces.cli.shown.outcome import Shown, empty, fix, next_step
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, table
from sift.interfaces.console.units import size

UNREADABLE_HEADER = ("collection", "file", "problem")
NONE = "-"
LEFT_REMEDY = "the next sift index deletes it, once the program using it has stopped"
COPIED = ", copied: it lay on another drive"
TITLES = MappingProxyType({TitleFrom.HEADING: "headings", TitleFrom.NAME: "file names"})
SWITCHED = MappingProxyType(
    {True: "searched again", False: "left out of searches, and of index and embed unless named"}
)


def _changed(result: CollectionChanged, platform: str, style: Style) -> Shown:
    titled = f"from {TITLES[result.title_from]}: {counted(result.retitled, 'title')} changed"
    linked = "read" if result.follow_links else "left out"
    lines = (
        labelled("title", f"{result.name}: {titled}", Tone.DONE, style),
        labelled("links", f"{result.name}: files linked from outside the folder {linked}"),
        labelled("agent", f"{result.name}: {'seen by agents' if result.agents else 'hidden from agents'}"),
    )
    embed = (
        (labelled("embed", f"{counted(result.to_embed, 'passage')} to embed again, which carry their title"),)
        if result.to_embed
        else ()
    )
    steps = [("sift", "index", result.name)] if result.index_again else []
    steps += [("sift", "embed", result.name)] if result.to_embed else []
    return Shown((), (*lines, *embed, *(next_step(words, platform, style) for words in steps)))


def added(result: CollectionAdded | CollectionChanged, platform: str, style: Style) -> Shown:
    if isinstance(result, CollectionChanged):
        return _changed(result, platform, style)
    vault = ", as in an Obsidian vault" if result.vault else ""
    titles = (labelled("title", f"from file names{vault}"),) if result.title_from is TitleFrom.NAME else ()
    links = (labelled("links", "followed out of the folder"),) if result.follow_links else ()
    hidden = () if result.agents else (labelled("agent", "hidden from agents"),)
    return Shown(
        (),
        (
            labelled("added", f"{result.name}  {result.folder}", Tone.DONE, style),
            labelled("store", result.store),
            *titles,
            *links,
            *hidden,
            next_step(("sift", "index", result.name), platform, style),
        ),
    )


def _left(files: tuple[str, ...], style: Style) -> tuple[str, ...]:
    return tuple(
        line for file in files for line in (labelled("left", f"{file}: in use, not deleted"), *fix(LEFT_REMEDY, style))
    )


def removed(result: CollectionRemoved, _platform: str, style: Style) -> Shown:
    held = f" of {counted(result.documents, 'document')}" if result.documents is not None else ""
    unread = (labelled("store", f"could not be read: {result.problem.message}"),) if result.problem is not None else ()
    said = f"{result.name}: its store{held} deleted; its files stay where they are"
    return Shown((), (labelled("gone", said, Tone.DONE, style), *unread, *_left(result.left, style)))


def moved(result: StoresMoved, _platform: str, style: Style) -> Shown:
    lines = [
        line
        for each in result.moved
        for line in (
            labelled("moved", f"{each.name}  {each.new}{COPIED if each.copied else ''}", Tone.DONE, style),
            *_left(each.left, style),
        )
    ]
    already = (labelled("kept", f"{', '.join(result.already)}: there already"),) if result.already else ()
    return Shown((), (*lines, *already))


def switched(result: CollectionsSwitched, _platform: str, style: Style) -> Shown:
    state, label = ("active", "on") if result.active else ("inactive", "off")
    said = f"{', '.join(result.changed)}: {SWITCHED[result.active]}"
    changed = (labelled(label, said, Tone.DONE, style),) if result.changed else ()
    already = (labelled("kept", f"{', '.join(result.already)}: {state} already"),) if result.already else ()
    return Shown((), (*changed, *already))


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
        "passages",
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
            numbers = _counts(held, columns)
        case _:
            numbers = (NONE,) * (len(_header(columns)) - 2 - len(active))
    return (collection.name, collection.folder, *numbers, *active)


def _models(result: Status) -> tuple[str, ...]:
    readable = [each.contents for each in result.collections if isinstance(each.contents, Counted)]
    held = sorted({each.model for each in readable if each.model is not None})
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
