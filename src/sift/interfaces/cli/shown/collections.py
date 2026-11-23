from sift.api.collections import CollectionAdded, CollectionStatus, Counted, Status, TitleFrom, Unavailable
from sift.interfaces.cli.shown.outcome import Shown, empty, fix, next_step
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, table
from sift.interfaces.console.units import size

STATUS_HEADER = ("collection", "folder", "documents", "chunks", "size")
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


def _counted(collection: CollectionStatus, held: Counted, *, linked: bool, unreadable: bool) -> tuple[str, ...]:
    links = (f"{held.links:,}", f"{held.unresolved:,}") if linked else ()
    unread = (f"{held.unreadable:,}",) if unreadable else ()
    return (
        collection.name,
        collection.folder,
        f"{held.documents:,}",
        f"{held.chunks:,}",
        size(held.stored_bytes),
        *links,
        *unread,
    )


def _row(collection: CollectionStatus, *, linked: bool, unreadable: bool) -> tuple[str, ...]:
    match collection.contents:
        case Counted() as held:
            return _counted(collection, held, linked=linked, unreadable=unreadable)
        case _:
            return (collection.name, collection.folder, *(NONE,) * (len(STATUS_HEADER) - 2 + 2 * linked + unreadable))


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
    unreadable = any(isinstance(each.contents, Counted) and each.contents.unreadable for each in result.collections)
    linked = any(isinstance(each.contents, Counted) and each.contents.links for each in result.collections)
    header = (*STATUS_HEADER, *(("links", "unresolved") if linked else ()), *(("unreadable",) * unreadable))
    numbers = frozenset(range(2, len(header)))
    rows = (header, *(_row(each, linked=linked, unreadable=unreadable) for each in result.collections))
    messages = tuple(line for each in result.collections for line in _problems(each, style))
    return Shown((*table(rows, numbers), *_unreadable_files(result)), messages)
