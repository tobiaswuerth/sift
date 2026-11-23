import json
import textwrap
from typing import NamedTuple

from sift.api.collections import CollectionAdded, CollectionStatus, Counted, Status, TitleFrom
from sift.api.documents import Documents, DocumentText
from sift.api.embedding import EmbedResult
from sift.api.errors import SiftError
from sift.api.fields import Fields, FieldValue
from sift.api.indexing import IndexResult
from sift.api.search import Hit, SearchResult
from sift.interfaces.console.progress import DOT
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import GAP, labelled, printable, table
from sift.interfaces.console.units import duration, size

STATUS_HEADER = ("collection", "folder", "documents", "chunks", "size")
UNREADABLE_HEADER = ("collection", "file", "problem")
NONE = "-"
SNIPPET_CHARACTERS = 200


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


def embedded(result: EmbedResult, platform: str, style: Style) -> Shown:
    return _empty(platform, style) if not result.collections else Shown((), ())


def _text(text: str) -> tuple[str, ...]:
    return (text.removesuffix("\n"),) if text else ()


def _document(found: DocumentText) -> tuple[str, ...]:
    held = tuple(printable(f"{key}: {_value(value)}") for key, value in (found.fields or {}).items())
    return (*held, "", *_text(found.text)) if held else _text(found.text)


def documents(result: Documents, _platform: str, style: Style) -> Shown:
    if len(result.found) == 1 and not result.unfound:
        return Shown(_document(result.found[0]), ())
    shown = [line for found in result.found for line in (f"==> {printable(found.id)} <==", *_document(found), "")]
    messages = [
        line
        for unfound in result.unfound
        for line in (
            labelled("error", f"{unfound.id}: {unfound.message}", Tone.ERROR, style),
            *_fix(unfound.remedy, style),
        )
    ]
    return Shown(tuple(shown[:-1]), tuple(messages))


def _passage(text: str) -> str:
    return printable(textwrap.shorten(" ".join(text.split()), SNIPPET_CHARACTERS, placeholder=" ..."))


def _value(value: FieldValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _fields(fields: Fields) -> str:
    return printable("  ".join(f"{key}: {_value(value)}" for key, value in fields.items()))


def _hit(rank: int, hit: Hit, ranks: int, scores: int) -> tuple[str, ...]:
    indent = " " * (ranks + scores + 2 * len(GAP))
    head = f"{rank:>{ranks}}{GAP}{hit.score:>{scores}.2f}{GAP}{printable(hit.title)}"
    text = (f"{indent}{_passage(hit.text)}",) if hit.text else ()
    also = (f"{indent}also {', '.join(printable(each) for each in hit.also)}",) if hit.also else ()
    fields = (f"{indent}{_fields(hit.fields)}",) if hit.fields else ()
    return (head, f"{indent}{printable(hit.id)}:{hit.lines}", *also, *fields, *text)


def _footer(result: SearchResult, seconds: float, style: Style, kind: str) -> str:
    found = f"{len(result.hits):,} result{'s' * (len(result.hits) != 1)}" if result.hits else "no matches"
    searched = f"{result.searched} collection{'s' * (result.searched != 1)}"
    separator = f"{GAP}{DOT[0] if style.unicode else DOT[1]}{GAP}"
    partial = ("partial: better matches may be missing",) if result.partial else ()
    return separator.join((found, kind, *partial, searched, duration(seconds)))


def hits(result: SearchResult, seconds: float, platform: str, style: Style, kind: str = "keyword") -> Shown:
    if not result.searched and not result.unsearched:
        return _empty(platform, style)
    ranks = len(str(len(result.hits)))
    scores = max((len(f"{hit.score:.2f}") for hit in result.hits), default=0)
    lines = [line for rank, hit in enumerate(result.hits, 1) for line in _hit(rank, hit, ranks, scores)]
    problems = [
        line
        for each in result.unsearched
        for line in (
            labelled("error", f"{each.collection}: {each.message}", Tone.ERROR, style),
            *_fix(each.remedy, style),
        )
    ]
    return Shown(tuple(lines), (*problems, _footer(result, seconds, style, kind)), failed=bool(result.unsearched))


def failure(error: SiftError | OSError, style: Style) -> Shown:
    if isinstance(error, OSError):
        return Shown((), (labelled("error", str(error), Tone.ERROR, style),), failed=True)
    return Shown((), (labelled("error", error.message, Tone.ERROR, style), *_fix(error.remedy, style)), failed=True)
