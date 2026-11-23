import errno
import json
import textwrap
from types import MappingProxyType
from typing import NamedTuple

from sift.api.collections import CollectionAdded, CollectionStatus, Counted, Status, TitleFrom
from sift.api.documents import Documents, DocumentText
from sift.api.embedding import EmbedResult
from sift.api.errors import InvalidRequestError, ModelUnavailableError, NotFoundError, SiftError, StoreUnavailableError
from sift.api.fields import Fields, FieldValue
from sift.api.indexing import IndexResult
from sift.api.links import LinkIn, LinkOut, Links, Unresolved
from sift.api.search import Hit, SearchResult, Unsearched
from sift.interfaces.console.progress import DOT
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import GAP, labelled, printable, printable_json, table
from sift.interfaces.console.units import duration, size

STATUS_HEADER = ("collection", "folder", "documents", "chunks", "size")
UNREADABLE_HEADER = ("collection", "file", "problem")
NONE = "-"
SNIPPET_CHARACTERS = 200
SMALLEST_SHOWN = 0.005
DONE, FAILED, USAGE = 0, 1, 2
ERROR_KINDS: MappingProxyType[type[SiftError], str] = MappingProxyType(
    {
        InvalidRequestError: "invalid_request",
        NotFoundError: "not_found",
        StoreUnavailableError: "store_unavailable",
        ModelUnavailableError: "model_unavailable",
    }
)
EXIT_CODES: MappingProxyType[type[SiftError], int] = MappingProxyType({InvalidRequestError: USAGE})
OS_REMEDIES = MappingProxyType(
    {
        errno.ENOENT: "check that the path exists",
        errno.EACCES: "check that you may read and write the path",
        errno.ENOSPC: "free space on the disk",
    }
)
OS_REMEDY = "check the path and its permissions"


class Shown(NamedTuple):
    results: tuple[str, ...]
    messages: tuple[str, ...]
    code: int = DONE


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


def _row(collection: CollectionStatus, *, linked: bool, unreadable: bool) -> tuple[str, ...]:
    held = collection.contents
    if not isinstance(held, Counted):
        return (collection.name, collection.folder, *(NONE,) * (len(STATUS_HEADER) - 2 + 2 * linked + unreadable))
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
    linked = any(isinstance(each.contents, Counted) and each.contents.links for each in result.collections)
    header = (*STATUS_HEADER, *(("links", "unresolved") if linked else ()), *(("unreadable",) * unreadable))
    numbers = frozenset(range(2, len(header)))
    rows = (header, *(_row(each, linked=linked, unreadable=unreadable) for each in result.collections))
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


def _score(score: float) -> str:
    return f"{score:.2f}" if score == 0 or abs(score) >= SMALLEST_SHOWN else f"{score:.1e}"


def _hit(rank: int, hit: Hit, ranks: int, scores: int) -> tuple[str, ...]:
    indent = " " * (ranks + scores + 2 * len(GAP))
    head = f"{rank:>{ranks}}{GAP}{_score(hit.score):>{scores}}{GAP}{printable(hit.title)}"
    text = (f"{indent}{_passage(hit.text)}",) if hit.text else ()
    also = (f"{indent}also {', '.join(printable(each) for each in hit.also)}",) if hit.also else ()
    fields = (f"{indent}{_fields(hit.fields)}",) if hit.fields else ()
    return (head, f"{indent}{printable(hit.id)}:{hit.lines}", *also, *fields, *text)


def _footer(result: SearchResult, seconds: float, style: Style, kind: str) -> str:
    found = f"{len(result.hits):,} result{'s' * (len(result.hits) != 1)}" if result.hits else "no matches"
    ruled_out = f", {result.ruled_out} ruled out by the links" if result.ruled_out else ""
    searched = f"{result.searched} collection{'s' * (result.searched != 1)}{ruled_out}"
    separator = f"{GAP}{DOT[0] if style.unicode else DOT[1]}{GAP}"
    partial = ("partial: better matches may be missing",) if result.partial else ()
    return separator.join((found, kind, *partial, searched, duration(seconds)))


def _unsearched(unsearched: tuple[Unsearched, ...], style: Style) -> tuple[str, ...]:
    return tuple(
        line
        for each in unsearched
        for line in (
            labelled("error", f"{each.collection}: {each.message}", Tone.ERROR, style),
            *_fix(each.remedy, style),
        )
    )


def hits(result: SearchResult, seconds: float, platform: str, style: Style, kind: str = "keyword") -> Shown:
    if not result.searched and not result.ruled_out and not result.unsearched:
        return _empty(platform, style)
    ranks = len(str(len(result.hits)))
    scores = max((len(_score(hit.score)) for hit in result.hits), default=0)
    lines = [line for rank, hit in enumerate(result.hits, 1) for line in _hit(rank, hit, ranks, scores)]
    problems = _unsearched(result.unsearched, style)
    return Shown(tuple(lines), (*problems, _footer(result, seconds, style, kind)))


def _error_json(kind: str, message: str, remedy: str) -> str:
    said = {"error": {"kind": kind, "message": message, "remedy": remedy}}
    return printable_json(json.dumps(said, ensure_ascii=False, indent=2))


def failure(error: SiftError | OSError, style: Style, *, as_json: bool = False) -> Shown:
    if isinstance(error, OSError):
        kind, message, remedy = "os", str(error), OS_REMEDIES.get(error.errno or 0, OS_REMEDY)
        code = FAILED
    else:
        kind, message, remedy = ERROR_KINDS[type(error)], error.message, error.remedy
        code = EXIT_CODES.get(type(error), FAILED)
    lines = (labelled("error", message, Tone.ERROR, style), *_fix(remedy, style))
    return Shown((_error_json(kind, message, remedy),) if as_json else (), lines, code)


def _written(link: LinkOut) -> str:
    field = f"{link.field}: " if link.field else ""
    fragment = f"#{link.fragment}" if link.fragment else ""
    return f"{field}{link.target}{fragment}"


def _leads(link: LinkOut) -> str:
    if link.id is not None:
        return link.id + (" (alias)" if link.via == "alias" else "")
    return f"ambiguous: {', '.join(link.ambiguous)}" if link.ambiguous else "unresolved"


def _through(through: str | None) -> str:
    return f"  (through {through})" if through else ""


def _out(heading: str, found: tuple[LinkOut, ...]) -> tuple[str, ...]:
    lines = [f"{each.line:>6}  {_written(each)}  ->  {_leads(each)}{_through(each.through)}" for each in found]
    return (f"{heading}:" if found else f"{heading}: nothing", *lines)


def _in(heading: str, found: tuple[LinkIn, ...]) -> tuple[str, ...]:
    lines = [
        line
        for each in found
        for line in (
            f"  {each.id}  {each.title}{_through(each.through)}",
            *(f"{linking.line:>6}  {linking.text}" for linking in each.lines),
        )
    ]
    return (f"{heading}:" if found else f"{heading}: nothing", *lines)


def links(result: Links, _platform: str, style: Style) -> Shown:
    note = f"{result.title} ({result.id})"
    out = () if result.out is None else _out(f"{note} links to", result.out)
    into = () if result.into is None else _in(f"Linking to {note}", result.into)
    return Shown(tuple(printable(line) for line in (*out, *into)), _unsearched(result.unsearched, style))


def unresolved(result: Unresolved, _platform: str, style: Style) -> Shown:
    problems = _unsearched(result.unsearched, style)
    if not result.targets:
        nothing = () if result.unsearched else (labelled("nothing", "every link leads to a note"),)
        return Shown((), (*problems, *nothing))
    rows = [
        (f"{each.documents:,}", each.target, ", ".join(each.ids) + (", ..." if each.documents > len(each.ids) else ""))
        for each in result.targets
    ]
    return Shown(table(rows, frozenset({0})), problems)
