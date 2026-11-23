import posixpath
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import unquote

from sift.api.fields import Fields, FieldValue

MARKDOWN = (".md", ".markdown")
NOT_READ = frozenset(
    [
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".ico",
        ".jpeg",
        ".jpg",
        ".png",
        ".svg",
        ".tif",
        ".tiff",
        ".webp",
        ".3gp",
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".avi",
        ".mkv",
        ".mov",
        ".mp4",
        ".ogv",
        ".webm",
        ".base",
        ".canvas",
        ".csv",
        ".json",
        ".tsv",
        ".xml",
        ".yaml",
        ".yml",
        ".7z",
        ".gz",
        ".rar",
        ".tar",
        ".zip",
        ".doc",
        ".ppt",
        ".xls",
    ]
)
WIKILINK = re.compile(r"(!?)\[\[([^\[\]\n]+)\]\]")
SHOWN = r"!?\[((?:[^\[\]\n]|\[[^\[\]\n]*\])*)\]"
ADDRESS = r"\(\s*(<[^<>\n]*>|(?:[^\s()<>]|\([^\s()<>]*\))+)"
LINK_TITLE = r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"
MARKDOWN_LINK = re.compile(SHOWN + ADDRESS + LINK_TITLE)
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]+:")
BACKTICKS = re.compile(r"`+")
ALIAS_FIELDS = ("aliases", "alias")


class LinkKind(StrEnum):
    WIKILINK = "wikilink"
    EMBED = "embed"
    MARKDOWN = "markdown"


class NameKind(StrEnum):
    NAME = "name"
    ALIAS = "alias"


@dataclass(frozen=True, slots=True)
class Link:
    field: str | None
    line: int
    kind: LinkKind
    target: str
    key: str
    fragment: str | None
    shown: str | None


@dataclass(frozen=True, slots=True, order=True)
class Name:
    key: str
    kind: NameKind


def folded(name: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def name_key(name: str) -> str:
    stem, extension = posixpath.splitext(name)
    return folded(stem if extension.lower() in MARKDOWN else name)


def without_code_spans(line: str) -> str:
    if "`" not in line:
        return line
    runs = [found.span() for found in BACKTICKS.finditer(line)]
    following: list[int | None] = [None] * len(runs)
    last_of_length: dict[int, int] = {}
    for index in range(len(runs) - 1, -1, -1):
        length = runs[index][1] - runs[index][0]
        following[index], last_of_length[length] = last_of_length.get(length), index
    parts: list[str] = []
    index, end = 0, 0
    while index < len(runs):
        closing = following[index]
        if closing is None:
            index += 1
            continue
        parts += [line[end : runs[index][0]], " " * (runs[closing][1] - runs[index][0])]
        index, end = closing + 1, runs[closing][1]
    return "".join(parts) + line[end:]


@dataclass(frozen=True, slots=True)
class Written:
    kind: LinkKind
    target: str
    fragment: str
    shown: str


def _link(field: str | None, line: int, written: Written) -> Link | None:
    target = written.target.strip()
    last = target.rsplit("/", 1)[-1]
    if last in ("", ".", "..") or posixpath.splitext(last)[1].lower() in NOT_READ:
        return None
    fragment, shown = written.fragment.strip() or None, written.shown.strip() or None
    return Link(field, line, written.kind, target, name_key(last), fragment, shown)


def _wikilinks(line: str, number: int, field: str | None) -> Iterator[tuple[int, Link | None]]:
    for found in WIKILINK.finditer(line):
        target, _, shown = found[2].partition("|")
        target, _, fragment = target.removesuffix("\\").partition("#")
        kind = LinkKind.EMBED if found[1] else LinkKind.WIKILINK
        yield found.start(), _link(field, number, Written(kind, target, fragment, shown))


def _markdown_links(line: str, number: int) -> Iterator[tuple[int, Link | None]]:
    blanked = WIKILINK.sub(lambda found: " " * len(found[0]), line)
    for found in MARKDOWN_LINK.finditer(blanked):
        address = found[2][1:-1] if found[2].startswith("<") else found[2]
        if not SCHEME.match(address) and not address.startswith(("#", "//")):
            target, _, fragment = address.partition("#")
            written = Written(LinkKind.MARKDOWN, unquote(target), unquote(fragment), found[1])
            yield found.start(), _link(None, number, written)


def in_line(line: str, number: int, field: str | None = None) -> list[Link]:
    line = without_code_spans(line)
    found = [*_wikilinks(line, number, field)]
    if field is None and "](" in line:
        found += _markdown_links(line, number)
    return [link for _, link in sorted(found, key=lambda item: item[0]) if link]


def front_matter_links(head: str) -> Iterator[Link]:
    field: str | None = None
    for number, line in enumerate(head.split("\n")[:-1], 1):
        name, colon, rest = line.partition(":")
        starts = bool(colon and name.strip() and not name[0].isspace() and name[0] not in "#-")
        field = name.strip() if starts else field
        written = rest if starts else line
        if field and "[[" in written and not written.lstrip().startswith("#"):
            yield from in_line(written, number, field)


def each_of(found: FieldValue) -> Sequence[FieldValue]:
    return found if isinstance(found, Sequence) and not isinstance(found, str) else (found,)


def aliases(fields: Fields) -> list[str]:
    return [
        alias.strip()
        for field in ALIAS_FIELDS
        for alias in each_of(fields.get(field))
        if isinstance(alias, str) and alias.strip()
    ]


def names(path: str, fields: Fields) -> tuple[Name, ...]:
    own = Name(name_key(path.rsplit("/", 1)[-1]), NameKind.NAME)
    return tuple(sorted({own, *(Name(name_key(alias), NameKind.ALIAS) for alias in aliases(fields))}))


def link_target(text: str) -> str | None:
    written = text.strip()
    found = in_line(written, 1) if WIKILINK.fullmatch(written) else []
    return found[0].target if found else None
