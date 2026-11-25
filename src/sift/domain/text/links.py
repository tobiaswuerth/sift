import posixpath
import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple
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
WIKILINK_OPENS = "[["
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


class NamedTarget(NamedTuple):
    target: str
    key: str


def folded(name: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def spelled(name: str) -> str:
    return unicodedata.normalize("NFC", name).lower()


def _keyed(name: str, key: Callable[[str], str]) -> str:
    stem, extension = posixpath.splitext(name)
    return key(stem if extension.lower() in MARKDOWN else name)


def name_key(name: str) -> str:
    return _keyed(name, folded)


def spelled_key(name: str) -> str:
    return _keyed(name, spelled)


def last_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def target_key(target: str) -> str:
    return name_key(last_name(target))


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
    last = last_name(target)
    if last in ("", ".", "..") or posixpath.splitext(last)[1].lower() in NOT_READ:
        return None
    fragment, shown = written.fragment.strip() or None, written.shown.strip() or None
    return Link(field, line, written.kind, target, name_key(last), fragment, shown)


def _split(inner: str) -> tuple[str, str, str]:
    target, _, shown = inner.partition("|")
    target, _, fragment = target.removesuffix("\\").partition("#")
    return target, fragment, shown


def named_target(text: str) -> NamedTarget:
    written = text.strip()
    found = WIKILINK.fullmatch(written)
    target = _split(found[2] if found else written)[0].strip()
    return NamedTarget(target, target_key(target))


def _wikilinks(line: str, number: int, field: str | None) -> Iterator[tuple[int, Link | None]]:
    for found in WIKILINK.finditer(line):
        target, fragment, shown = _split(found[2])
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
    own = Name(target_key(path), NameKind.NAME)
    return tuple(sorted({own, *(Name(name_key(alias), NameKind.ALIAS) for alias in aliases(fields))}))


def link_target(text: str) -> str | None:
    written = text.strip()
    found = in_line(written, 1) if WIKILINK.fullmatch(written) else []
    return found[0].target if found else None
