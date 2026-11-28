import hashlib
import itertools
import posixpath
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

from sift.api.collections import TitleFrom
from sift.api.fields import Fields
from sift.domain.text.chunks import Chunk, chunk_of, chunks, groups, pieces
from sift.domain.text.decoding import text_pieces
from sift.domain.text.front_matter import FrontMatter, front_matter, front_matter_links
from sift.domain.text.json_values import REPLACEMENT, composed_fields
from sift.domain.text.links import MARKDOWN, WIKILINK_OPENS, Link, in_line
from sift.domain.text.markdown import TextLine, heading_of, text_lines, title_of, with_prose

NUL = "\N{NULL}"
NEWLINE = "\n"
HEAD_OPENS = re.compile(r"---[ \t]*\n")
HEAD_CLOSES = re.compile(r"---[ \t]*\n?")
MOST_HEAD = 1 << 20
FRONT_MATTER_TOO_LONG = f"its front matter is longer than the {MOST_HEAD:,} characters sift reads as fields"
DIGEST_BYTES = 16
LINK_ADDRESS_OPENS = "]("


@dataclass(frozen=True, slots=True)
class Document:
    key: str
    digest: str
    title: str
    fields: Fields
    text: str
    chunks: tuple[Chunk, ...]
    links: tuple[Link, ...]


@dataclass(frozen=True, slots=True)
class NoteTitle:
    title: str


type NotePart = str | Chunk | Link | NoteTitle


@dataclass(frozen=True, slots=True)
class NoteAsked:
    name: str
    title_from: TitleFrom
    fields: Fields


@dataclass(frozen=True, slots=True)
class NoteRead:
    fields: Fields
    parts: Iterator[NotePart]
    problem: str | None = None


def storable(text: str) -> str:
    return text.replace(NUL, REPLACEMENT)


def digest(raw: bytes) -> str:
    return hashlib.blake2b(raw, digest_size=DIGEST_BYTES).hexdigest()


def title_field(fields: Fields) -> str | None:
    title = fields.get("title")
    if isinstance(title, bool) or not isinstance(title, str | int | float):
        return None
    return title_of(str(title)) or None


def file_stem(name: str) -> str:
    return posixpath.splitext(name)[0]


class _HeadEnd(Enum):
    CLOSED = "closed"
    NONE = "none"
    TOO_LONG = "too long"


def _head(lines: Iterator[TextLine]) -> tuple[list[TextLine], _HeadEnd]:
    taken: list[TextLine] = []
    size = 0
    for line in lines:
        taken.append(line)
        size += len(line.text)
        opening = len(taken) == 1
        if opening and not HEAD_OPENS.fullmatch(line.text):
            return taken, _HeadEnd.NONE
        if not line.whole or size > MOST_HEAD:
            return taken, _HeadEnd.TOO_LONG
        if not opening and HEAD_CLOSES.fullmatch(line.text):
            return taken, _HeadEnd.CLOSED
    return taken, _HeadEnd.NONE


def _linking(text: str) -> bool:
    return WIKILINK_OPENS in text or LINK_ADDRESS_OPENS in text


def _body(lines: Iterable[TextLine], *, markdown: bool, headings: list[str]) -> Iterator[NotePart]:
    texts: list[str] = []
    links: list[Link] = []

    def tapped() -> Iterator[tuple[TextLine, bool]]:
        for line, outside in with_prose(lines):
            texts.append(line.text)
            if outside and markdown and _linking(line.text):
                links.extend(in_line(line.text.removesuffix(NEWLINE), line.number))
            if outside and not headings and (found := heading_of(line.text)):
                headings.append(found)
            yield line, outside

    for group in groups(pieces(tapped())):
        if texts:
            yield storable("".join(texts))
            texts.clear()
        yield from links
        links.clear()
        if (found := chunk_of(group)) is not None:
            yield found


def _parts(found: FrontMatter | None, head: str, body: Iterable[TextLine], read: NoteAsked) -> Iterator[NotePart]:
    markdown = read.name.lower().endswith(MARKDOWN)
    if found:
        yield storable(head)
        if markdown and _linking(head):
            yield from front_matter_links(found)
    headings: list[str] = []
    chunked_body = False
    for part in _body(body, markdown=markdown, headings=headings):
        chunked_body = chunked_body or isinstance(part, Chunk)
        yield part
    if found and found.inner and (found.lost or not chunked_body):
        yield from chunks(found.inner, 2, found.inner_start)
    chosen = title_field(read.fields) or (headings[0] if headings and read.title_from is TitleFrom.HEADING else None)
    yield NoteTitle(storable(chosen or file_stem(read.name)))


def read_note(pieces: Iterable[str], name: str, title_from: TitleFrom = TitleFrom.HEADING) -> NoteRead:
    lines = text_lines(pieces)
    taken, ended = _head(lines)
    closed = ended is _HeadEnd.CLOSED
    head = "".join(line.text for line in taken) if closed else ""
    found = front_matter(head) if closed else None
    body = lines if found else itertools.chain(taken, lines)
    fields: Fields = composed_fields(found.fields) if found else {}
    problem = FRONT_MATTER_TOO_LONG if ended is _HeadEnd.TOO_LONG else None
    return NoteRead(fields, _parts(found, head, body, NoteAsked(name, title_from, fields)), problem)


def document_of(read: NoteRead, digested: str) -> Document:
    texts: list[str] = []
    found_chunks: list[Chunk] = []
    found_links: list[Link] = []
    title = ""
    for part in read.parts:
        match part:
            case str():
                texts.append(part)
            case Chunk():
                found_chunks.append(part)
            case Link():
                found_links.append(part)
            case _:
                title = part.title
    return Document("", digested, title, read.fields, "".join(texts), tuple(found_chunks), tuple(found_links))


def note(raw: bytes, name: str, title_from: TitleFrom = TitleFrom.HEADING) -> Document:
    return document_of(read_note(text_pieces((raw,)), name, title_from), digest(raw))
