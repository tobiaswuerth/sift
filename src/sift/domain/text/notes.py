import hashlib
import itertools
import posixpath
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from sift.api.collections import TitleFrom
from sift.api.fields import Fields
from sift.domain.text.chunks import Chunk, chunk_of, chunks, groups, pieces
from sift.domain.text.decoding import text_pieces
from sift.domain.text.front_matter import FrontMatter, front_matter
from sift.domain.text.json_values import REPLACEMENT, composed_fields
from sift.domain.text.links import MARKDOWN, Link, front_matter_links, in_line
from sift.domain.text.markdown import TextLine, heading_of, text_lines, title_of, with_prose

NUL = "\N{NULL}"
NEWLINE = "\n"
HEAD_OPENS = re.compile(r"---[ \t]*\n")
HEAD_CLOSES = re.compile(r"---[ \t]*\n?")
MOST_HEAD = 1 << 20
DIGEST_BYTES = 16
WIKILINK_OPENS, LINK_ADDRESS_OPENS = "[[", "]("


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


def _head(lines: Iterator[TextLine]) -> tuple[list[TextLine], bool]:
    taken: list[TextLine] = []
    size = 0
    for line in lines:
        taken.append(line)
        size += len(line.text)
        opening = len(taken) == 1
        if not line.whole or size > MOST_HEAD or (opening and not HEAD_OPENS.fullmatch(line.text)):
            return taken, False
        if not opening and HEAD_CLOSES.fullmatch(line.text):
            return taken, True
    return taken, False


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
            yield from front_matter_links(head)
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
    taken, closed = _head(lines)
    head = "".join(line.text for line in taken) if closed else ""
    found = front_matter(head) if closed else None
    body = lines if found else itertools.chain(taken, lines)
    fields: Fields = composed_fields(found.fields) if found else {}
    return NoteRead(fields, _parts(found, head, body, NoteAsked(name, title_from, fields)))


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
