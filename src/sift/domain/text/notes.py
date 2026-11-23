import hashlib
import posixpath
from dataclasses import dataclass

from sift.api.collections import TitleFrom
from sift.api.fields import Fields
from sift.domain.text.chunks import Chunk, chunks
from sift.domain.text.decoding import file_text
from sift.domain.text.front_matter import FrontMatter, front_matter
from sift.domain.text.json_values import REPLACEMENT, composed_fields
from sift.domain.text.links import MARKDOWN, Link, links
from sift.domain.text.markdown import heading, one_line

TEXT_FILES = (".md", ".markdown", ".txt")
NUL = "\N{NULL}"


@dataclass(frozen=True, slots=True)
class Document:
    key: str
    digest: str
    title: str
    fields: Fields
    text: str
    chunks: tuple[Chunk, ...]
    links: tuple[Link, ...]


def storable(text: str) -> str:
    return text.replace(NUL, REPLACEMENT)


def digest(raw: bytes) -> str:
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def title_field(fields: Fields) -> str | None:
    title = fields.get("title")
    if isinstance(title, bool) or not isinstance(title, str | int | float):
        return None
    return one_line(str(title)) or None


def file_stem(name: str) -> str:
    return posixpath.splitext(name)[0]


def title(fields: Fields, body: str, name: str, title_from: TitleFrom) -> str:
    chosen = title_field(fields)
    if chosen is None and title_from is TitleFrom.HEADING:
        chosen = heading(body)
    return chosen or file_stem(name)


def _chunks_of(text: str, head: FrontMatter | None, first_line: int) -> tuple[Chunk, ...]:
    end = head.end if head else 0
    found = chunks(text[end:], first_line, end)
    if found or head is None or not head.inner:
        return found
    return chunks(head.inner, 2, head.inner_start)


def note(raw: bytes, name: str, title_from: TitleFrom = TitleFrom.HEADING) -> Document:
    text = file_text(raw)
    head = front_matter(text)
    fields: Fields = composed_fields(head.fields) if head else {}
    end = head.end if head else 0
    first_line = 1 + text.count("\n", 0, end)
    found = links(text, end, first_line) if name.lower().endswith(MARKDOWN) else ()
    named = title(fields, text[end:], name, title_from)
    return Document("", digest(raw), storable(named), fields, storable(text), _chunks_of(text, head, first_line), found)
