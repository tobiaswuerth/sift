import posixpath
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

from sift.api.fields import Fields
from sift.domain.text.chunks import chunks
from sift.domain.text.json_values import composed_fields
from sift.domain.text.markdown import heading
from sift.domain.text.notes import Document, digest, file_stem, storable, title_field
from sift.domain.text.scans import OcrSetting

NO_TEXT = "no text in it"
SCANNED = ".pdf"
NO_FIELDS: Fields = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Converted:
    markdown: str
    properties: Fields = NO_FIELDS


@dataclass(frozen=True, slots=True)
class Unread:
    problem: str


@dataclass(frozen=True, slots=True)
class Scan:
    document: Document
    raw: bytes


@dataclass(frozen=True, slots=True)
class FileRead:
    documents: tuple[Document, ...]
    problems: tuple[str, ...] = ()
    scans: tuple[Scan, ...] = ()
    read_again: bool = False
    skipped_under: OcrSetting | None = None


@dataclass(frozen=True, slots=True)
class Refused:
    problem: str
    read_again: bool


type Conversion = Converted | Unread
type Convert = Callable[[bytes, str], Conversion]
type Outcome = FileRead | Refused


def too_large(size: int, most: int) -> Refused:
    return Refused(f"{size:,} bytes: sift reads documents of at most {most:,} bytes; split the file", read_again=False)


def _scanned(name: str) -> bool:
    return posixpath.splitext(name)[1].lower() == SCANNED


def document(raw: bytes, name: str, conversion: Conversion) -> Document | Unread | Scan:
    if isinstance(conversion, Unread):
        return conversion
    text = storable(conversion.markdown)
    fields = composed_fields(conversion.properties)
    if not text.strip():
        unread = Document("", digest(raw), storable(title_field(fields) or file_stem(name)), fields, "", (), ())
        return Scan(unread, raw) if _scanned(name) else Unread(NO_TEXT)
    named = title_field(fields) or heading(text) or file_stem(name)
    return Document("", digest(raw), storable(named), fields, text, chunks(text), ())
