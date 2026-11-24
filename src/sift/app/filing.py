import functools
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from sift.api.events import Reporter
from sift.app.scans import SCANNING, ScanModel, Scanning
from sift.domain.plan.indexing import MOST_FILE_BYTES
from sift.domain.text.notes import Document
from sift.domain.text.records import MOST_RECORD_CHARACTERS, SkippedLine
from sift.ingest.files import Identity, pieces_of
from sift.ingest.records import lines_of
from sift.ingest.sandbox import OpenReading, sandboxed
from sift.storage.catalog import Collection
from sift.storage.database import BATCHES, Batches, Connection
from sift.storage.documents import MOST_DESCRIPTION_BYTES, description_bytes


@dataclass(frozen=True, slots=True)
class Limits:
    most_record_characters: int = MOST_RECORD_CHARACTERS
    most_file_bytes: int = MOST_FILE_BYTES
    most_description_bytes: int = MOST_DESCRIPTION_BYTES
    most_files_a_write: int = 1000
    most_bytes_a_write: int = 64 << 20
    most_records_a_write: int = 2000
    most_characters_a_write: int = 64 << 20
    batches: Batches = BATCHES


LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class Tally:
    added: int = 0
    changed: int = 0
    removed: int = 0

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(self.added + other.added, self.changed + other.changed, self.removed + other.removed)


def too_large_to_describe(document: Document, most: int) -> str | None:
    size = description_bytes(document)
    return f"{size:,} bytes of name, title and fields, more than the {most:,} a document holds" if size > most else None


type Lines = Callable[[Path, int, Identity | None], Iterator[tuple[int, str | SkippedLine]]]
type ReadPieces = Callable[[Path, Identity | None], AbstractContextManager[Iterable[bytes]]]


DISTRIBUTION = "sift-rag"


@functools.cache
def installed() -> str:
    return metadata.version(DISTRIBUTION)


@dataclass(frozen=True, slots=True)
class Readers:
    lines: Lines = lines_of
    pieces: ReadPieces = pieces_of
    clock: Callable[[], int] = time.time_ns
    files: OpenReading = sandboxed
    scanning: Scanning = SCANNING
    release: Callable[[], str] = installed


READERS = Readers()


@dataclass(frozen=True, slots=True)
class Job:
    collection: Collection
    connection: Connection
    report: Reporter
    limits: Limits
    readers: Readers
    ocr: ScanModel
