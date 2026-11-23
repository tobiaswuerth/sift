import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sift.api.events import Reporter
from sift.domain.plan.indexing import MOST_FILE_BYTES, MOST_TEXT_BYTES, Sizes
from sift.domain.text.documents import Convert
from sift.domain.text.records import SkippedLine
from sift.ingest.documents import convert
from sift.ingest.files import Identity, at_most
from sift.ingest.records import lines_of
from sift.storage.catalog import Collection
from sift.storage.database import BATCHES, Batches, Connection


@dataclass(frozen=True, slots=True)
class Limits:
    most_text_bytes: int = MOST_TEXT_BYTES
    most_file_bytes: int = MOST_FILE_BYTES
    most_files_a_write: int = 1000
    most_bytes_a_write: int = 64 << 20
    most_records_a_write: int = 2000
    batches: Batches = BATCHES

    @property
    def sizes(self) -> Sizes:
        return Sizes(self.most_text_bytes, self.most_file_bytes)


@dataclass(frozen=True, slots=True)
class Tally:
    added: int = 0
    changed: int = 0
    removed: int = 0

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(self.added + other.added, self.changed + other.changed, self.removed + other.removed)


type Lines = Callable[[Path, int, Identity | None], Iterator[tuple[int, str | SkippedLine]]]
type ReadWhole = Callable[[Path, int, Identity | None], bytes]


@dataclass(frozen=True, slots=True)
class Readers:
    lines: Lines = lines_of
    whole: ReadWhole = at_most
    clock: Callable[[], int] = time.time_ns
    convert: Convert = convert


READERS = Readers()


@dataclass(frozen=True, slots=True)
class Job:
    collection: Collection
    connection: Connection
    report: Reporter
    limits: Limits
    readers: Readers
