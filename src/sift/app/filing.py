from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sift.api.events import Reporter
from sift.domain.plan.indexing import MOST_TEXT_BYTES
from sift.storage.catalog import Collection
from sift.storage.database import BATCHES, Batches, Connection


@dataclass(frozen=True, slots=True)
class Limits:
    most_text_bytes: int = MOST_TEXT_BYTES
    most_files_a_write: int = 1000
    most_bytes_a_write: int = 64 << 20
    most_records_a_write: int = 2000
    batches: Batches = BATCHES


@dataclass(frozen=True, slots=True)
class Tally:
    added: int = 0
    changed: int = 0
    removed: int = 0

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(self.added + other.added, self.changed + other.changed, self.removed + other.removed)


type Lines = Callable[[Path], Iterator[tuple[int, str]]]


@dataclass(frozen=True, slots=True)
class Job:
    collection: Collection
    connection: Connection
    report: Reporter
    limits: Limits
    lines: Lines
