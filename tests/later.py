from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from functools import partial

from sift.app.filing import Readers
from sift.domain.text.documents import Convert, Outcome
from sift.ingest.documents import convert
from sift.ingest.files import at_most
from sift.ingest.readers import ReadWhole, ToRead, read_file
from sift.ingest.sandbox import FileReading, OpenReading, ReadOne

LATER_NS = 2**62


def later() -> int:
    return LATER_NS


def _one_by_one(read: ReadOne, asked: Sequence[ToRead]) -> Iterator[tuple[int, Outcome]]:
    return enumerate(map(read, asked))


@contextmanager
def in_process(read: ReadOne = read_file) -> Generator[FileReading]:
    yield partial(_one_by_one, read)


def reading(whole: ReadWhole = at_most, converting: Convert = convert) -> OpenReading:
    return partial(in_process, partial(read_file, whole=whole, converting=converting))


LATER = Readers(clock=later)
