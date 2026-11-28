from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from functools import partial

from sift.app.filing import Readers
from sift.domain.text.documents import Convert
from sift.ingest.documents import convert
from sift.ingest.files import at_most
from sift.ingest.pictures import picture
from sift.ingest.readers import ReadWhole, read_file
from sift.ingest.sandbox import FileReading, OpenReading, Picturing, ReadOne

LATER_NS = 2**62


def later() -> int:
    return LATER_NS


def _one_by_one[A, B](read: Callable[[A], B], asked: Sequence[A]) -> Iterator[tuple[int, B]]:
    return enumerate(map(read, asked))


@contextmanager
def in_process(read: ReadOne = read_file) -> Generator[FileReading]:
    yield partial(_one_by_one, read)


@contextmanager
def pictured_here() -> Generator[Picturing]:
    yield partial(_one_by_one, picture)


def reading(whole: ReadWhole = at_most, converting: Convert = convert) -> OpenReading:
    return partial(in_process, partial(read_file, whole=whole, converting=converting))


LATER = Readers(clock=later)
