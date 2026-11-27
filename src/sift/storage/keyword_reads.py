import itertools
import json
import queue
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager
from typing import NamedTuple, cast

from sift.api.errors import StoreUnavailableError
from sift.api.ids import DocumentId
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS
from sift.storage.database import Connection, Reading, Value, WritingDocuments, file_of, read_only, rows
from sift.storage.texts import TextSpan, spans_text, spans_text_at_once

DOCUMENTS = """SELECT id, substr(title, 1, ?) AS title, substr(meta, 1, ?) AS meta FROM keyword_documents
               WHERE id IN (SELECT value FROM json_each(?))"""
DOCUMENTS_AT_ONCE = """SELECT json_group_array(json_array(id, substr(title, 1, ?), substr(meta, 1, ?)))
                       FROM keyword_documents WHERE id IN (SELECT value FROM json_each(?))"""
AHEAD = 2
WAITED_SECONDS = 600.0
STUCK = "passages were not read within {seconds:.0f} s: reading the store is stuck"
STUCK_REMEDY = (
    "stop sift and index again: what it built so far is kept; if it sticks again, check the drive of the store"
)


class Span(NamedTuple):
    id: int
    document: DocumentId
    start: int
    length: int


class Named(NamedTuple):
    id: int
    title: str
    meta: str | None


class Passages(NamedTuple):
    spans: Sequence[Span]
    documents: list[Named]
    texts: list[str]


type ReadPart = Callable[[Reading, Sequence[Span]], Passages]
type ReadParts = Callable[[WritingDocuments, Sequence[Span]], AbstractContextManager[Iterator[Passages]]]


def _named(spans: Sequence[Span]) -> tuple[int, int, str]:
    cut = MOST_FIELD_CHARACTERS + 1
    return cut, cut, json.dumps(sorted({span.document for span in spans}))


def _texts(spans: Sequence[Span]) -> list[TextSpan]:
    return [TextSpan(span.document, span.start, span.length) for span in spans]


def passages_of(transaction: Reading, spans: Sequence[Span]) -> Passages:
    documents = rows(transaction, Named, DOCUMENTS, _named(spans))
    return Passages(spans, documents, spans_text(transaction, _texts(spans)))


def passages_at_once(transaction: Reading, spans: Sequence[Span]) -> Passages:
    (found,) = rows(transaction, Value, DOCUMENTS_AT_ONCE, _named(spans))
    documents = [Named._make(each) for each in cast("list[list[object]]", json.loads(found.value))]
    return Passages(spans, documents, spans_text_at_once(transaction, _texts(spans)))


@contextmanager
def read_in(transaction: WritingDocuments, spans: Sequence[Span]) -> Generator[Iterator[Passages]]:
    yield (passages_of(transaction, part) for part in itertools.batched(spans, transaction.batches.chunks_a_read))


def _ready(future: "Future[Passages]", waited: float) -> Passages:
    try:
        return future.result(timeout=waited)
    except TimeoutError:
        raise StoreUnavailableError(STUCK.format(seconds=waited), STUCK_REMEDY) from None


@contextmanager
def read_ahead(
    transaction: WritingDocuments,
    spans: Sequence[Span],
    read: ReadPart = passages_at_once,
    waited: float = WAITED_SECONDS,
) -> Generator[Iterator[Passages]]:
    parts = list(itertools.batched(spans, transaction.batches.chunks_a_read_beside))
    readers = max(1, min(transaction.batches.readers, len(parts)))
    idle: queue.SimpleQueue[Connection] = queue.SimpleQueue()

    def taken(part: Sequence[Span]) -> Passages:
        connection = idle.get()
        try:
            return read(Reading(connection), part)
        finally:
            idle.put(connection)

    with ExitStack() as held:
        path = file_of(transaction.connection)
        for _ in range(readers):
            idle.put(held.enter_context(closing(read_only(path))))
        pool = ThreadPoolExecutor(readers, thread_name_prefix="sift-read")
        ahead = readers * AHEAD
        pending = deque(pool.submit(taken, part) for part in parts[:ahead])

        def each() -> Iterator[Passages]:
            for at in range(len(parts)):
                future = pending.popleft()
                if at + ahead < len(parts):
                    pending.append(pool.submit(taken, parts[at + ahead]))
                yield _ready(future, waited)

        try:
            yield each()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True, cancel_futures=True)
