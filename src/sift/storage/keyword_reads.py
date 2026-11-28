import itertools
import json
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from typing import NamedTuple, cast

from sift.api.ids import DocumentId
from sift.domain.errors import DamagedError
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS
from sift.storage.database import Reading, Value, WritingDocuments, file_of, rows
from sift.storage.readers import WAITED_SECONDS, readers_of, ready
from sift.storage.texts import TextSpan, spans_text, spans_text_at_once

DOCUMENTS = """SELECT id, substr(title, 1, ?) AS title, substr(meta, 1, ?) AS meta FROM keyword_documents
               WHERE id IN (SELECT value FROM json_each(?))"""
DOCUMENTS_AT_ONCE = """SELECT json_group_array(json_array(id, substr(title, 1, ?), substr(meta, 1, ?)))
                       FROM keyword_documents WHERE id IN (SELECT value FROM json_each(?))"""
AHEAD = 2
MISSING = "the document of a passage is missing from the store"
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


def _whole(passages: Passages) -> Passages:
    held = {each.id for each in passages.documents}
    if any(span.document not in held for span in passages.spans):
        raise DamagedError(MISSING)
    return passages


def passages_of(transaction: Reading, spans: Sequence[Span]) -> Passages:
    documents = rows(transaction, Named, DOCUMENTS, _named(spans))
    return _whole(Passages(spans, documents, spans_text(transaction, _texts(spans))))


def passages_at_once(transaction: Reading, spans: Sequence[Span]) -> Passages:
    (found,) = rows(transaction, Value, DOCUMENTS_AT_ONCE, _named(spans))
    documents = [Named._make(each) for each in cast("list[list[object]]", json.loads(found.value))]
    return _whole(Passages(spans, documents, spans_text_at_once(transaction, _texts(spans))))


@contextmanager
def read_in(transaction: WritingDocuments, spans: Sequence[Span]) -> Generator[Iterator[Passages]]:
    yield (passages_of(transaction, part) for part in itertools.batched(spans, transaction.batches.chunks_a_read))


@contextmanager
def read_ahead(
    transaction: WritingDocuments,
    spans: Sequence[Span],
    read: ReadPart = passages_at_once,
    waited: float = WAITED_SECONDS,
) -> Generator[Iterator[Passages]]:
    parts = list(itertools.batched(spans, transaction.batches.chunks_a_read_beside))
    readers = max(1, min(transaction.batches.readers, len(parts)))
    connections = readers_of(file_of(transaction.connection), readers)

    def taken(part: Sequence[Span]) -> Passages:
        connection = connections.taken()
        try:
            return read(Reading(connection), part)
        finally:
            connections.given_back(connection)

    pool = ThreadPoolExecutor(readers, thread_name_prefix="sift-read")
    ahead = readers * AHEAD

    def each(pending: deque["Future[Passages]"]) -> Iterator[Passages]:
        for at in range(len(parts)):
            future = pending.popleft()
            if at + ahead < len(parts):
                pending.append(pool.submit(taken, parts[at + ahead]))
            yield ready(future, waited, STUCK, STUCK_REMEDY)

    try:
        yield each(deque(pool.submit(taken, part) for part in parts[:ahead]))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        connections.leave()
        raise
    pool.shutdown(wait=True, cancel_futures=True)
    connections.leave()
