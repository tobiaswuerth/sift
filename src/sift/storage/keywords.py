import dataclasses
import hashlib
import json
import re
from collections import Counter
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import NamedTuple, cast

import numpy as np

from sift.api.errors import StoreUnavailableError
from sift.api.ids import CollectionId, DocumentId, first_row
from sift.domain.arrays import ints
from sift.domain.codec.segments import FORMAT, GAP, SHAPE, Segment, Shape, segment, stream
from sift.domain.errors import DamagedError
from sift.domain.text.tokens import TABLE_DIGEST, tokens
from sift.storage.database import (
    BATCHES,
    Batches,
    Connection,
    Reading,
    Writing,
    WritingDocuments,
    reading,
    rows,
    run,
    run_each,
    writing,
)
from sift.storage.keyword_merges import merged_once, planned, settled
from sift.storage.keyword_writes import Head, lengths_added, postings_added, segment_added

SHAPE_KEY, THROUGH_KEY, FINGERPRINT_KEY = "keyword_shape", "keywords_through", "keyword_fingerprint"
DOCUMENTS = "SELECT id, title, meta, text FROM keyword_documents WHERE id IN (SELECT value FROM json_each(?))"
NUMBERS_PROBE = """SELECT coalesce(group_concat(j.atom, ' '), '') FROM json_tree(
                       '[1e20, 0.1, 1.0, -0.0, 1.5e-7, 3.141592653589793, 12345678901234567890, true, false]') j
                   WHERE j.atom IS NOT NULL"""
REMAKE_REMEDY = "run sift index to remake it"
REMAKING = "remaking "
STALE = "the store's keyword index is of another version of sift"
UNFINISHED = "the store's keyword index is not yet wholly remade"
TAKEN_OVER = "another sift is remaking this keyword index for itself"
TAKEN_OVER_REMEDY = (
    "run sift index with one sift at a time: sifts whose SQLite writes numbers otherwise remake it in turn"
)
DIGITS = re.compile("[0-9]{1,19}")
MOST_ROW = (1 << 63) - 1


class Value(NamedTuple):
    value: str


class Span(NamedTuple):
    id: int
    document: int
    start: int
    length: int


class Held(NamedTuple):
    id: int
    title: str
    meta: str | None
    text: str


class Totals(NamedTuple):
    chunks: int
    tokens: int


class Range(NamedTuple):
    id: int
    first: int
    last: int


class Count(NamedTuple):
    n: int


def fingerprint(transaction: Reading) -> str:
    (rendered,) = rows(transaction, Value, NUMBERS_PROBE)
    return hashlib.blake2b(f"{FORMAT}|{TABLE_DIGEST}|{rendered.value}".encode(), digest_size=16).hexdigest()


def keyword_meta(transaction: Reading, shape: Shape, start: int) -> tuple[tuple[str, str], ...]:
    return (
        (SHAPE_KEY, json.dumps(dataclasses.asdict(shape))),
        (THROUGH_KEY, str(start)),
        (FINGERPRINT_KEY, fingerprint(transaction)),
    )


def _meta(transaction: Reading, key: str) -> str | None:
    found = rows(transaction, Value, "SELECT value FROM meta WHERE key = ?", (key,))
    return found[0].value if found else None


def _required(transaction: Reading, key: str) -> str:
    found = _meta(transaction, key)
    if found is None:
        raise DamagedError(f"the store's meta holds no {key}")
    return found


def current(transaction: Reading) -> bool:
    return _meta(transaction, FINGERPRINT_KEY) == fingerprint(transaction)


def _shape(written: str) -> Shape:
    try:
        return Shape(**cast("dict[str, int]", json.loads(written)))
    except (TypeError, ValueError) as error:
        raise DamagedError(f"the keyword shape {written!r} is none: {error}") from None


def shape_of(transaction: Reading) -> Shape:
    return _shape(_required(transaction, SHAPE_KEY))


def _number(transaction: Reading, key: str) -> int:
    written = _required(transaction, key)
    if DIGITS.fullmatch(written) is None or int(written) > MOST_ROW:
        raise DamagedError(f"the store's {key} is {written!r}, no number")
    return int(written)


def _through(transaction: Reading) -> int:
    return _number(transaction, THROUGH_KEY)


def totals(transaction: Reading) -> Totals:
    (found,) = rows(transaction, Totals, "SELECT chunks, tokens FROM counts")
    return found


def _tokens(streams: Sequence[Sequence[str]]) -> int:
    return sum(map(len, streams)) - 2 * len(streams)


def _streams(transaction: Reading, spans: Sequence[Span]) -> list[list[str]]:
    wanted = json.dumps(sorted({span.document for span in spans}))
    documents = {held.id: held for held in rows(transaction, Held, DOCUMENTS, (wanted,))}
    named = {key: (tokens(held.title), tokens(held.meta or "")) for key, held in documents.items()}
    return [
        stream(
            named[span.document][0],
            tokens(documents[span.document].text[span.start : span.start + span.length]),
            named[span.document][1],
        )
        for span in spans
    ]


def _added(transaction: Writing, written: Segment) -> None:
    head = Head(written.first, written.last, written.chunks, written.tokens, written.average)
    made = segment_added(transaction, head)
    postings_added(transaction, made, written.postings)
    lengths_added(transaction, made, written.lengths)


def _counted(transaction: Writing, added: Iterable[tuple[str, int]], tokens_added: int) -> None:
    run_each(
        transaction,
        """INSERT INTO keyword_terms(term, chunks) VALUES (?, ?)
           ON CONFLICT(term) DO UPDATE SET chunks = chunks + excluded.chunks""",
        added,
    )
    run(transaction, "UPDATE counts SET tokens = tokens + ?", (tokens_added,))


def _uncounted(transaction: Writing, gone: Counter[str], tokens_gone: int) -> None:
    run_each(transaction, "DELETE FROM keyword_terms WHERE term = ? AND chunks = ?", list(gone.items()))
    run_each(
        transaction,
        "UPDATE keyword_terms SET chunks = chunks - ? WHERE term = ?",
        [(n, term) for term, n in gone.items()],
    )
    run(transaction, "UPDATE counts SET tokens = tokens - ?", (tokens_gone,))


def _index_new(transaction: WritingDocuments, shape: Shape) -> int:
    spans = rows(
        transaction,
        Span,
        "SELECT id, document, start, length FROM chunks WHERE id > ? ORDER BY id LIMIT ?",
        (_through(transaction), transaction.batches.chunks_a_segment),
    )
    if not spans:
        return 0
    streams = _streams(transaction, spans)
    (indexed,) = rows(transaction, Count, "SELECT coalesce(sum(chunks - dead), 0) FROM keyword_segments")
    average = (totals(transaction).tokens + _tokens(streams)) / (indexed.n + len(spans))
    written = segment(np.array([span.id for span in spans], np.int64), streams, average, shape)
    _added(transaction, written)
    _counted(transaction, ((p.term, p.chunks) for p in written.postings), written.tokens)
    run(transaction, "UPDATE meta SET value = ? WHERE key = ?", (str(written.last), THROUGH_KEY))
    return len(spans)


def index_new(transaction: WritingDocuments) -> None:
    shape = shape_of(transaction)
    while _index_new(transaction, shape):
        pass
    settled(transaction, shape)


def _dead_by_segment(transaction: Writing, dead: Sequence[int]) -> list[tuple[int, int]]:
    held = rows(transaction, Range, "SELECT id, first, last FROM keyword_segments ORDER BY first")
    starts = np.array([segment.first for segment in held], np.int64)
    owners = ints(np.searchsorted(starts, np.array(dead, np.int64), side="right") - 1)
    if any(index < 0 or chunk > held[index].last for index, chunk in zip(owners, dead, strict=True)):
        raise DamagedError("the keyword index holds no segment of a chunk it indexed")
    return [(n, held[index].id) for index, n in Counter(owners).items()]


def _forget(transaction: Writing, documents: Sequence[DocumentId]) -> None:
    spans = rows(
        transaction,
        Span,
        """SELECT id, document, start, length FROM chunks
           WHERE document IN (SELECT value FROM json_each(?)) AND id <= ? ORDER BY id""",
        (json.dumps(list(documents)), _through(transaction)),
    )
    streams = _streams(transaction, spans)
    _uncounted(transaction, Counter(token for each in streams for token in set(each) - {GAP}), _tokens(streams))
    dead = [span.id for span in spans]
    run(transaction, "INSERT INTO keyword_dead(chunk) SELECT value FROM json_each(?)", (json.dumps(dead),))
    run_each(
        transaction, "UPDATE keyword_segments SET dead = dead + ? WHERE id = ?", _dead_by_segment(transaction, dead)
    )
    run(transaction, "DELETE FROM keyword_segments WHERE dead = chunks")


def forget(transaction: WritingDocuments, documents: Sequence[DocumentId]) -> None:
    size = transaction.batches.documents_a_forget
    for at in range(0, len(documents), size):
        _forget(transaction, documents[at : at + size])


def _anew(transaction: Writing, made: str) -> None:
    run(transaction, "DELETE FROM keyword_segments")
    run(transaction, "DELETE FROM keyword_terms")
    run(transaction, "DELETE FROM keyword_dead")
    run(transaction, "UPDATE counts SET tokens = 0")
    written = _meta(transaction, SHAPE_KEY)
    shape = json.dumps(dataclasses.asdict(SHAPE if written is None else _shape(written)))
    start = str(first_row(CollectionId(_number(transaction, "collection"))))
    run_each(
        transaction,
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(SHAPE_KEY, shape), (THROUGH_KEY, start), (FINGERPRINT_KEY, made)],
    )


def _left(transaction: Reading) -> int:
    (found,) = rows(transaction, Count, "SELECT count(*) FROM chunks WHERE id > ?", (_through(transaction),))
    return found.n


def _done_if_whole(transaction: Writing, made: str) -> bool:
    if _left(transaction):
        return False
    run(transaction, "UPDATE meta SET value = ? WHERE key = ?", (made, FINGERPRINT_KEY))
    return True


def remake_started(connection: Connection) -> int:
    with reading(connection) as transaction:
        if current(transaction):
            return 0
    with writing(connection) as transaction:
        made = fingerprint(transaction)
        found = _meta(transaction, FINGERPRINT_KEY)
        if found not in {made, REMAKING + made}:
            _anew(transaction, REMAKING + made)
        return 0 if found == made or _done_if_whole(transaction, made) else _left(transaction)


def remade(connection: Connection, batches: Batches = BATCHES) -> Iterator[int]:
    done = 0
    whole = False
    while not whole:
        with writing(connection) as transaction:
            made, found = fingerprint(transaction), _meta(transaction, FINGERPRINT_KEY)
            if found == made:
                return
            if found != REMAKING + made:
                raise StoreUnavailableError(TAKEN_OVER, TAKEN_OVER_REMEDY)
            held, shape = WritingDocuments(connection, batches), shape_of(transaction)
            done += _index_new(held, shape)
            settled(held, shape)
            whole = _done_if_whole(transaction, made)
        yield done


def refuse_stale(transaction: Reading) -> None:
    made = fingerprint(transaction)
    found = _meta(transaction, FINGERPRINT_KEY)
    if found == REMAKING + made:
        raise StoreUnavailableError(UNFINISHED, REMAKE_REMEDY)
    if found is not None and found.startswith(REMAKING):
        raise StoreUnavailableError(TAKEN_OVER, TAKEN_OVER_REMEDY)
    if found != made:
        raise StoreUnavailableError(STALE, REMAKE_REMEDY)


def settle(connection: Connection, batches: Batches = BATCHES) -> Iterator[int]:
    with reading(connection) as transaction:
        refuse_stale(transaction)
        if not planned(transaction, batches.merging):
            return
    done = 0
    yield done
    while True:
        with writing(connection) as transaction:
            refuse_stale(transaction)
            written = merged_once(WritingDocuments(connection, batches), shape_of(transaction))
        if not written:
            return
        done += written
        yield done


@contextmanager
def writing_documents(connection: Connection, batches: Batches = BATCHES) -> Generator[WritingDocuments]:
    with writing(connection) as transaction:
        refuse_stale(transaction)
        held = WritingDocuments(connection, batches)
        yield held
        index_new(held)
