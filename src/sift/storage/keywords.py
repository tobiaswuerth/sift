import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import NamedTuple, cast

import numpy as np

from sift.api.ids import CollectionId, DocumentId, first_row
from sift.domain.arrays import ints
from sift.domain.codec.segments import FORMAT, GAP, SHAPE, Long, Segment, Shape, segment, stream
from sift.domain.text.tokens import TABLE_DIGEST, tokens
from sift.storage.database import (
    BATCHES,
    Batches,
    Connection,
    Reading,
    Writing,
    WritingDocuments,
    rows,
    run,
    run_each,
    writing,
)

SHAPE_KEY, THROUGH_KEY, FINGERPRINT_KEY = "keyword_shape", "keywords_through", "keyword_fingerprint"
DOCUMENTS = "SELECT id, title, meta, text FROM keyword_documents WHERE id IN (SELECT value FROM json_each(?))"
NUMBERS_PROBE = """SELECT coalesce(group_concat(j.atom, ' '), '') FROM json_tree(
                       '[1e20, 0.1, 1.0, -0.0, 1.5e-7, 3.141592653589793, 12345678901234567890, true, false]') j
                   WHERE j.atom IS NOT NULL"""


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


class Identity(NamedTuple):
    id: int


class Totals(NamedTuple):
    chunks: int
    tokens: int


class Range(NamedTuple):
    id: int
    first: int


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


def current(transaction: Reading) -> bool:
    return _meta(transaction, FINGERPRINT_KEY) == fingerprint(transaction)


def shape_of(transaction: Reading) -> Shape:
    written = _meta(transaction, SHAPE_KEY)
    return SHAPE if written is None else Shape(**cast("dict[str, int]", json.loads(written)))


def _through(transaction: Reading) -> int:
    return int(_meta(transaction, THROUGH_KEY) or 0)


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
    (made,) = rows(
        transaction,
        Identity,
        """INSERT INTO keyword_segments(first, last, chunks, dead, tokens, average)
           VALUES (?, ?, ?, 0, ?, ?) RETURNING id""",
        (written.first, written.last, written.chunks, written.tokens, written.average),
    )
    identity = made.id
    run_each(
        transaction,
        "INSERT INTO keyword_postings(segment, term, chunks, postings, positions, ranges) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (identity, p.term, p.chunks, None, None, p.held.ranges)
            if isinstance(p.held, Long)
            else (identity, p.term, p.chunks, p.held.postings, p.held.positions, None)
            for p in written.postings
        ],
    )
    run_each(
        transaction,
        "INSERT INTO keyword_blocks(segment, term, number, postings, positions) VALUES (?, ?, ?, ?, ?)",
        [
            (identity, p.term, number, part.postings, part.positions)
            for p in written.postings
            if isinstance(p.held, Long)
            for number, part in enumerate(p.held.blocks)
        ],
    )
    run_each(
        transaction,
        "INSERT INTO keyword_lengths(segment, block, lengths) VALUES (?, ?, ?)",
        [(identity, block, data) for block, data in written.lengths.items()],
    )


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


def _index_new(transaction: WritingDocuments, shape: Shape) -> bool:
    spans = rows(
        transaction,
        Span,
        "SELECT id, document, start, length FROM chunks WHERE id > ? ORDER BY id LIMIT ?",
        (_through(transaction), transaction.batches.chunks_a_segment),
    )
    if not spans:
        return False
    streams = _streams(transaction, spans)
    held = totals(transaction)
    average = (held.tokens + _tokens(streams)) / held.chunks
    written = segment(np.array([span.id for span in spans], np.int64), streams, average, shape)
    _added(transaction, written)
    _counted(transaction, ((p.term, p.chunks) for p in written.postings), written.tokens)
    run(transaction, "UPDATE meta SET value = ? WHERE key = ?", (str(written.last), THROUGH_KEY))
    return True


def index_new(transaction: WritingDocuments) -> None:
    shape = shape_of(transaction)
    while _index_new(transaction, shape):
        pass


def _dead_by_segment(transaction: Writing, dead: Sequence[int]) -> list[tuple[int, int]]:
    held = rows(transaction, Range, "SELECT id, first FROM keyword_segments ORDER BY first")
    starts = np.array([segment.first for segment in held], np.int64)
    owners = np.searchsorted(starts, np.array(dead, np.int64), side="right") - 1
    return [(n, held[index].id) for index, n in Counter(ints(owners)).items()]


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


def _collection(transaction: Reading) -> CollectionId:
    return CollectionId(int(_meta(transaction, "collection") or 0))


def _anew(transaction: Writing, made: str) -> None:
    run(transaction, "DELETE FROM keyword_segments")
    run(transaction, "DELETE FROM keyword_terms")
    run(transaction, "DELETE FROM keyword_dead")
    run(transaction, "UPDATE counts SET tokens = 0")
    shape = json.dumps(dataclasses.asdict(shape_of(transaction)))
    start = str(first_row(_collection(transaction)))
    run_each(
        transaction,
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(SHAPE_KEY, shape), (THROUGH_KEY, start), (FINGERPRINT_KEY, made)],
    )


@contextmanager
def writing_documents(connection: Connection, batches: Batches = BATCHES) -> Generator[WritingDocuments]:
    with writing(connection) as transaction:
        if not current(transaction):
            _anew(transaction, fingerprint(transaction))
        held = WritingDocuments(connection, batches)
        yield held
        index_new(held)
