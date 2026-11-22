import dataclasses
import json
from collections import Counter
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import NamedTuple, cast

import numpy as np

from sift.api.ids import DocumentId
from sift.domain.codec.segments import GAP, Long, Segment, Shape, segment, stream
from sift.domain.text.tokens import tokens
from sift.storage.database import Connection, Reading, Writing, rows, run, run_each, writing

SHAPE_KEY, THROUGH_KEY = "keyword_shape", "keywords_through"
DOCUMENTS = "SELECT id, title, meta, text FROM keyword_documents WHERE id IN (SELECT value FROM json_each(?))"
SEGMENTS_HELD = """UPDATE keyword_segments
                   SET dead = dead + (SELECT count(*) FROM json_each(?) WHERE value BETWEEN first AND last)
                   WHERE id IN (SELECT s.id FROM keyword_segments s
                                JOIN json_each(?) j ON j.value BETWEEN s.first AND s.last)"""


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


def keyword_meta(shape: Shape, start: int) -> tuple[tuple[str, str], ...]:
    return (SHAPE_KEY, json.dumps(dataclasses.asdict(shape))), (THROUGH_KEY, str(start))


def _meta(transaction: Reading, key: str) -> str:
    (found,) = rows(transaction, Value, "SELECT value FROM meta WHERE key = ?", (key,))
    return found.value


def shape_of(transaction: Reading) -> Shape:
    return Shape(**cast("dict[str, int]", json.loads(_meta(transaction, SHAPE_KEY))))


def through(transaction: Reading) -> int:
    return int(_meta(transaction, THROUGH_KEY))


def totals(transaction: Reading) -> Totals:
    (found,) = rows(transaction, Totals, "SELECT chunks, tokens FROM counts")
    return found


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
    run_each(
        transaction,
        "DELETE FROM keyword_terms WHERE term = ? AND chunks = ?",
        [(term, n) for term, n in gone.items()],
    )
    run_each(
        transaction,
        "UPDATE keyword_terms SET chunks = chunks - ? WHERE term = ?",
        [(n, term) for term, n in gone.items()],
    )
    run(transaction, "UPDATE counts SET tokens = tokens - ?", (tokens_gone,))


def index_new(transaction: Writing) -> None:
    spans = rows(
        transaction,
        Span,
        "SELECT id, document, start, length FROM chunks WHERE id > ? ORDER BY id",
        (through(transaction),),
    )
    if not spans:
        return
    streams = _streams(transaction, spans)
    held = totals(transaction)
    added = sum(len(each) for each in streams) - 2 * len(streams)
    average = (held.tokens + added) / held.chunks
    written = segment(np.array([span.id for span in spans], np.int64), streams, average, shape_of(transaction))
    _added(transaction, written)
    _counted(transaction, ((p.term, p.chunks) for p in written.postings), written.tokens)
    run(transaction, "UPDATE meta SET value = ? WHERE key = ?", (str(written.last), THROUGH_KEY))


def forget(transaction: Writing, documents: Sequence[DocumentId]) -> None:
    spans = rows(
        transaction,
        Span,
        """SELECT id, document, start, length FROM chunks
           WHERE document IN (SELECT value FROM json_each(?)) AND id <= ? ORDER BY id""",
        (json.dumps(list(documents)), through(transaction)),
    )
    if not spans:
        return
    streams = _streams(transaction, spans)
    held = Counter(token for each in streams for token in set(each) - {GAP})
    _uncounted(transaction, held, sum(map(len, streams)) - 2 * len(streams))
    dead = json.dumps([span.id for span in spans])
    run(transaction, "INSERT INTO keyword_dead(chunk) SELECT value FROM json_each(?)", (dead,))
    run(transaction, SEGMENTS_HELD, (dead, dead))
    run(transaction, "DELETE FROM keyword_segments WHERE dead = chunks")


@contextmanager
def writing_documents(connection: Connection) -> Generator[Writing]:
    with writing(connection) as transaction:
        yield transaction
        index_new(transaction)
