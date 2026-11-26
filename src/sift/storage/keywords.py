import dataclasses
import json
import re
from collections import Counter
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import NamedTuple, cast

import numpy as np

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import CollectionId, DocumentId, first_row
from sift.domain.arrays import ints
from sift.domain.codec.segments import GAP, Segment, segment, stream
from sift.domain.codec.shape import SHAPE, Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.keyword_state import KeywordState, keyword_state, remaking
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS, field_tokens, tokens
from sift.storage.database import (
    BATCHES,
    Batches,
    Connection,
    Count,
    Reading,
    Value,
    Writing,
    WritingDocuments,
    reading,
    rows,
    run,
    run_each,
    writing,
)
from sift.storage.keyword_merges import Totals, merged_once, planned, settled
from sift.storage.keyword_meta import FINGERPRINT_KEY, SHAPE_KEY, THROUGH_KEY, drop_beside, fingerprint
from sift.storage.keyword_writes import Head, lengths_added, postings_added, segment_added
from sift.storage.texts import TextSpan, spans_text

DOCUMENTS = """SELECT id, substr(title, 1, ?) AS title, substr(meta, 1, ?) AS meta FROM keyword_documents
               WHERE id IN (SELECT value FROM json_each(?))"""
REMAKE_REMEDY = "index the collection to remake it"
STALE = "the store's keyword index is of another version of sift"
UNFINISHED = "the store's keyword index is not yet wholly remade"
TAKEN_OVER = "another sift began remaking this keyword index for itself and has not finished"
TAKEN_OVER_REMEDY = "index it from one sift only: sifts whose SQLite renders numbers apart remake it in turn"
DIGITS = re.compile("[0-9]{1,19}")
MOST_ROW = (1 << 63) - 1
TEXTS_AT_ONCE = 1000


class Span(NamedTuple):
    id: int
    document: DocumentId
    start: int
    length: int


class Named(NamedTuple):
    id: int
    title: str
    meta: str | None


class Range(NamedTuple):
    id: int
    first: int
    last: int


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


def _streams(transaction: Reading, spans: Sequence[Span]) -> Iterator[list[str]]:
    wanted = json.dumps(sorted({span.document for span in spans}))
    cut = MOST_FIELD_CHARACTERS + 1
    documents = {held.id: held for held in rows(transaction, Named, DOCUMENTS, (cut, cut, wanted))}
    named = {key: (field_tokens(held.title), field_tokens(held.meta or "")) for key, held in documents.items()}
    seen: dict[str, str] = {}
    for at in range(0, len(spans), TEXTS_AT_ONCE):
        part = spans[at : at + TEXTS_AT_ONCE]
        cut = spans_text(transaction, [TextSpan(span.document, span.start, span.length) for span in part])
        for span, text in zip(part, cut, strict=True):
            title, meta = named[span.document]
            yield stream(title, [seen.setdefault(each, each) for each in tokens(text)], meta)


def _up_to(streams: Iterator[list[str]], most: int) -> Iterator[list[str]]:
    held = 0
    for each in streams:
        yield each
        held += len(each)
        if held >= most:
            return


def streams_of(transaction: WritingDocuments, spans: Sequence[Span]) -> list[list[str]]:
    return list(_up_to(_streams(transaction, spans), transaction.batches.tokens_a_segment))


def segment_written(
    transaction: Writing, spans: Sequence[Span], streams: Sequence[Sequence[str]], average: float, shape: Shape
) -> Segment:
    written = segment(np.array([span.id for span in spans], np.int64), streams, average, shape)
    made = segment_added(transaction, Head(written.first, written.last, written.chunks, written.average))
    postings_added(transaction, made, written.postings)
    lengths_added(transaction, made, written.lengths)
    return written


def _terms_added(transaction: Writing, added: Iterable[tuple[str, int]], tokens_added: int) -> None:
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
        "SELECT place AS id, document, start, length FROM chunks WHERE place > ? ORDER BY place LIMIT ?",
        (_through(transaction), transaction.batches.chunks_a_segment),
    )
    if not spans:
        return 0
    streams = streams_of(transaction, spans)
    spans = spans[: len(streams)]
    (indexed,) = rows(transaction, Count, "SELECT coalesce(sum(chunks - dead), 0) FROM keyword_segments")
    average = (totals(transaction).tokens + _tokens(streams)) / (indexed.n + len(spans))
    written = segment_written(transaction, spans, streams, average, shape)
    _terms_added(transaction, ((p.term, p.chunks) for p in written.postings), written.tokens)
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
        """SELECT place AS id, document, start, length FROM chunks
           WHERE document IN (SELECT value FROM json_each(?)) AND place <= ? ORDER BY place""",
        (json.dumps(list(documents)), _through(transaction)),
    )
    gone: Counter[str] = Counter()
    held = 0
    for each in _streams(transaction, spans):
        gone.update(set(each) - {GAP})
        held += _tokens((each,))
    _uncounted(transaction, gone, held)
    dead = [span.id for span in spans]
    run(transaction, "INSERT INTO keyword_dead(place) SELECT value FROM json_each(?)", (json.dumps(dead),))
    run_each(
        transaction, "UPDATE keyword_segments SET dead = dead + ? WHERE id = ?", _dead_by_segment(transaction, dead)
    )
    run(transaction, "DELETE FROM keyword_segments WHERE dead = chunks")


def forget(transaction: WritingDocuments, documents: Sequence[DocumentId]) -> None:
    size = transaction.batches.documents_a_forget
    for at in range(0, len(documents), size):
        _forget(transaction, documents[at : at + size])


def _anew(transaction: Writing, made: str, start: int) -> None:
    drop_beside(transaction)
    run(transaction, "DELETE FROM keyword_segments")
    run(transaction, "DELETE FROM keyword_terms")
    run(transaction, "DELETE FROM keyword_dead")
    run(transaction, "UPDATE counts SET tokens = 0")
    written = _meta(transaction, SHAPE_KEY)
    shape = json.dumps(dataclasses.asdict(SHAPE if written is None else _shape(written)))
    run_each(
        transaction,
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(SHAPE_KEY, shape), (THROUGH_KEY, str(start)), (FINGERPRINT_KEY, made)],
    )


def remake_above(transaction: Writing, place: int) -> None:
    _anew(transaction, remaking(fingerprint(transaction)), place)


def _left(transaction: Reading) -> int:
    (found,) = rows(transaction, Count, "SELECT count(*) FROM chunks WHERE place > ?", (_through(transaction),))
    return found.n


def _any_left(transaction: Reading) -> bool:
    sql = "SELECT EXISTS(SELECT 1 FROM chunks WHERE place > ?)"
    (found,) = rows(transaction, Count, sql, (_through(transaction),))
    return bool(found.n)


def _done_if_whole(transaction: Writing, made: str) -> bool:
    if _any_left(transaction):
        return False
    run(transaction, "UPDATE meta SET value = ? WHERE key = ?", (made, FINGERPRINT_KEY))
    return True


def remake_started(connection: Connection) -> int:
    with reading(connection) as transaction:
        if current(transaction):
            return 0
    with writing(connection) as transaction:
        made = fingerprint(transaction)
        state = keyword_state(_meta(transaction, FINGERPRINT_KEY), made)
        if state not in (KeywordState.CURRENT, KeywordState.REMAKING):
            _anew(transaction, remaking(made), first_row(CollectionId(_number(transaction, "collection"))))
        return 0 if state is KeywordState.CURRENT or _done_if_whole(transaction, made) else _left(transaction)


@contextmanager
def writing_keywords(connection: Connection, batches: Batches = BATCHES) -> Generator[WritingDocuments]:
    with writing(connection):
        yield WritingDocuments(connection, batches)


def remade(connection: Connection, batches: Batches = BATCHES) -> Iterator[int]:
    done = 0
    whole = False
    while not whole:
        with writing_keywords(connection, batches) as transaction:
            made = fingerprint(transaction)
            state = keyword_state(_meta(transaction, FINGERPRINT_KEY), made)
            if state is KeywordState.CURRENT:
                return
            if state is not KeywordState.REMAKING:
                raise StoreUnavailableError(TAKEN_OVER, TAKEN_OVER_REMEDY, code=Problem.KEYWORDS_REMADE)
            shape = shape_of(transaction)
            done += _index_new(transaction, shape)
            settled(transaction, shape)
            whole = _done_if_whole(transaction, made)
        yield done


def refuse_stale(transaction: Reading) -> None:
    match keyword_state(_meta(transaction, FINGERPRINT_KEY), fingerprint(transaction)):
        case KeywordState.CURRENT:
            return
        case KeywordState.REMAKING:
            raise StoreUnavailableError(UNFINISHED, REMAKE_REMEDY, code=Problem.KEYWORDS_REMADE)
        case KeywordState.TAKEN_OVER:
            raise StoreUnavailableError(TAKEN_OVER, TAKEN_OVER_REMEDY, code=Problem.KEYWORDS_REMADE)
        case _:
            raise StoreUnavailableError(STALE, REMAKE_REMEDY, code=Problem.KEYWORDS_REMADE)


def settle(connection: Connection, batches: Batches = BATCHES) -> Iterator[int]:
    with reading(connection) as transaction:
        refuse_stale(transaction)
        if not planned(transaction, batches.merging):
            return
    done = 0
    yield done
    while True:
        with writing_keywords(connection, batches) as transaction:
            refuse_stale(transaction)
            written = merged_once(transaction, shape_of(transaction))
        if not written:
            return
        done += written
        yield done


@contextmanager
def writing_documents(connection: Connection, batches: Batches = BATCHES) -> Generator[WritingDocuments]:
    with writing_keywords(connection, batches) as transaction:
        refuse_stale(transaction)
        drop_beside(transaction)
        yield transaction
        index_new(transaction)
