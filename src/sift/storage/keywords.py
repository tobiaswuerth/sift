import dataclasses
import json
import re
from collections import Counter
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import NamedTuple, cast

import numpy as np

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import CollectionId, DocumentId, first_row
from sift.domain.arrays import ints
from sift.domain.codec.segments import GAP, Building, WrittenSegment, building, gathered, stream, written
from sift.domain.codec.shape import FORMAT, SHAPE, Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.keyword_state import KeywordState, keyword_state, remaking
from sift.domain.text.tokens import field_tokens, tokens
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
from sift.storage.keyword_meta import (
    FINGERPRINT_KEY,
    FORMAT_KEY,
    SHAPE_KEY,
    THROUGH_KEY,
    drop_beside,
    fingerprint,
)
from sift.storage.keyword_reads import Passages, ReadParts, Span, read_ahead, read_in
from sift.storage.keyword_writes import LENGTHS_ROW, Head, rows_added, segment_added

REMAKE_REMEDY = "index the collection to remake it"
STALE = "the store's keyword index is of another version of sift"
UNFINISHED = "the store's keyword index is not yet wholly remade"
TAKEN_OVER = "another sift began remaking this keyword index for itself and has not finished"
TAKEN_OVER_REMEDY = "index it from one sift only: sifts whose SQLite renders numbers apart remake it in turn"
NEWER = "the store's keyword index was made by a newer sift"
NEWER_REMEDY = "upgrade sift"
GONE = """INSERT INTO keyword_gone(term, segment, chunks) VALUES (?, ?, ?)
          ON CONFLICT(term, segment) DO UPDATE SET chunks = chunks + excluded.chunks"""
DIGITS = re.compile("[0-9]{1,19}")
MOST_ROW = (1 << 63) - 1


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


def _streams(transaction: WritingDocuments, spans: Sequence[Span]) -> Iterator[list[str]]:
    seen: dict[str, str] = {}
    with read_in(transaction, spans) as reads:
        for read in reads:
            named = {held.id: (field_tokens(held.title), field_tokens(held.meta or "")) for held in read.documents}
            for span, text in zip(read.spans, read.texts, strict=True):
                title, meta = named[span.document]
                yield stream(title, [seen.setdefault(each, each) for each in tokens(text)], meta)


def _taken(built: Building, read: Passages) -> int:
    at = {held.id: number for number, held in enumerate(read.documents)}
    fields = [(held.title, held.meta or "") for held in read.documents]
    places = np.array([span.id for span in read.spans], np.int64)
    return built.add(places, [at[span.document] for span in read.spans], fields, read.texts)


def built_of(transaction: WritingDocuments, spans: Sequence[Span], reads: ReadParts) -> Building:
    built = building(transaction.batches.tokens_a_segment)
    with reads(transaction, spans) as each:
        for read in each:
            if _taken(built, read) < len(read.spans):
                break
    return built


def segment_written(transaction: Writing, built: Building, average: float, shape: Shape) -> WrittenSegment:
    held = gathered(built)
    made = written(built, segment_added(transaction, Head(held.first, held.last, held.chunks, average)), average, shape)
    rows_added(transaction, made.filed)
    run_each(transaction, LENGTHS_ROW, made.lengths)
    return made


def _index_new(transaction: WritingDocuments, shape: Shape, reads: ReadParts) -> int:
    spans = rows(
        transaction,
        Span,
        "SELECT place AS id, document, start, length FROM chunks WHERE place > ? ORDER BY place LIMIT ?",
        (_through(transaction), transaction.batches.chunks_a_segment),
    )
    if not spans:
        return 0
    built = built_of(transaction, spans, reads)
    (indexed,) = rows(transaction, Count, "SELECT coalesce(sum(chunks - dead), 0) FROM keyword_segments")
    held = gathered(built)
    average = (totals(transaction).tokens + held.tokens) / (indexed.n + held.chunks)
    made = segment_written(transaction, built, average, shape)
    run(transaction, "UPDATE counts SET tokens = tokens + ?", (made.tokens,))
    run(transaction, "UPDATE meta SET value = ? WHERE key = ?", (str(made.last), THROUGH_KEY))
    return made.chunks


def index_new(transaction: WritingDocuments) -> None:
    shape = shape_of(transaction)
    while _index_new(transaction, shape, read_in):
        pass
    settled(transaction, shape)


def _owners(transaction: Writing, dead: Sequence[int]) -> list[int]:
    held = rows(transaction, Range, "SELECT id, first, last FROM keyword_segments ORDER BY first")
    starts = np.array([segment.first for segment in held], np.int64)
    owners = ints(np.searchsorted(starts, np.array(dead, np.int64), side="right") - 1)
    if any(index < 0 or chunk > held[index].last for index, chunk in zip(owners, dead, strict=True)):
        raise DamagedError("the keyword index holds no segment of a chunk it indexed")
    return [held[index].id for index in owners]


def _forget(transaction: WritingDocuments, documents: Sequence[DocumentId]) -> None:
    spans = rows(
        transaction,
        Span,
        """SELECT place AS id, document, start, length FROM chunks
           WHERE document IN (SELECT value FROM json_each(?)) AND place <= ? ORDER BY place""",
        (json.dumps(list(documents)), _through(transaction)),
    )
    dead = [span.id for span in spans]
    owners = _owners(transaction, dead)
    gone: Counter[tuple[str, int]] = Counter()
    held = 0
    for owner, each in zip(owners, _streams(transaction, spans), strict=True):
        gone.update((term, owner) for term in set(each) - {GAP})
        held += _tokens((each,))
    run_each(transaction, GONE, [(term, owner, chunks) for (term, owner), chunks in gone.items()])
    run(transaction, "UPDATE counts SET tokens = tokens - ?", (held,))
    run(transaction, "INSERT INTO keyword_dead(place) SELECT value FROM json_each(?)", (json.dumps(dead),))
    emptied = [(chunks, owner) for owner, chunks in Counter(owners).items()]
    run_each(transaction, "UPDATE keyword_segments SET dead = dead + ? WHERE id = ?", emptied)
    run(transaction, "DELETE FROM keyword_segments WHERE dead = chunks")


def forget(transaction: WritingDocuments, documents: Sequence[DocumentId]) -> None:
    size = transaction.batches.documents_a_forget
    for at in range(0, len(documents), size):
        _forget(transaction, documents[at : at + size])


def refuse_newer(transaction: Reading) -> None:
    if _number(transaction, FORMAT_KEY) > FORMAT:
        raise StoreUnavailableError(NEWER, NEWER_REMEDY, code=Problem.STORE_NEWER)


def _anew(transaction: Writing, made: str, start: int) -> None:
    refuse_newer(transaction)
    drop_beside(transaction)
    run(transaction, "DELETE FROM keyword_segments")
    run(transaction, "DELETE FROM keyword_dead")
    run(transaction, "UPDATE counts SET tokens = 0")
    written = _meta(transaction, SHAPE_KEY)
    shape = json.dumps(dataclasses.asdict(SHAPE if written is None else _shape(written)))
    run_each(
        transaction,
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(SHAPE_KEY, shape), (THROUGH_KEY, str(start)), (FINGERPRINT_KEY, made), (FORMAT_KEY, str(FORMAT))],
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
            done += _index_new(transaction, shape, read_ahead)
            settled(transaction, shape)
            whole = _done_if_whole(transaction, made)
        yield done


def refuse_stale(transaction: Reading) -> None:
    refuse_newer(transaction)
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
