from collections.abc import Iterable, Iterator

from sift.domain.codec.shape import Shape
from sift.storage.database import (
    BATCHES,
    Batches,
    Connection,
    Count,
    Writing,
    WritingDocuments,
    reading,
    rows,
    run,
    run_each,
)
from sift.storage.keyword_merges import settled
from sift.storage.keyword_meta import Side
from sift.storage.keywords import (
    Span,
    built_of,
    current,
    segment_written,
    shape_of,
    totals,
    writing_keywords,
)

BEGUN = "INSERT INTO keyword_beside(id, above, through, made_at) VALUES (1, ?, ?, ?)"
SPANS = """SELECT placed.place AS id, c.document, c.start, c.length
           FROM keyword_places AS placed JOIN chunks c ON c.id = placed.chunk
           WHERE placed.place > (SELECT through FROM keyword_beside) ORDER BY placed.place LIMIT ?"""
LEFT = "SELECT count(*) FROM keyword_places WHERE place > (SELECT through FROM keyword_beside)"
ANY_LEFT = "SELECT EXISTS(SELECT 1 FROM keyword_places WHERE place > (SELECT through FROM keyword_beside))"
SWITCHED = (
    "DELETE FROM keyword_segments WHERE id IN (SELECT id FROM keyword_served)",
    "UPDATE chunks SET place = placed.place FROM keyword_places AS placed WHERE placed.chunk = chunks.id",
    """UPDATE meta SET value = (SELECT CAST(through AS TEXT) FROM keyword_beside)
       WHERE key = 'keywords_through'""",
    "UPDATE counts SET ordered_at = (SELECT made_at FROM keyword_beside)",
    "DELETE FROM keyword_places",
    "DELETE FROM keyword_beside",
)


def begin(transaction: Writing, above: int, placed: Iterable[tuple[int, int]], made_at: int) -> None:
    run(transaction, BEGUN, (above, above, made_at))
    run_each(transaction, "INSERT INTO keyword_places(place, chunk) VALUES (?, ?)", placed)


def left(connection: Connection) -> int:
    with reading(connection) as transaction:
        if not current(transaction):
            return 0
        (found,) = rows(transaction, Count, LEFT)
        return found.n


def _indexed(transaction: WritingDocuments, shape: Shape) -> int:
    spans = rows(transaction, Span, SPANS, (transaction.batches.chunks_a_segment,))
    if not spans:
        return 0
    held = totals(transaction)
    written = segment_written(transaction, built_of(transaction, spans), held.tokens / held.chunks, shape)
    run(transaction, "UPDATE keyword_beside SET through = ?", (written.last,))
    return written.chunks


def _switched_once_whole(transaction: Writing) -> None:
    (waiting,) = rows(transaction, Count, ANY_LEFT)
    if not waiting.n:
        for sql in SWITCHED:
            run(transaction, sql)


def _step(connection: Connection, batches: Batches) -> int:
    with writing_keywords(connection, batches) as transaction:
        if not current(transaction):
            return 0
        shape = shape_of(transaction)
        added = _indexed(transaction, shape)
        if added:
            settled(transaction, shape, Side.BUILT)
            _switched_once_whole(transaction)
        return added


def built(connection: Connection, batches: Batches = BATCHES) -> Iterator[int]:
    done = 0
    while added := _step(connection, batches):
        done += added
        yield done
