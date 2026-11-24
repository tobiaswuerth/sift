from collections.abc import Mapping, Sequence
from typing import NamedTuple

from sift.domain.codec.segments import LongPostings, Postings, ShortPostings
from sift.storage.database import IdRow, Writing, rows, run, run_each


class Head(NamedTuple):
    first: int
    last: int
    chunks: int
    average: float


def segment_added(transaction: Writing, head: Head) -> int:
    (made,) = rows(
        transaction,
        IdRow,
        """INSERT INTO keyword_segments(first, last, chunks, dead, average)
           VALUES (?, ?, ?, 0, ?) RETURNING id""",
        head,
    )
    return made.id


POSTINGS_ROW = (
    "INSERT INTO keyword_postings(segment, term, chunks, postings, positions, ranges) VALUES (?, ?, ?, ?, ?, ?)"
)
BLOCK_ROW = "INSERT INTO keyword_blocks(segment, term, number, postings, positions) VALUES (?, ?, ?, ?, ?)"
UNFINISHED_RANGES = b""


def postings_added(transaction: Writing, segment: int, postings: Sequence[Postings]) -> None:
    run_each(
        transaction,
        POSTINGS_ROW,
        [
            (segment, p.term, p.chunks, None, None, p.held.ranges)
            if isinstance(p.held, LongPostings)
            else (segment, p.term, p.chunks, p.held.postings, p.held.positions, None)
            for p in postings
        ],
    )
    run_each(
        transaction,
        BLOCK_ROW,
        [
            (segment, p.term, number, part.postings, part.positions)
            for p in postings
            if isinstance(p.held, LongPostings)
            for number, part in enumerate(p.held.blocks)
        ],
    )


def long_begun(transaction: Writing, segment: int, term: str) -> None:
    run(transaction, POSTINGS_ROW, (segment, term, 1, None, None, UNFINISHED_RANGES))


def blocks_added(transaction: Writing, segment: int, term: str, first: int, blocks: Sequence[ShortPostings]) -> None:
    run_each(
        transaction,
        BLOCK_ROW,
        [(segment, term, first + number, part.postings, part.positions) for number, part in enumerate(blocks)],
    )


def long_finished(transaction: Writing, segment: int, term: str, chunks: int, ranges: bytes) -> None:
    sql = "UPDATE keyword_postings SET chunks = ?, ranges = ? WHERE segment = ? AND term = ?"
    run(transaction, sql, (chunks, ranges, segment, term))


def lengths_added(transaction: Writing, segment: int, lengths: Mapping[int, bytes]) -> None:
    run_each(
        transaction,
        "INSERT INTO keyword_lengths(segment, block, lengths) VALUES (?, ?, ?)",
        [(segment, block, data) for block, data in lengths.items()],
    )
