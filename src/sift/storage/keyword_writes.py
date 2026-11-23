from collections.abc import Mapping, Sequence
from typing import NamedTuple

from sift.domain.codec.segments import LongPostings, Postings
from sift.storage.database import IdRow, Writing, rows, run_each


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


def postings_added(transaction: Writing, segment: int, postings: Sequence[Postings]) -> None:
    run_each(
        transaction,
        "INSERT INTO keyword_postings(segment, term, chunks, postings, positions, ranges) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (segment, p.term, p.chunks, None, None, p.held.ranges)
            if isinstance(p.held, LongPostings)
            else (segment, p.term, p.chunks, p.held.postings, p.held.positions, None)
            for p in postings
        ],
    )
    run_each(
        transaction,
        "INSERT INTO keyword_blocks(segment, term, number, postings, positions) VALUES (?, ?, ?, ?, ?)",
        [
            (segment, p.term, number, part.postings, part.positions)
            for p in postings
            if isinstance(p.held, LongPostings)
            for number, part in enumerate(p.held.blocks)
        ],
    )


def lengths_added(transaction: Writing, segment: int, lengths: Mapping[int, bytes]) -> None:
    run_each(
        transaction,
        "INSERT INTO keyword_lengths(segment, block, lengths) VALUES (?, ?, ?)",
        [(segment, block, data) for block, data in lengths.items()],
    )
