from collections.abc import Mapping, Sequence
from typing import NamedTuple

from sift.domain.codec.segments import Long, Postings
from sift.storage.database import Writing, rows, run_each


class Identity(NamedTuple):
    id: int


class Head(NamedTuple):
    first: int
    last: int
    chunks: int
    tokens: int
    average: float


def segment_added(transaction: Writing, head: Head) -> int:
    (made,) = rows(
        transaction,
        Identity,
        """INSERT INTO keyword_segments(first, last, chunks, dead, tokens, average)
           VALUES (?, ?, ?, 0, ?, ?) RETURNING id""",
        head,
    )
    return made.id


def postings_added(transaction: Writing, segment: int, postings: Sequence[Postings]) -> None:
    run_each(
        transaction,
        "INSERT INTO keyword_postings(segment, term, chunks, postings, positions, ranges) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (segment, p.term, p.chunks, None, None, p.held.ranges)
            if isinstance(p.held, Long)
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
            if isinstance(p.held, Long)
            for number, part in enumerate(p.held.blocks)
        ],
    )


def lengths_added(transaction: Writing, segment: int, lengths: Mapping[int, bytes]) -> None:
    run_each(
        transaction,
        "INSERT INTO keyword_lengths(segment, block, lengths) VALUES (?, ?, ?)",
        [(segment, block, data) for block, data in lengths.items()],
    )
