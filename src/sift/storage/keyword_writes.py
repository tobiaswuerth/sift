from collections.abc import Mapping, Sequence
from typing import NamedTuple

from sift.domain.codec.segments import ListBlockRow, ListsRow
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


POSTINGS_ROW = (
    "INSERT INTO keyword_postings(segment, term, chunks, postings, positions, ranges) VALUES (?, ?, ?, ?, ?, ?)"
)
BLOCK_ROW = "INSERT INTO keyword_blocks(segment, term, number, postings, positions) VALUES (?, ?, ?, ?, ?)"


def rows_added(transaction: Writing, lists: Sequence[ListsRow], blocks: Sequence[ListBlockRow]) -> None:
    run_each(transaction, POSTINGS_ROW, lists)
    run_each(transaction, BLOCK_ROW, blocks)


def lengths_added(transaction: Writing, segment: int, lengths: Mapping[int, bytes]) -> None:
    run_each(
        transaction,
        "INSERT INTO keyword_lengths(segment, block, lengths) VALUES (?, ?, ?)",
        [(segment, block, data) for block, data in lengths.items()],
    )
