from collections.abc import Mapping
from typing import NamedTuple

from sift.domain.codec.segments import FiledLists
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


PACK_ROW = "INSERT INTO keyword_packs(segment, first, lists, places) VALUES (?, ?, ?, ?)"
RANGES_ROW = "INSERT INTO keyword_ranges(segment, term, chunks, ranges) VALUES (?, ?, ?, ?)"
BLOCK_ROW = "INSERT INTO keyword_blocks(segment, term, number, postings, positions) VALUES (?, ?, ?, ?, ?)"


def rows_added(transaction: Writing, filed: FiledLists) -> None:
    run_each(transaction, PACK_ROW, filed.packs)
    run_each(transaction, RANGES_ROW, filed.ranged)
    run_each(transaction, BLOCK_ROW, filed.blocks)


def lengths_added(transaction: Writing, segment: int, lengths: Mapping[int, bytes]) -> None:
    run_each(
        transaction,
        "INSERT INTO keyword_lengths(segment, block, lengths) VALUES (?, ?, ?)",
        [(segment, block, data) for block, data in lengths.items()],
    )
