import itertools
import json
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids, first, firsts, ints, last
from sift.domain.codec.postings import lengths_of, with_lengths
from sift.domain.codec.segments import MergedChunks, MergeLimits, fed, fed_packs, merger, stepped
from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.segments import Held, Policy, backlog_merges, merges
from sift.storage.database import IdRow, Reading, WritingDocuments, row_batches, rows, run, streamed
from sift.storage.keyword_meta import Side
from sift.storage.keyword_writes import Head, lengths_added, rows_added, segment_added

LISTED = (
    "SELECT lists, places FROM keyword_packs WHERE segment = :segment ORDER BY first",
    """SELECT term, :block, postings, positions FROM keyword_blocks WHERE segment = :segment ORDER BY term, number""",
)
ROWS_AT_ONCE = (32, 32)
FED = (fed_packs, fed)
LENGTHS = """SELECT segment, block, lengths FROM keyword_lengths WHERE segment IN (SELECT value FROM json_each(?))
             ORDER BY segment, block"""
HELD = "SELECT coalesce(sum(chunks - dead), 0), (SELECT tokens FROM counts) FROM keyword_served"
LIVE = (
    "SELECT place AS id FROM chunks WHERE place BETWEEN ? AND ? ORDER BY place",
    "SELECT place AS id FROM keyword_places WHERE place BETWEEN ? AND ? ORDER BY place",
)
PLANNED = (
    "SELECT id, first, last, chunks, dead FROM keyword_served ORDER BY first",
    "SELECT id, first, last, chunks, dead FROM keyword_built ORDER BY first",
)
DROPPED = (
    "DELETE FROM keyword_blocks WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_packs WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_ranges WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_lengths WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_segments WHERE id IN (SELECT value FROM json_each(?))",
)


class PlacedSegment(NamedTuple):
    id: int
    first: int
    last: int
    chunks: int
    dead: int


class SegmentLengths(NamedTuple):
    segment: int
    block: int
    lengths: bytes


class Totals(NamedTuple):
    chunks: int
    tokens: int


def _ids(transaction: WritingDocuments, sql: str, low: int, high: int) -> Ids:
    return np.fromiter((found.id for found in streamed(transaction, IdRow, sql, (low, high))), np.int64)


def _listed(
    transaction: WritingDocuments, merged: Sequence[PlacedSegment], block: int
) -> list[Iterator[list[tuple[object, ...]]]]:
    return [
        row_batches(transaction, sql, {"segment": each.id, "block": block}, size)
        for each in merged
        for sql, size in zip(LISTED, ROWS_AT_ONCE, strict=True)
    ]


def _lengths(transaction: WritingDocuments, merged: Sequence[PlacedSegment], live: Ids, shift: int) -> Ids:
    dense = {
        (row.segment, row.block): lengths_of(row.lengths, shift)
        for row in rows(transaction, SegmentLengths, LENGTHS, (json.dumps([each.id for each in merged]),))
    }
    starts = np.array([each.first for each in merged], np.int64)
    owners = np.array([each.id for each in merged], np.int64)[np.searchsorted(starts, live, side="right") - 1]
    blocks = live >> shift
    found = np.empty(len(live), np.int64)
    cuts = [*ints(np.flatnonzero(firsts(owners) | firsts(blocks))), len(live)]
    for start, end in itertools.pairwise(cuts):
        key = (first(owners[start:end]), first(blocks[start:end]))
        if key not in dense:
            raise DamagedError("the lengths of some chunks are missing from the keyword index")
        found[start:end] = dense[key][live[start:end] & ((1 << shift) - 1)]
    return found


def _merged(transaction: WritingDocuments, merged: Sequence[PlacedSegment], shape: Shape, side: Side) -> None:
    low, high = merged[0].first, merged[-1].last
    named = json.dumps([each.id for each in merged])
    live = _ids(transaction, LIVE[side], low, high)
    if not len(live):
        raise DamagedError("segments of the keyword index hold no chunk the store holds")
    lengths = _lengths(transaction, merged, live, shape.lengths_shift)
    (held,) = rows(transaction, Totals, HELD)
    average = held.tokens / held.chunks
    made = segment_added(transaction, Head(first(live), last(live), len(live), average))
    dead = _ids(
        transaction, "SELECT place AS id FROM keyword_dead WHERE place BETWEEN ? AND ? ORDER BY place", low, high
    )
    batches, listed = transaction.batches, _listed(transaction, merged, shape.block)
    limits = MergeLimits(batches.postings_a_merge, batches.bytes_a_merge, batches.tokens_a_segment)
    held = merger(MergedChunks(live, lengths, dead, average), shape, limits, len(listed))
    done = False
    while not done:
        step = stepped(held, made)
        rows_added(transaction, step.filed)
        for source in step.wanted:
            while FED[source % len(FED)](held, source, next(listed[source], ())):
                pass
        done = step.done
    lengths_added(transaction, made, with_lengths(live, lengths, shape.lengths_shift))
    for sql in DROPPED:
        run(transaction, sql, (named,))


type Plan = Callable[[Sequence[Held], Policy], tuple[range, ...]]


def _plan(transaction: Reading, policy: Policy, side: Side, plan: Plan) -> list[list[PlacedSegment]]:
    held = rows(transaction, PlacedSegment, PLANNED[side])
    return [[held[at] for at in each] for each in plan([Held(each.chunks, each.dead) for each in held], policy)]


def _planned(transaction: WritingDocuments, side: Side) -> list[list[PlacedSegment]]:
    return _plan(transaction, transaction.batches.merging, side, merges)


def planned(transaction: Reading, policy: Policy) -> bool:
    return bool(_plan(transaction, policy, Side.SERVED, backlog_merges))


def _live(merged: Sequence[PlacedSegment]) -> int:
    return sum(each.chunks - each.dead for each in merged)


def settled(transaction: WritingDocuments, shape: Shape, side: Side = Side.SERVED) -> None:
    written, most = 0, transaction.batches.chunks_a_settle
    while fitting := [merged for merged in _planned(transaction, side) if written + _live(merged) <= most]:
        for merged in fitting:
            if written + _live(merged) <= most:
                _merged(transaction, merged, shape, side)
                written += _live(merged)


def merged_once(transaction: WritingDocuments, shape: Shape) -> int:
    chosen = _plan(transaction, transaction.batches.merging, Side.SERVED, backlog_merges)
    if not chosen:
        return 0
    _merged(transaction, chosen[0], shape, Side.SERVED)
    return _live(chosen[0])
