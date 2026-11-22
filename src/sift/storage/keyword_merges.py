import heapq
import itertools
import json
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ascending, Ids, among, first, firsts, ints, last, run_starts, total
from sift.domain.codec.postings import blocked_all, lengths_of, positions_all, with_lengths
from sift.domain.codec.segments import Lists, Shape, postings_of
from sift.domain.errors import DamagedError
from sift.domain.plan.segments import Held, Policy, merges
from sift.storage.database import Batches, Reading, WritingDocuments, rows, run, streamed
from sift.storage.keyword_writes import Head, lengths_added, postings_added, segment_added

SHORT = """SELECT :rank, term, -1, chunks, length(postings) + length(positions), postings, positions
           FROM keyword_postings WHERE segment = :segment AND postings IS NOT NULL ORDER BY term"""
BLOCKS = """SELECT :rank, term, number, :block, length(postings) + length(positions), postings, positions
            FROM keyword_blocks WHERE segment = :segment ORDER BY term, number"""
LENGTHS = """SELECT segment, block, lengths FROM keyword_lengths WHERE segment IN (SELECT value FROM json_each(?))
             ORDER BY segment, block"""
HELD = "SELECT coalesce(sum(chunks - dead), 0), (SELECT tokens FROM counts) FROM keyword_segments"
DROPPED = (
    "DELETE FROM keyword_blocks WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_postings WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_lengths WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_segments WHERE id IN (SELECT value FROM json_each(?))",
)


class Placed(NamedTuple):
    id: int
    first: int
    last: int
    chunks: int
    dead: int


class Listed(NamedTuple):
    rank: int
    term: str
    number: int
    chunks: int
    size: int
    postings: bytes
    positions: bytes


class Chunk(NamedTuple):
    id: int


class LengthsRow(NamedTuple):
    segment: int
    block: int
    lengths: bytes


class Totals(NamedTuple):
    chunks: int
    tokens: int


def _ids(transaction: WritingDocuments, sql: str, low: int, high: int) -> Ids:
    return np.fromiter((found.id for found in streamed(transaction, Chunk, sql, (low, high))), np.int64)


def _order(row: Listed) -> tuple[str, int, int]:
    return row.term, row.rank, row.number


def _listed(transaction: WritingDocuments, merged: Sequence[Placed], block: int) -> Iterator[Listed]:
    streams = [
        streamed(transaction, Listed, sql, {"rank": rank, "segment": each.id, "block": block})
        for rank, each in enumerate(merged)
        for sql in (SHORT, BLOCKS)
    ]
    return heapq.merge(*streams, key=_order)


def batched(listed: Iterator[Listed], batches: Batches) -> Iterator[list[Listed]]:
    batch: list[Listed] = []
    volume = size = 0
    for _, rows_of_term in itertools.groupby(listed, key=lambda row: row.term):
        held = list(rows_of_term)
        batch.extend(held)
        volume += sum(row.chunks for row in held)
        size += sum(row.size for row in held)
        if volume >= batches.postings_a_merge or size >= batches.bytes_a_merge:
            yield batch
            batch, volume, size = [], 0, 0
    if batch:
        yield batch


def _lengths(transaction: WritingDocuments, merged: Sequence[Placed], live: Ids, shift: int) -> Ids:
    dense = {
        (row.segment, row.block): lengths_of(row.lengths, shift)
        for row in rows(transaction, LengthsRow, LENGTHS, (json.dumps([each.id for each in merged]),))
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


def _lists(listed: Sequence[Listed], dead: Ids, live: Ids, lengths: Ids) -> Lists:
    places, counts, bounds = blocked_all([row.postings for row in listed])
    positions = positions_all([row.positions for row in listed], counts)
    names = sorted({row.term for row in listed})
    coded = {name: code for code, name in enumerate(names)}
    codes = np.repeat(np.array([coded[row.term] for row in listed], np.int64), np.diff(bounds))
    kept = ~among(places, Ascending(dead))
    positions = positions[np.repeat(kept, counts)]
    places, counts, codes = places[kept], counts[kept], codes[kept]
    at = np.minimum(np.searchsorted(live, places), max(len(live) - 1, 0))
    if len(places) and not bool(np.array_equal(live[at], places)):
        raise DamagedError("the keyword index holds chunks the store does not")
    starts = ints(run_starts(codes))
    terms = tuple(names[code] for code in ints(codes[starts]))
    return Lists(terms, tuple(starts), places, counts, positions, lengths[at])


def _merged(transaction: WritingDocuments, merged: Sequence[Placed], shape: Shape) -> None:
    low, high = merged[0].first, merged[-1].last
    named = json.dumps([each.id for each in merged])
    live = _ids(transaction, "SELECT id FROM chunks WHERE id BETWEEN ? AND ? ORDER BY id", low, high)
    if not len(live):
        raise DamagedError("segments of the keyword index hold no chunk the store holds")
    lengths = _lengths(transaction, merged, live, shape.lengths_shift)
    (held,) = rows(transaction, Totals, HELD)
    average = held.tokens / held.chunks
    made = segment_added(transaction, Head(first(live), last(live), len(live), total(lengths), average))
    dead = _ids(transaction, "SELECT chunk FROM keyword_dead WHERE chunk BETWEEN ? AND ? ORDER BY chunk", low, high)
    listed = _listed(transaction, merged, shape.block)
    for batch in batched(listed, transaction.batches):
        lists = _lists(batch, dead, live, lengths)
        if lists.terms:
            postings_added(transaction, made, postings_of(lists, average, shape))
    lengths_added(transaction, made, with_lengths(live, lengths, shape.lengths_shift))
    for sql in DROPPED:
        run(transaction, sql, (named,))


def _plan(transaction: Reading, policy: Policy) -> list[list[Placed]]:
    held = rows(transaction, Placed, "SELECT id, first, last, chunks, dead FROM keyword_segments ORDER BY first")
    return [[held[at] for at in each] for each in merges([Held(each.chunks, each.dead) for each in held], policy)]


def _planned(transaction: WritingDocuments) -> list[list[Placed]]:
    return _plan(transaction, transaction.batches.merging)


def planned(transaction: Reading, policy: Policy) -> bool:
    return bool(_plan(transaction, policy))


def _live(merged: Sequence[Placed]) -> int:
    return sum(each.chunks - each.dead for each in merged)


def settled(transaction: WritingDocuments, shape: Shape) -> None:
    written, most = 0, transaction.batches.chunks_a_settle
    while fitting := [merged for merged in _planned(transaction) if written + _live(merged) <= most]:
        for merged in fitting:
            if written + _live(merged) <= most:
                _merged(transaction, merged, shape)
                written += _live(merged)


def merged_once(transaction: WritingDocuments, shape: Shape) -> int:
    chosen = _planned(transaction)
    if not chosen:
        return 0
    _merged(transaction, chosen[0], shape)
    return _live(chosen[0])
