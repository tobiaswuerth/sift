import heapq
import itertools
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ascending, Ids, after, among, first, firsts, ints, last, run_starts, total
from sift.domain.codec.postings import blocked_all, encoded, lengths_of, positions_all, with_lengths
from sift.domain.codec.segments import (
    Lists,
    MergedRows,
    Merging,
    Postings,
    RangesPart,
    ShortPostings,
    blocks_of,
    lists_merged,
    merger,
    ranges_joined,
    ranges_part,
)
from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from sift.domain.plan.segments import Held, Policy, backlog_merges, merges
from sift.storage.database import Batches, IdRow, Reading, WritingDocuments, rows, run, streamed
from sift.storage.keyword_meta import Side
from sift.storage.keyword_writes import (
    Head,
    blocks_added,
    lengths_added,
    long_begun,
    long_finished,
    postings_added,
    rows_added,
    segment_added,
)

SHORT = """SELECT :rank, term, -1, chunks, length(postings) + length(positions), postings, positions
           FROM keyword_postings WHERE segment = :segment AND postings IS NOT NULL ORDER BY term"""
BLOCKS = """SELECT :rank, term, number, :block, length(postings) + length(positions), postings, positions
            FROM keyword_blocks WHERE segment = :segment ORDER BY term, number"""
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
ROWS_AT_ONCE = 64
DROPPED = (
    "DELETE FROM keyword_blocks WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_postings WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_lengths WHERE segment IN (SELECT value FROM json_each(?))",
    "DELETE FROM keyword_segments WHERE id IN (SELECT value FROM json_each(?))",
)


class PlacedSegment(NamedTuple):
    id: int
    first: int
    last: int
    chunks: int
    dead: int


class MergedTerm(NamedTuple):
    rank: int
    term: str
    number: int
    chunks: int
    size: int
    postings: bytes
    positions: bytes


class SegmentLengths(NamedTuple):
    segment: int
    block: int
    lengths: bytes


class Totals(NamedTuple):
    chunks: int
    tokens: int


def _ids(transaction: WritingDocuments, sql: str, low: int, high: int) -> Ids:
    return np.fromiter((found.id for found in streamed(transaction, IdRow, sql, (low, high))), np.int64)


def _order(row: MergedTerm) -> tuple[str, int, int]:
    return row.term, row.rank, row.number


def _listed(transaction: WritingDocuments, merged: Sequence[PlacedSegment], block: int) -> Iterator[MergedTerm]:
    streams = [
        streamed(transaction, MergedTerm, sql, {"rank": rank, "segment": each.id, "block": block}, ROWS_AT_ONCE)
        for rank, each in enumerate(merged)
        for sql in (SHORT, BLOCKS)
    ]
    return heapq.merge(*streams, key=_order)


class TermsBatch(NamedTuple):
    rows: list[MergedTerm]


class LargeTerm(NamedTuple):
    rows: Iterator[MergedTerm]


def _too_large(postings: int, size: int, batches: Batches) -> bool:
    return postings >= batches.postings_a_merge or size >= batches.bytes_a_merge


class _Taken(NamedTuple):
    rows: list[MergedTerm]
    postings: int
    size: int


def _taken(rows_of_term: Iterator[MergedTerm], batches: Batches) -> _Taken | LargeTerm:
    taken: list[MergedTerm] = []
    postings = size = 0
    for row in rows_of_term:
        taken.append(row)
        postings, size = postings + row.chunks, size + row.size
        if _too_large(postings, size, batches):
            return LargeTerm(itertools.chain(taken, rows_of_term))
    return _Taken(taken, postings, size)


def batched(listed: Iterator[MergedTerm], batches: Batches) -> Iterator[TermsBatch | LargeTerm]:
    batch: list[MergedTerm] = []
    postings = size = 0
    for _, rows_of_term in itertools.groupby(listed, key=lambda row: row.term):
        match _taken(rows_of_term, batches):
            case LargeTerm() as large:
                if batch:
                    yield TermsBatch(batch)
                    batch, postings, size = [], 0, 0
                yield large
            case taken:
                batch.extend(taken.rows)
                postings, size = postings + taken.postings, size + taken.size
                if _too_large(postings, size, batches):
                    yield TermsBatch(batch)
                    batch, postings, size = [], 0, 0
    if batch:
        yield TermsBatch(batch)


def _pieces(rows: Iterator[MergedTerm], batches: Batches) -> Iterator[list[MergedTerm]]:
    piece: list[MergedTerm] = []
    postings = size = 0
    for row in rows:
        if piece and _too_large(postings + row.chunks, size + row.size, batches):
            yield piece
            piece, postings, size = [], 0, 0
        piece.append(row)
        postings, size = postings + row.chunks, size + row.size
    yield piece


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


@dataclass(eq=False, frozen=True, slots=True)
class _Merging:
    made: int
    dead: Ids
    live: Ids
    lengths: Ids
    average: float
    shape: Shape
    kernel: Merging


@dataclass(eq=False, frozen=True, slots=True)
class _Decoded:
    rows: Sequence[MergedTerm]
    places: Ids
    counts: Ids
    bounds: list[int]
    held: list[int]


def _decoded(rows: Sequence[MergedTerm]) -> _Decoded:
    places, counts, bounds = blocked_all([row.postings for row in rows])
    return _Decoded(rows, places, counts, bounds, ints(np.diff(after(0, np.cumsum(counts))[bounds])))


class _Span(NamedTuple):
    decoded: _Decoded
    start: int
    end: int


def _spans(sizes: Sequence[int], most: int) -> Iterator[tuple[int, int]]:
    start = held = 0
    for at, size in enumerate(sizes):
        if at > start and held + size > most:
            yield start, at
            start, held = at, 0
        held += size
    yield start, len(sizes)


def _within(spans: Iterator[_Span], most: int) -> Iterator[_Span]:
    for span in spans:
        for start, end in _spans(span.decoded.held[span.start : span.end], most):
            yield _Span(span.decoded, span.start + start, span.start + end)


def _lists(merging: _Merging, span: _Span) -> Lists:
    dead, live, lengths = merging.dead, merging.live, merging.lengths
    listed, bounds = span.decoded.rows[span.start : span.end], span.decoded.bounds[span.start : span.end + 1]
    places, counts = span.decoded.places[bounds[0] : bounds[-1]], span.decoded.counts[bounds[0] : bounds[-1]]
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


def _short_written(transaction: WritingDocuments, merging: _Merging, lists: Lists) -> None:
    coded = encoded(lists.ids, lists.counts, lists.positions, [0, len(lists.ids)])
    held = ShortPostings(coded.postings[0], coded.positions[0])
    postings_added(transaction, merging.made, (Postings(lists.terms[0], len(lists.ids), held),))


def _joined(held: Sequence[Lists]) -> Lists:
    def joined(parts: list[Ids]) -> Ids:
        return np.concatenate([np.zeros(0, np.int64), *parts])

    ids, counts = joined([each.ids for each in held]), joined([each.counts for each in held])
    positions, lengths = joined([each.positions for each in held]), joined([each.lengths for each in held])
    return Lists(held[0].terms, (0,), ids, counts, positions, lengths)


def _cut(lists: Lists, end: int) -> tuple[Lists, Lists]:
    at = total(lists.counts[:end])
    head = Lists(lists.terms, (0,), lists.ids[:end], lists.counts[:end], lists.positions[:at], lists.lengths[:end])
    rest = Lists(lists.terms, (0,), lists.ids[end:], lists.counts[end:], lists.positions[at:], lists.lengths[end:])
    return head, rest


def _written_blocks(transaction: WritingDocuments, merging: _Merging, pending: Lists, written: int) -> int:
    blocks = blocks_of(pending.ids, pending.counts, pending.positions, merging.shape)
    blocks_added(transaction, merging.made, pending.terms[0], written, blocks)
    return written + len(blocks)


def _term_written(transaction: WritingDocuments, merging: _Merging, spans: Iterator[_Span]) -> None:
    shape, held, parts, chunks = merging.shape, list[Lists](), list[RangesPart](), 0
    written: int | None = None
    for span in _within(spans, transaction.batches.tokens_a_segment):
        lists = _lists(merging, span)
        if not lists.terms:
            continue
        parts.append(ranges_part(lists.ids, lists.counts, lists.lengths, merging.average, shape))
        held.append(lists)
        chunks += len(lists.ids)
        if chunks >= shape.long:
            if written is None:
                long_begun(transaction, merging.made, lists.terms[0])
            pending = _joined(held)
            ready, rest = _cut(pending, len(pending.ids) // shape.block * shape.block)
            written, held = _written_blocks(transaction, merging, ready, written or 0), [rest]
    if written is not None:
        last = _joined(held)
        _written_blocks(transaction, merging, last, written)
        long_finished(transaction, merging.made, last.terms[0], chunks, ranges_joined(parts))
    elif held:
        _short_written(transaction, merging, _joined(held))


def _batch_written(transaction: WritingDocuments, merging: _Merging, rows: Sequence[MergedTerm]) -> None:
    given = MergedRows([row.term for row in rows], [row.postings for row in rows], [row.positions for row in rows])
    made = lists_merged(merging.kernel, merging.made, given, transaction.batches.tokens_a_segment)
    rows_added(transaction, made.lists, made.blocks)
    for start, end in made.large:
        _term_written(transaction, merging, iter([_Span(_decoded(rows[start:end]), 0, end - start)]))


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
    merging = _Merging(made, dead, live, lengths, average, shape, merger(live, lengths, dead, average, shape))
    for batch in batched(_listed(transaction, merged, shape.block), transaction.batches):
        match batch:
            case TermsBatch(rows=terms_rows):
                _batch_written(transaction, merging, terms_rows)
            case large:
                pieces = map(_decoded, _pieces(large.rows, transaction.batches))
                _term_written(transaction, merging, (_Span(each, 0, len(each.rows)) for each in pieces))
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
