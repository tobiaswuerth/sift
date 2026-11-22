import itertools
from dataclasses import dataclass, field

import numpy as np

from sift.domain.arrays import Ids, Mask, after, among, distinct, first, ints, last, run_starts, total
from sift.domain.codec.postings import (
    Ranges,
    blocked,
    blocked_all,
    lengths_of,
    positions_of,
    ranges_of,
)
from sift.domain.codec.segments import Shape
from sift.domain.errors import DamagedError
from sift.storage import keyword_lists
from sift.storage.database import Reading
from sift.storage.keyword_lists import SegmentRow
from sift.storage.keywords import shape_of, totals


@dataclass(frozen=True, slots=True)
class Whole:
    segment: int
    term: str
    places: Ids
    counts: Ids


@dataclass(frozen=True, slots=True)
class Ranged:
    segment: int
    term: str
    ranges: Ranges


type Stored = Whole | Ranged
type Postings = tuple[Ids, Ids]


@dataclass(frozen=True, slots=True)
class Dense:
    base: int
    lengths: Ids
    loaded: Mask


@dataclass(slots=True)
class Opened:
    transaction: Reading
    shape: Shape
    segments: tuple[SegmentRow, ...]
    dead: Ids
    chunks: int
    tokens: int
    lengths: dict[int, Dense] = field(default_factory=dict[int, Dense])
    blocks: dict[tuple[int, str, int], Postings] = field(default_factory=dict[tuple[int, str, int], Postings])


def opened(transaction: Reading) -> Opened:
    held = totals(transaction)
    found = tuple(keyword_lists.segments(transaction))
    if any(before.last >= after.first for before, after in itertools.pairwise(found)):
        raise DamagedError("segments of the keyword index overlap")
    return Opened(transaction, shape_of(transaction), found, keyword_lists.dead(transaction), held.chunks, held.tokens)


def live(store: Opened, places: Ids, counts: Ids) -> Postings:
    if not len(store.dead):
        return places, counts
    kept = ~among(places, store.dead)
    return places[kept], counts[kept]


def stored(store: Opened, terms: list[str]) -> dict[str, list[Stored]]:
    found: dict[str, list[Stored]] = {term: [] for term in terms}
    parts = keyword_lists.parts(store.transaction, terms)
    whole = [part for part in parts if not part.ranged]
    places, counts, bounds = blocked_all([part.data for part in whole])
    for part, (start, end) in zip(whole, itertools.pairwise(bounds), strict=True):
        found[part.term].append(Whole(part.segment, part.term, places[start:end], counts[start:end]))
    placed = {held.id: held for held in store.segments}
    for part in parts:
        if part.ranged:
            found[part.term].append(Ranged(part.segment, part.term, _placed(store, placed, part.segment, part.data)))
    return found


def _placed(store: Opened, placed: dict[int, SegmentRow], segment: int, data: bytes) -> Ranges:
    ranges, shift = ranges_of(data), store.shape.range_shift
    held = placed.get(segment)
    if held is None or first(ranges.ids) < held.first >> shift or last(ranges.ids) > held.last >> shift:
        raise DamagedError("the ranges of a list lie outside its segment")
    return ranges


def slices(starts: Ids, ends: Ids) -> Ids:
    sizes = ends - starts
    return np.repeat(starts - after(0, np.cumsum(sizes)[:-1]), sizes) + np.arange(total(sizes))


def _runs(keys: Ids) -> list[tuple[int, int]]:
    return list(itertools.pairwise([*ints(run_starts(keys)), len(keys)]))


def _one[T](found: list[T], what: str) -> T:
    if len(found) != 1:
        raise DamagedError(f"{what} is missing from the keyword index")
    return found[0]


def _block(store: Opened, held: Ranged, number: int) -> Postings:
    key = (held.segment, held.term, number)
    if key not in store.blocks:
        row = _one(keyword_lists.blocks(store.transaction, held.segment, held.term, [number]), "a block of postings")
        places, counts = blocked(row.postings)
        size = min(store.shape.block, last(held.ranges.ends) - number * store.shape.block)
        if len(places) != size:
            raise DamagedError(f"a block holds {len(places):,} postings where its ranges place {size:,}")
        store.blocks[key] = places, counts
    return store.blocks[key]


def _indexes(held: Ranged, chosen: Ids) -> Ids:
    return slices(held.ranges.starts[chosen], held.ranges.ends[chosen])


def _at(store: Opened, held: Ranged, indexes: Ids) -> Postings:
    numbers = indexes // store.shape.block
    places, counts = np.empty(len(indexes), np.int64), np.empty(len(indexes), np.int64)
    for start, end in _runs(numbers):
        number = first(numbers[start:end])
        found, tf = _block(store, held, number)
        at = indexes[start:end] - number * store.shape.block
        places[start:end], counts[start:end] = found[at], tf[at]
    return places, counts


def postings_in(store: Opened, held: Ranged, chosen: Ids) -> Postings:
    return live(store, *_at(store, held, _indexes(held, chosen)))


def everything(store: Opened, held: Stored) -> Postings:
    if isinstance(held, Whole):
        return live(store, held.places, held.counts)
    return postings_in(store, held, np.arange(len(held.ranges.ids)))


def _segments_at(store: Opened, places: Ids) -> Ids:
    starts = np.array([segment.first for segment in store.segments], np.int64)
    ends = np.array([segment.last for segment in store.segments], np.int64)
    at = np.searchsorted(starts, places, side="right") - 1
    if len(places) and (int(at.min()) < 0 or bool((places > ends[np.maximum(at, 0)]).any())):
        raise DamagedError("the keyword index holds chunks outside its segments")
    return at


def _dense(store: Opened, segment: SegmentRow) -> Dense:
    if segment.id not in store.lengths:
        shift = store.shape.lengths_shift
        base, top = segment.first >> shift, segment.last >> shift
        size = (top - base + 1) << shift
        store.lengths[segment.id] = Dense(base, np.zeros(size, np.int64), np.zeros(top - base + 1, np.bool_))
    return store.lengths[segment.id]


def _loaded(store: Opened, segment: SegmentRow, blocks: Ids) -> Dense:
    held, shift = _dense(store, segment), store.shape.lengths_shift
    missing = distinct(blocks[~held.loaded[blocks - held.base]])
    if len(missing):
        for row in keyword_lists.lengths(store.transaction, segment.id, ints(missing)):
            at = (row.block - held.base) << shift
            held.lengths[at : at + (1 << shift)] = lengths_of(row.lengths, shift)
            held.loaded[row.block - held.base] = True
        if not bool(held.loaded[missing - held.base].all()):
            raise DamagedError("the lengths of some chunks are missing from the keyword index")
    return held


def lengths_at(store: Opened, places: Ids) -> Ids:
    shift = store.shape.lengths_shift
    order = np.argsort(places, kind="stable")
    ordered = places[order]
    at = _segments_at(store, ordered)
    found = np.empty(len(places), np.int64)
    for start, end in _runs(at):
        segment = store.segments[first(at[start:end])]
        here = ordered[start:end]
        held = _loaded(store, segment, here >> shift)
        found[start:end] = held.lengths[here - (held.base << shift)]
    result = np.empty(len(places), np.int64)
    result[order] = found
    return result


def _positions_whole(store: Opened, held: Whole, wanted: Ids) -> tuple[Ids, Ids, Ids]:
    row = _one(keyword_lists.positions(store.transaction, held.segment, held.term, None), "a list of positions")
    flat = positions_of(row.positions, held.counts)
    return _chosen(held.places, held.counts, flat, wanted)


def _chosen(places: Ids, counts: Ids, flat: Ids, wanted: Ids) -> tuple[Ids, Ids, Ids]:
    kept = among(places, wanted)
    return places[kept], counts[kept], flat[np.repeat(kept, counts)]


def _positions_ranged(store: Opened, held: Ranged, wanted: Ids) -> tuple[Ids, Ids, Ids]:
    chosen = np.flatnonzero(among(held.ranges.ids, distinct(wanted >> store.shape.range_shift)))
    numbers = ints(distinct(_indexes(held, chosen) // store.shape.block))
    parts = [_none()]
    for row in keyword_lists.positions(store.transaction, held.segment, held.term, numbers):
        places, counts = _block(store, held, row.number)
        parts.append(_chosen(places, counts, positions_of(row.positions, counts), wanted))
    return joined(parts)


def _none() -> tuple[Ids, Ids, Ids]:
    empty = np.zeros(0, np.int64)
    return empty, empty, empty


def joined(parts: list[tuple[Ids, Ids, Ids]]) -> tuple[Ids, Ids, Ids]:
    places, counts, positions = zip(*parts, strict=True)
    return np.concatenate(places), np.concatenate(counts), np.concatenate(positions)


def positions_at(store: Opened, held: Stored, wanted: Ids) -> tuple[Ids, Ids, Ids]:
    if isinstance(held, Whole):
        return _positions_whole(store, held, wanted)
    return _positions_ranged(store, held, wanted)


def owners(store: Opened, places: Ids) -> Ids:
    found = keyword_lists.owners(store.transaction, places)
    if set(ints(places)) - found.keys():
        raise DamagedError("the keyword index holds chunks the store does not")
    return np.fromiter(map(found.__getitem__, ints(places)), np.int64, len(places))
