import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sift.domain.arrays import Ascending, Ids, after, among, distinct, first, ints, run_starts, total
from sift.domain.codec.postings import (
    HeldBlock,
    HeldRanges,
    HeldTable,
    Ranges,
    blocked_all,
    blocked_within,
    occurrences,
    postings_of,
    ranges_in,
    ranges_kept,
)
from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from sift.storage import keyword_lists
from sift.storage.database import Reading
from sift.storage.keyword_lists import Part, SegmentRow
from sift.storage.keywords import shape_of, totals


@dataclass(eq=False, frozen=True, slots=True)
class Whole:
    segment: int
    term: str
    places: Ids
    counts: Ids


@dataclass(eq=False, frozen=True, slots=True)
class Ranged:
    segment: int
    term: str
    chunks: int
    ranges: HeldRanges


type StoredList = Whole | Ranged
type PostingArrays = tuple[Ids, Ids]
type BlockKey = tuple[int, str, int]


class Keeping[K, V](Protocol):
    def get(self, key: K, /) -> V | None: ...

    def __setitem__(self, key: K, value: V, /) -> None: ...


@dataclass(eq=False, frozen=True, slots=True)
class DecodedLists:
    lengths: Keeping[int, HeldTable]
    blocks: Keeping[BlockKey, HeldBlock]


def decoded_anew() -> DecodedLists:
    return DecodedLists(dict[int, HeldTable](), dict[BlockKey, HeldBlock]())


@dataclass(eq=False, slots=True)
class Opened:
    transaction: Reading
    shape: Shape
    segments: tuple[SegmentRow, ...]
    chunks: int
    tokens: int
    decoded: DecodedLists = field(default_factory=decoded_anew)
    dead: dict[int, Ascending] = field(default_factory=dict[int, Ascending])


def opened(transaction: Reading, decoded: DecodedLists | None = None) -> Opened:
    held = totals(transaction)
    found = tuple(keyword_lists.segments(transaction))
    if any(before.last >= after.first for before, after in itertools.pairwise(found)):
        raise DamagedError("segments of the keyword index overlap")
    kept = decoded_anew() if decoded is None else decoded
    return Opened(transaction, shape_of(transaction), found, held.chunks, held.tokens, kept)


def segment_row(store: Opened, segment: int) -> SegmentRow:
    return _one([each for each in store.segments if each.id == segment], "a segment")


def dead(store: Opened, segment: int) -> Ascending:
    if segment not in store.dead:
        held = segment_row(store, segment)
        store.dead[segment] = Ascending(keyword_lists.dead_within(store.transaction, held.first, held.last))
    return store.dead[segment]


def live(store: Opened, segment: int, places: Ids, counts: Ids) -> PostingArrays:
    gone = dead(store, segment)
    if not len(gone.values):
        return places, counts
    kept = ~among(places, gone)
    return places[kept], counts[kept]


def stored(store: Opened, terms: list[str]) -> dict[str, list[StoredList]]:
    return _listed(store, keyword_lists.parts(store.transaction, terms), terms)


def stored_within(store: Opened, prefix: str) -> dict[str, list[StoredList]]:
    parts = keyword_lists.parts_within(store.transaction, prefix)
    return _listed(store, parts, sorted({part.term for part in parts}))


def _listed(store: Opened, parts: list[Part], terms: list[str]) -> dict[str, list[StoredList]]:
    found: dict[str, list[StoredList]] = {term: [] for term in terms}
    whole = [part for part in parts if not part.ranged]
    places, counts, bounds = blocked_all([part.data for part in whole])
    for part, (start, end) in zip(whole, itertools.pairwise(bounds), strict=True):
        if end - start != part.chunks:
            raise DamagedError(f"the list of {part.term!r} holds {end - start:,} passages, not its {part.chunks:,}")
        found[part.term].append(Whole(part.segment, part.term, places[start:end], counts[start:end]))
    ranged = [part for part in parts if part.ranged]
    shift, placed = store.shape.range_shift, {held.id: held for held in store.segments}
    told = [(placed[part.segment].first >> shift, placed[part.segment].last >> shift, part.chunks) for part in ranged]
    for part, ranges in zip(ranged, ranges_kept([part.data for part in ranged], told), strict=True):
        found[part.term].append(Ranged(part.segment, part.term, part.chunks, ranges))
    return found


def slices(starts: Ids, ends: Ids) -> Ids:
    sizes = ends - starts
    return np.repeat(starts - after(0, np.cumsum(sizes)[:-1]), sizes) + np.arange(total(sizes))


def _runs(keys: Ids) -> list[tuple[int, int]]:
    return list(itertools.pairwise([*ints(run_starts(keys)), len(keys)]))


def _one[T](found: list[T], what: str) -> T:
    if len(found) != 1:
        raise DamagedError(f"{what} is missing from the keyword index")
    return found[0]


def decoded_blocks(store: Opened, held: Ranged, numbers: Sequence[int]) -> dict[int, HeldBlock]:
    found: dict[int, HeldBlock] = {}
    missing: list[int] = []
    for number in numbers:
        kept = store.decoded.blocks.get((held.segment, held.term, number))
        if kept is None:
            missing.append(number)
        else:
            found[number] = kept
    if not missing:
        return found
    read = keyword_lists.blocks(store.transaction, held.segment, held.term, missing)
    if [row.number for row in read] != missing:
        raise DamagedError("a block of postings is missing from the keyword index")
    decoded = blocked_within([row.postings for row in read], missing, held.ranges, store.shape)
    for number, block in zip(missing, decoded, strict=True):
        found[number] = block
        store.decoded.blocks[held.segment, held.term, number] = block
    return found


def lengths_read(store: Opened, segment: int, numbers: Sequence[int]) -> list[bytes]:
    read = keyword_lists.lengths(store.transaction, segment, numbers)
    if [row.block for row in read] != list(numbers):
        raise DamagedError("the lengths of some chunks are missing from the keyword index")
    return [row.lengths for row in read]


def _indexes(ranges: Ranges, chosen: Ids) -> Ids:
    return slices(ranges.starts[chosen], ranges.ends[chosen])


def _postings(store: Opened, held: Ranged, numbers: Sequence[int]) -> tuple[Ids, Ids, list[int]]:
    found = decoded_blocks(store, held, numbers)
    return postings_of([found[number] for number in numbers])


def _at(store: Opened, held: Ranged, indexes: Ids) -> PostingArrays:
    numbers = indexes // store.shape.block
    runs = _runs(numbers)
    wanted = [first(numbers[start:end]) for start, end in runs]
    places, counts, bounds = _postings(store, held, wanted)
    at = np.empty(len(indexes), np.int64)
    for (start, end), number, offset in zip(runs, wanted, bounds[:-1], strict=True):
        at[start:end] = indexes[start:end] - number * store.shape.block + offset
    return places[at], counts[at]


def postings_in(store: Opened, held: Ranged, ranges: Ranges, chosen: Ids) -> PostingArrays:
    return live(store, held.segment, *_at(store, held, _indexes(ranges, chosen)))


def everything(store: Opened, held: StoredList) -> PostingArrays:
    if isinstance(held, Whole):
        return live(store, held.segment, held.places, held.counts)
    places, counts, _ = _postings(store, held, range(-(-held.chunks // store.shape.block)))
    return live(store, held.segment, places, counts)


def _positions_whole(store: Opened, held: Whole, wanted: Ascending) -> tuple[Ids, Ids, Ids]:
    row = _one(keyword_lists.positions(store.transaction, held.segment, held.term, None), "a list of positions")
    return occurrences([row.positions], [0, len(held.places)], (held.places, held.counts), wanted.values)


def _positions_ranged(store: Opened, held: Ranged, wanted: Ascending) -> tuple[Ids, Ids, Ids]:
    ranges = ranges_in(held.ranges)
    chosen = np.flatnonzero(among(ranges.ids, distinct(Ascending(wanted.values >> store.shape.range_shift))))
    numbers = ints(distinct(Ascending(_indexes(ranges, chosen) // store.shape.block)).values)
    places, counts, bounds = _postings(store, held, numbers)
    read = keyword_lists.positions(store.transaction, held.segment, held.term, numbers) if numbers else []
    return occurrences([row.positions for row in read], bounds, (places, counts), wanted.values)


def joined(parts: list[tuple[Ids, Ids, Ids]]) -> tuple[Ids, Ids, Ids]:
    places, counts, positions = zip(*parts, strict=True)
    return np.concatenate(places), np.concatenate(counts), np.concatenate(positions)


def positions_at(store: Opened, held: StoredList, wanted: Ascending) -> tuple[Ids, Ids, Ids]:
    if isinstance(held, Whole):
        return _positions_whole(store, held, wanted)
    return _positions_ranged(store, held, wanted)


def owners(store: Opened, places: Ids) -> tuple[Ids, Ids]:
    found, chunks, documents = keyword_lists.chunks_placed(store.transaction, ints(places))
    at = np.minimum(np.searchsorted(found, places), max(len(found) - 1, 0))
    if len(places) and (not len(found) or not bool(np.array_equal(found[at], places))):
        raise DamagedError("the keyword index holds chunks the store does not")
    return chunks[at], documents[at]
