import itertools
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from sift.api.ids import ChunkId
from sift.domain.arrays import Ids, Mask, ascending, distinct, floats, ints, last, read_only
from sift.domain.codec.codes import (
    MOST_BITS,
    WORD_BYTES,
    Bits,
    Floats,
    Halves,
    Words,
    bit_scores,
    code_bytes,
    code_of,
    distances,
    width_of,
    words,
)
from sift.retrieval.allowed import AllowedChunks

MOST_RESCORED = 1000
SHORTLIST = 10
CODES_AXES = 2
SCAN_SHARE = 0.2
MARGIN = 1.25
SCANNING = 8
PART_ROWS = 32_768

type ExactRead = Callable[[int, Ids], tuple[Ids, Halves]]


class ScanPart(NamedTuple):
    number: int
    store: int
    rows: range
    at: int


type Spread = Callable[[Callable[[ScanPart], None], Sequence[ScanPart]], None]


def over_threads(scan: Callable[[ScanPart], None], parts: Sequence[ScanPart]) -> None:
    if len(parts) <= 1:
        for part in parts:
            scan(part)
        return
    with ThreadPoolExecutor(min(SCANNING, len(parts))) as pool:
        for _ in pool.map(scan, parts):
            pass


@dataclass(frozen=True, slots=True)
class Reads:
    exact: ExactRead
    allowed: Sequence[AllowedChunks | None] = ()
    spread: Spread = over_threads


@dataclass(eq=False, frozen=True, slots=True)
class Coded:
    ids: Ids
    bits: Bits
    words: Words
    width: int


class Near(NamedTuple):
    chunk: ChunkId
    score: float


@dataclass(eq=False, frozen=True, slots=True)
class Candidates:
    ids: Ids
    owners: Ids
    rows: Ids


def coded(ids: Ids, bits: Bits, width: int) -> Coded:
    if bits.dtype != np.uint8 or bits.ndim != CODES_AXES or len(bits) != len(ids):
        raise ValueError("codes are a uint8 matrix of a row a chunk: code them so")
    if len(ids) and not (0 < width <= MOST_BITS and width_of(bits) == code_bytes(width)):
        raise ValueError(f"codes of {width:,} dimensions are {code_bytes(width)} bytes each: code them so")
    return Coded(read_only(ids), read_only(bits), read_only(words(bits)), width)


@dataclass(frozen=True, slots=True, eq=False)
class Measured:
    distance: NDArray[np.uint16]
    held_ids: tuple[Ids, ...]
    sizes: tuple[int, ...]
    counted: Ids

    @property
    def total(self) -> int:
        return len(self.distance)

    @property
    def within(self) -> Ids:
        occupied = np.flatnonzero(self.counted)
        return np.cumsum(self.counted[: last(occupied) + 1 if len(occupied) else 1])

    def owners(self, at: Ids) -> Ids:
        return np.searchsorted(np.cumsum(self.sizes), at, side="right").astype(np.int64)

    def rows(self, at: Ids) -> Ids:
        starts = np.concatenate([np.zeros(1, np.int64), np.cumsum(self.sizes)[:-1]])
        return at - starts[self.owners(at)]

    def ids(self, at: Ids) -> Ids:
        if len(self.held_ids) == 1:
            return self.held_ids[0][at]
        owners, rows = self.owners(at), self.rows(at)
        found = np.empty(len(at), np.int64)
        for store in ints(np.unique(owners)):
            mine = np.equal(owners, store)
            found[mine] = self.held_ids[store][rows[mine]]
        return found


def _parts(sizes: Sequence[int]) -> list[ScanPart]:
    rows = max(PART_ROWS, -(-sum(sizes) // SCANNING))
    firsts = itertools.accumulate(sizes, initial=0)
    placed = [
        (store, range(start, min(start + rows, size)), first + start)
        for store, (size, first) in enumerate(zip(sizes, firsts, strict=False))
        for start in range(0, size, rows)
    ]
    return [ScanPart(number, *each) for number, each in enumerate(placed)]


def _measured(held: Sequence[Coded], query: Floats, spread: Spread) -> Measured:
    asked = code_of(query)
    sizes = tuple(len(each.ids) for each in held)
    distance = np.empty(sum(sizes), np.uint16)
    parts = _parts(sizes)
    counted = np.zeros((len(parts), 8 * WORD_BYTES * len(asked) + 1), np.int64)

    def scan(part: ScanPart) -> None:
        found = distances(held[part.store].words[part.rows.start : part.rows.stop], asked)
        distance[part.at : part.at + len(part.rows)] = found
        counted[part.number] = np.bincount(found, minlength=width_of(counted))

    spread(scan, parts)
    return Measured(distance, tuple(each.ids for each in held), sizes, counted.sum(axis=0))


def _candidates(found: Measured, at: Ids) -> Candidates:
    return Candidates(found.ids(at), found.owners(at), found.rows(at))


def _nearest_of(found: Measured, among: Ids | None, want: int) -> Candidates:
    distance = found.distance if among is None else found.distance[among]
    if len(distance) <= want:
        return _candidates(found, np.arange(len(distance)) if among is None else among)
    within = found.within if among is None else np.cumsum(np.bincount(distance))
    cutoff = int(np.searchsorted(within, want))
    close = np.flatnonzero(distance <= cutoff)
    at = close if among is None else among[close]
    return _candidates(found, at[np.lexsort((found.ids(at), distance[close]))[:want]])


def _let(found: Measured, chosen: Ids, allowed: Sequence[AllowedChunks | None]) -> Mask:
    let = np.ones(len(chosen), np.bool_)
    owners = found.owners(chosen)
    for store, each in enumerate(allowed):
        mine = np.flatnonzero(np.equal(owners, store))
        if each is not None and len(mine):
            let[mine] = each.checked(found.ids(chosen[mine]))
    return let


def _listed(held: Sequence[Coded], found: Measured, allowed: Sequence[AllowedChunks | None]) -> Ids:
    let = np.ones(found.total, np.bool_)
    start = 0
    for store, each in enumerate(allowed):
        size = found.sizes[store]
        if each is not None and size:
            let[start : start + size] = np.isin(held[store].ids, each.listed())
        start += size
    return np.flatnonzero(let)


def _passing(held: Sequence[Coded], found: Measured, want: int, allowed: Sequence[AllowedChunks | None]) -> Ids:
    within = found.within
    counts = ints(within)
    kept: list[Ids] = [np.zeros(0, np.int64)]
    passed, low, n = 0, 0, want
    while n <= SCAN_SHARE * found.total:
        high = int(np.searchsorted(within, n))
        band = np.flatnonzero(
            np.logical_and(np.greater_equal(found.distance, low), np.less_equal(found.distance, high))
        )
        let = _let(found, band, allowed)
        kept.append(band[let])
        passed, checked, low = passed + int(let.sum()), counts[high], high + 1
        if passed >= want or low >= len(within):
            return np.concatenate(kept)
        n = max(2 * checked, checked + int((want - passed) * checked / max(passed, 1) * MARGIN))
    return _listed(held, found, allowed)


def _shortlist(held: Sequence[Coded], query: Floats, want: int, reads: Reads) -> Candidates:
    found = _measured(held, query, reads.spread)
    allowed = reads.allowed
    among = _passing(held, found, want, allowed) if any(each is not None for each in allowed) else None
    return _nearest_of(found, among, want)


def _by_bits(held: Sequence[Coded], query: Floats, short: Candidates, want: int) -> Candidates:
    if len(short.ids) <= want:
        return short
    scores = np.empty(len(short.ids), np.float32)
    for store, each in enumerate(held):
        mine = np.equal(short.owners, store)
        if mine.any():
            scores[mine] = bit_scores(each.bits[short.rows[mine]], query)
    best = np.lexsort((short.ids, -scores))[:want]
    return Candidates(short.ids[best], short.owners[best], short.rows[best])


def _refuse(held: Sequence[Coded], query: Floats, k: int) -> None:
    if k < 0:
        raise ValueError(f"{k} nearest were asked for: ask for none or more")
    if query.ndim != 1 or not bool(np.isfinite(query).all()):
        raise ValueError("the query is not one vector of finite numbers: embed it again")
    if any(len(each.ids) and each.width != len(query) for each in held):
        raise ValueError(f"the query has {len(query):,} dimensions, unlike the vectors: embed it with their model")


def nearest(
    held: Sequence[Coded], query: Floats, k: int, reads: Reads, rescore: int = MOST_RESCORED
) -> tuple[Near, ...]:
    _refuse(held, query, k)
    if not k:
        return ()
    want = max(rescore, k)
    chosen = _by_bits(held, query, _shortlist(held, query, SHORTLIST * want, reads), want)
    found: list[Ids] = [np.zeros(0, np.int64)]
    scored: list[Floats] = [np.zeros(0, np.float32)]
    for store in ints(distinct(ascending(chosen.owners)).values):
        ids, rows = reads.exact(store, np.sort(chosen.ids[chosen.owners == store]))
        found.append(ids)
        scored.append(rows.astype(np.float32) @ query if len(ids) else np.zeros(0, np.float32))
    ids, scores = np.concatenate(found), np.concatenate(scored)
    order = np.lexsort((ids, -scores))[:k]
    paired = zip(ints(ids[order]), floats(scores[order]), strict=True)
    return tuple(Near(ChunkId(chunk), score) for chunk, score in paired)
