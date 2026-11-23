from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from sift.domain.arrays import Ids, Mask, ascending, distinct, floats, ints
from sift.domain.codec.codes import (
    MOST_BITS,
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

RESCORE = 1000
SHORTLIST = 10
CODES_AXES = 2
SCAN_SHARE = 0.2
MARGIN = 1.25

type ExactRead = Callable[[int, Ids], tuple[Ids, Halves]]


@dataclass(frozen=True, slots=True)
class Reads:
    exact: ExactRead
    allowed: Sequence[AllowedChunks | None] = ()


@dataclass(frozen=True, slots=True)
class Coded:
    ids: Ids
    bits: Bits
    words: Words
    width: int


class Near(NamedTuple):
    chunk: int
    score: float


@dataclass(frozen=True, slots=True)
class Candidates:
    ids: Ids
    owners: Ids
    rows: Ids


def coded(ids: Ids, bits: Bits, width: int) -> Coded:
    if bits.dtype != np.uint8 or bits.ndim != CODES_AXES or len(bits) != len(ids):
        raise ValueError("codes are a uint8 matrix of a row a chunk: code them so")
    if len(ids) and not (0 < width <= MOST_BITS and width_of(bits) == code_bytes(width)):
        raise ValueError(f"codes of {width:,} dimensions are {code_bytes(width)} bytes each: code them so")
    return Coded(ids, bits, words(bits), width)


@dataclass(frozen=True, slots=True, eq=False)
class Measured:
    distance: NDArray[np.uint16]
    ids: Ids
    sizes: tuple[int, ...]

    def owners(self, at: Ids) -> Ids:
        return np.searchsorted(np.cumsum(self.sizes), at, side="right").astype(np.int64)

    def rows(self, at: Ids) -> Ids:
        starts = np.concatenate([np.zeros(1, np.int64), np.cumsum(self.sizes)[:-1]])
        return at - starts[self.owners(at)]


def _joined[T: np.generic](parts: list[NDArray[T]], empty: NDArray[T]) -> NDArray[T]:
    return parts[0] if len(parts) == 1 else np.concatenate([empty, *parts])


def _measured(held: Sequence[Coded], query: Floats) -> Measured:
    asked = code_of(query)
    return Measured(
        _joined([distances(each.words, asked) for each in held], np.zeros(0, np.uint16)),
        _joined([each.ids for each in held], np.zeros(0, np.int64)),
        tuple(len(each.ids) for each in held),
    )


def _candidates(found: Measured, at: Ids) -> Candidates:
    return Candidates(found.ids[at], found.owners(at), found.rows(at))


def _nearest_of(found: Measured, among: Ids | None, want: int) -> Candidates:
    distance = found.distance if among is None else found.distance[among]
    if len(distance) <= want:
        return _candidates(found, np.arange(len(distance)) if among is None else among)
    cutoff = int(np.searchsorted(np.cumsum(np.bincount(distance)), want))
    close = np.flatnonzero(distance <= cutoff)
    at = close if among is None else among[close]
    return _candidates(found, at[np.lexsort((found.ids[at], distance[close]))[:want]])


def _let(found: Measured, chosen: Ids, allowed: Sequence[AllowedChunks | None]) -> Mask:
    let = np.ones(len(chosen), np.bool_)
    owners = found.owners(chosen)
    for store, each in enumerate(allowed):
        mine = np.flatnonzero(np.equal(owners, store))
        if each is not None and len(mine):
            let[mine] = each.checked(found.ids[chosen[mine]])
    return let


def _listed(held: Sequence[Coded], found: Measured, allowed: Sequence[AllowedChunks | None]) -> Ids:
    let = np.ones(len(found.ids), np.bool_)
    start = 0
    for store, each in enumerate(allowed):
        size = found.sizes[store]
        if each is not None:
            let[start : start + size] = np.isin(held[store].ids, each.listed())
        start += size
    return np.flatnonzero(let)


def _passing(held: Sequence[Coded], found: Measured, want: int, allowed: Sequence[AllowedChunks | None]) -> Ids:
    within = np.cumsum(np.bincount(found.distance, minlength=1))
    counts = ints(within)
    kept: list[Ids] = [np.zeros(0, np.int64)]
    passed, low, n = 0, 0, want
    while n <= SCAN_SHARE * len(found.ids):
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


def _shortlist(held: Sequence[Coded], query: Floats, want: int, allowed: Sequence[AllowedChunks | None]) -> Candidates:
    found = _measured(held, query)
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


def nearest(held: Sequence[Coded], query: Floats, k: int, reads: Reads, rescore: int = RESCORE) -> tuple[Near, ...]:
    _refuse(held, query, k)
    if not k:
        return ()
    want = max(rescore, k)
    chosen = _by_bits(held, query, _shortlist(held, query, SHORTLIST * want, reads.allowed), want)
    found: list[Ids] = [np.zeros(0, np.int64)]
    scored: list[Floats] = [np.zeros(0, np.float32)]
    for store in ints(distinct(ascending(chosen.owners)).values):
        ids, rows = reads.exact(store, np.sort(chosen.ids[chosen.owners == store]))
        found.append(ids)
        scored.append(rows.astype(np.float32) @ query if len(ids) else np.zeros(0, np.float32))
    ids, scores = np.concatenate(found), np.concatenate(scored)
    order = np.lexsort((ids, -scores))[:k]
    return tuple(Near(chunk, score) for chunk, score in zip(ints(ids[order]), floats(scores[order]), strict=True))
