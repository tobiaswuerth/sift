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
from sift.retrieval.allowed import Allowed

RESCORE = 1000
SHORTLIST = 10
CODES_AXES = 2
SCAN_SHARE = 0.2
MARGIN = 1.25

type Exact = Callable[[int, Ids], tuple[Ids, Halves]]


@dataclass(frozen=True, slots=True)
class Reads:
    exact: Exact
    allowed: Sequence[Allowed | None] = ()


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


@dataclass(frozen=True, slots=True)
class Measured:
    distance: NDArray[np.uint16]
    ids: Ids
    owners: Ids
    rows: Ids


def _measured(held: Sequence[Coded], query: Floats) -> Measured:
    asked = code_of(query)
    return Measured(
        np.concatenate([np.zeros(0, np.uint16), *(distances(each.words, asked) for each in held)]),
        np.concatenate([np.zeros(0, np.int64), *(each.ids for each in held)]),
        np.repeat(np.arange(len(held)), [len(each.ids) for each in held]),
        np.concatenate([np.zeros(0, np.int64), *(np.arange(len(each.ids)) for each in held)]),
    )


def _nearest_of(found: Measured, among: Ids, want: int) -> Candidates:
    distance, ids = found.distance[among], found.ids[among]
    if len(among) > want:
        cutoff = int(np.searchsorted(np.cumsum(np.bincount(distance)), want))
        close = np.flatnonzero(distance <= cutoff)
        among = among[close[np.lexsort((ids[close], distance[close]))[:want]]]
    return Candidates(found.ids[among], found.owners[among], found.rows[among])


def _let(found: Measured, chosen: Ids, allowed: Sequence[Allowed | None]) -> Mask:
    let = np.ones(len(chosen), np.bool_)
    for store, each in enumerate(allowed):
        mine = np.flatnonzero(np.equal(found.owners[chosen], store))
        if each is not None and len(mine):
            let[mine] = each.checked(found.ids[chosen[mine]])
    return let


def _listed(held: Sequence[Coded], found: Measured, allowed: Sequence[Allowed | None]) -> Ids:
    let = np.ones(len(found.ids), np.bool_)
    for store, each in enumerate(allowed):
        if each is not None:
            mine = np.equal(found.owners, store)
            let[mine] = np.isin(held[store].ids, each.listed())
    return np.flatnonzero(let)


def _passing(held: Sequence[Coded], found: Measured, want: int, allowed: Sequence[Allowed | None]) -> Ids:
    within = np.cumsum(np.bincount(found.distance, minlength=1))
    counts = ints(within)
    kept: list[Ids] = [np.zeros(0, np.int64)]
    passed, low, n = 0, 0, want
    while n <= SCAN_SHARE * len(found.ids):
        high = min(int(np.searchsorted(within, n)), len(within) - 1)
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


def _shortlist(held: Sequence[Coded], query: Floats, want: int, allowed: Sequence[Allowed | None]) -> Candidates:
    found = _measured(held, query)
    every = np.arange(len(found.ids))
    among = _passing(held, found, want, allowed) if any(each is not None for each in allowed) else every
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
