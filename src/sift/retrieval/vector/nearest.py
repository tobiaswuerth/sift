from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids, ascending, distinct, floats, ints
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

RESCORE = 1000
SHORTLIST = 10
CODES_AXES = 2

type Exact = Callable[[int, Ids], tuple[Ids, Halves]]


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


def _shortlist(held: Sequence[Coded], query: Floats, want: int) -> Candidates:
    asked = code_of(query)
    found = np.concatenate([np.zeros(0, np.uint16), *(distances(each.words, asked) for each in held)])
    ids = np.concatenate([np.zeros(0, np.int64), *(each.ids for each in held)])
    owners = np.repeat(np.arange(len(held)), [len(each.ids) for each in held])
    rows = np.concatenate([np.zeros(0, np.int64), *(np.arange(len(each.ids)) for each in held)])
    if len(found) > want:
        cutoff = int(np.searchsorted(np.cumsum(np.bincount(found)), want))
        close = np.flatnonzero(found <= cutoff)
        chosen = close[np.lexsort((ids[close], found[close]))[:want]]
        return Candidates(ids[chosen], owners[chosen], rows[chosen])
    return Candidates(ids, owners, rows)


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


def nearest(held: Sequence[Coded], query: Floats, k: int, exact: Exact, rescore: int = RESCORE) -> tuple[Near, ...]:
    _refuse(held, query, k)
    if not k:
        return ()
    want = max(rescore, k)
    chosen = _by_bits(held, query, _shortlist(held, query, SHORTLIST * want), want)
    found: list[Ids] = [np.zeros(0, np.int64)]
    scored: list[Floats] = [np.zeros(0, np.float32)]
    for store in ints(distinct(ascending(chosen.owners)).values):
        ids, rows = exact(store, np.sort(chosen.ids[chosen.owners == store]))
        found.append(ids)
        scored.append(rows.astype(np.float32) @ query if len(ids) else np.zeros(0, np.float32))
    ids, scores = np.concatenate(found), np.concatenate(scored)
    order = np.lexsort((ids, -scores))[:k]
    return tuple(Near(chunk, score) for chunk, score in zip(ints(ids[order]), floats(scores[order]), strict=True))


def within(held: Coded, allowed: Ids) -> Coded:
    keep = np.isin(held.ids, allowed)
    return coded(held.ids[keep], held.bits[keep], held.width)
