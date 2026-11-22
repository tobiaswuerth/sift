from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids, floats, ints
from sift.domain.codec.codes import Bits, Floats, Words, bit_scores, code_of, distances, words
from sift.storage.vectors import Halves

RESCORE = 1000
SHORTLIST = 10

type Exact = Callable[[int, Ids], tuple[Ids, Halves]]


@dataclass(frozen=True, slots=True)
class Coded:
    ids: Ids
    bits: Bits
    words: Words


class Near(NamedTuple):
    chunk: int
    score: float


@dataclass(frozen=True, slots=True)
class Candidates:
    ids: Ids
    owners: Ids
    rows: Ids


def coded(ids: Ids, bits: Bits) -> Coded:
    return Coded(ids, bits, words(bits))


def _shortlist(held: Sequence[Coded], query: Floats, want: int) -> Candidates:
    asked = code_of(query)
    measured = [distances(each.words, asked) for each in held]
    found = np.concatenate([np.zeros(0, np.uint16), *measured])
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
        scores[mine] = bit_scores(each.bits[short.rows[mine]], query)
    best = np.lexsort((short.ids, -scores))[:want]
    return Candidates(short.ids[best], short.owners[best], short.rows[best])


def nearest(held: Sequence[Coded], query: Floats, k: int, exact: Exact, rescore: int = RESCORE) -> tuple[Near, ...]:
    if not sum(len(each.ids) for each in held):
        return ()
    want = max(rescore, k)
    chosen = _by_bits(held, query, _shortlist(held, query, SHORTLIST * want), want)
    found: list[Ids] = []
    scored: list[Floats] = []
    for store in range(len(held)):
        ids, rows = exact(store, np.sort(chosen.ids[chosen.owners == store]))
        found.append(ids)
        scored.append(rows.astype(np.float32) @ query if len(ids) else np.zeros(0, np.float32))
    ids, scores = np.concatenate(found), np.concatenate(scored)
    intact = np.isfinite(scores)
    ids, scores = ids[intact], scores[intact]
    order = np.lexsort((ids, -scores))[:k]
    return tuple(Near(chunk, score) for chunk, score in zip(ints(ids[order]), floats(scores[order]), strict=True))
