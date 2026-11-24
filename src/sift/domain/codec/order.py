from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sift.domain.arrays import Ids, ints
from sift.domain.codec.codes import Bits, Distances, Words, words

if TYPE_CHECKING:
    from numpy.random import Generator

LEVELS = (64, 64)
ROUNDS = 8
PER_CLUSTER = 300
STEP = 65_536
SEED = 0


@dataclass(frozen=True, slots=True)
class _Clustering:
    held: Words
    rng: "Generator"
    step: int


def _nearest(held: Words, centroids: Words, step: int) -> Ids:
    found = np.empty(len(held), np.int64)
    for start in range(0, len(held), step):
        rows = held[start : start + step]
        apart: Distances = np.empty((len(rows), len(centroids)), np.uint16)
        for at in range(len(centroids)):
            apart[:, at] = np.bitwise_count(rows ^ centroids[at : at + 1]).sum(axis=1, dtype=np.uint16)
        found[start : start + len(rows)] = apart.argmin(axis=1)
    return found


def _majority(held: Words, assigned: Ids, centroids: Words) -> Words:
    order = np.argsort(assigned, kind="stable")
    clusters, starts = np.unique(assigned[order], return_index=True)
    bits = np.unpackbits(held[order].view(np.uint8), axis=1)
    ones = np.add.reduceat(bits, starts, axis=0, dtype=np.uint32)
    sizes = np.diff(np.append(starts, len(order)))[:, None]
    made = centroids.copy()
    made[clusters] = np.packbits(ones * 2 > sizes, axis=1).view(np.uint64)
    return made


def _means(clustering: _Clustering, held: Words, k: int) -> tuple[Words, Ids]:
    rng, step, size = clustering.rng, clustering.step, PER_CLUSTER * k
    sample = held[rng.choice(len(held), size, replace=False)] if len(held) > size else held
    centroids = sample[rng.choice(len(sample), min(k, len(sample)), replace=False)].copy()
    for _ in range(ROUNDS):
        centroids = _majority(sample, _nearest(sample, centroids, step), centroids)
    return centroids, _nearest(held, centroids, step)


def _chain(centroids: Words) -> Ids:
    apart = np.bitwise_count(centroids[:, None, :] ^ centroids[None, :, :]).sum(axis=2)
    walk, left = [0], np.ones(len(centroids), np.bool_)
    left[0] = False
    while left.any():
        near = np.where(left, apart[walk[-1] : walk[-1] + 1].reshape(-1), np.iinfo(apart.dtype).max)
        walk.append(int(near.argmin()))
        left[walk[-1]] = False
    return np.array(walk, np.int64)


def _leaves(clustering: _Clustering, rows: Ids, level: int, prefix: int) -> Iterator[tuple[Ids, int]]:
    k, held = LEVELS[level], clustering.held
    centroids, assigned = _means(clustering, held if len(rows) == len(held) else held[rows], k)
    rank = np.empty(len(centroids), np.int64)
    rank[_chain(centroids)] = np.arange(len(centroids))
    below = int(np.prod(LEVELS[level + 1 :], dtype=np.int64))
    for cluster, ranked in enumerate(ints(rank)):
        mine = rows[assigned == cluster]
        code = prefix * k + ranked
        if level + 1 < len(LEVELS) and len(mine) > 1:
            yield from _leaves(clustering, mine, level + 1, code)
        else:
            yield mine, code * below


def topical(ids: Ids, bits: Bits, step: int = STEP) -> Ids:
    leaf = np.zeros(len(ids), np.int64)
    if len(ids) > 1:
        clustering = _Clustering(words(bits), np.random.default_rng(SEED), step)
        for rows, code in _leaves(clustering, np.arange(len(ids)), 0, 0):
            leaf[rows] = code
    return ids[np.lexsort((ids, leaf))]
