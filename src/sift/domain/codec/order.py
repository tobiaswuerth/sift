from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sift import _native
from sift.domain.arrays import Ids, integers, ints, whole
from sift.domain.codec.codes import Bits, Words, width_of, words

if TYPE_CHECKING:
    from numpy.random import Generator

LEVELS = (64, 64)
ROUNDS = 8
PER_CLUSTER = 300
SEED = 0
STORED_WORDS = np.dtype("<u8")


@dataclass(eq=False, frozen=True, slots=True)
class _Clustering:
    held: Words
    rng: "Generator"


def nearest(held: Words, centroids: Words) -> Ids:
    return integers(_native.nearest_centroids(held, centroids, width_of(centroids)))


def majority(held: Words, assigned: Ids, centroids: Words) -> Words:
    made = _native.majority_centroids(held, whole(assigned), centroids, width_of(centroids))
    return np.frombuffer(made, STORED_WORDS).astype(np.uint64).reshape(centroids.shape)


def _means(clustering: _Clustering, held: Words, k: int) -> tuple[Words, Ids]:
    rng, size = clustering.rng, PER_CLUSTER * k
    sample = held[rng.choice(len(held), size, replace=False)] if len(held) > size else held
    centroids = sample[rng.choice(len(sample), min(k, len(sample)), replace=False)].copy()
    for _ in range(ROUNDS):
        centroids = majority(sample, nearest(sample, centroids), centroids)
    return centroids, nearest(held, centroids)


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


def topical(ids: Ids, bits: Bits) -> Ids:
    leaf = np.zeros(len(ids), np.int64)
    if len(ids) > 1:
        clustering = _Clustering(words(bits), np.random.default_rng(SEED))
        for rows, code in _leaves(clustering, np.arange(len(ids)), 0, 0):
            leaf[rows] = code
    return ids[np.lexsort((ids, leaf))]
