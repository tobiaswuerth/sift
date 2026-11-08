"""The order of a collection's chunks in its keyword postings: topical, so that the chunks of a range of postings
(postings.RANGE_SHIFT) share their words, and the range's bounds are tight (see topk). In the order chunks are indexed,
a range mixes every topic: a question of a dozen common words bounded most ranges high, and pruning skipped little.

The order comes from the chunks' codes (see vectors), which the store keeps anyway: clusters of clusters (LEVELS), by
binary k-means, Hamming distance and each cluster's bits by majority, trained on a sample; each cluster next to the
one closest to it, and the chunks of a leaf in the order of their ids. Over 3.2 million arXiv abstracts, questions
scored a half to a twentieth as many ranges, the slowest in half the time (docs/benchmarks.md), and the postings took
8% less space."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

LEVELS = (64, 64)  # clusters per level: 4,096 leaves, a few ranges each in a collection of millions of chunks
ROUNDS = 8  # of k-means
PER_CLUSTER = 300  # codes k-means is trained on per cluster, at each node: training costs the same at any size
STEP = 65_536  # codes measured at a time, in a thread each


def topical(ids, bits):
    """`ids` in topical order, by `bits`, their codes (rows of uint8 in whole 64-bit words, as vectors.codes packs
    them): see the module."""
    ids = np.asarray(ids, np.int64)
    leaf = np.zeros(len(ids), np.int64)
    if len(ids) > 1:
        words = np.ascontiguousarray(bits).view(np.uint64)
        _cluster(words, np.arange(len(ids)), 0, 0, leaf, np.random.default_rng(0))
    return ids[np.lexsort((ids, leaf))]


def _cluster(words, rows, level, prefix, leaf, rng):
    """Number the leaves under the node of these rows, at this level, from prefix * LEVELS[level] on."""
    k = LEVELS[level]
    centroids, assign = _kmeans(words if len(rows) == len(words) else words[rows], k, rng)  # all: no copy of them
    rank = np.empty(len(centroids), np.int64)
    rank[_chain(centroids)] = np.arange(len(centroids))
    for c in range(len(centroids)):
        mine = rows[assign == c]
        code = prefix * k + rank[c]
        if level + 1 < len(LEVELS) and len(mine) > 1:
            _cluster(words, mine, level + 1, code, leaf, rng)
        else:
            leaf[mine] = code * np.prod(LEVELS[level + 1 :], dtype=np.int64)


def _kmeans(words, k, rng):
    """k centroids (codes as uint64 words), trained on a sample of `words`, and the nearest of them to every word."""
    size = PER_CLUSTER * k
    sample = words[rng.choice(len(words), size, replace=False)] if len(words) > size else words
    centroids = sample[rng.choice(len(sample), min(k, len(sample)), replace=False)].copy()
    for _ in range(ROUNDS):
        centroids = _majority(sample, _nearest(sample, centroids), centroids)
    return centroids, _nearest(words, centroids)


def _nearest(words, centroids):
    """The centroid nearest to every code by Hamming distance, the first of those as near, STEP codes at a time in
    threads: a part's distances alone are held."""
    out = np.empty(len(words), np.int64)

    def part(start):
        rows = words[start : start + STEP]
        distances = np.empty((len(rows), len(centroids)), np.uint16)
        for j, centroid in enumerate(centroids):
            distances[:, j] = np.bitwise_count(rows ^ centroid).sum(1)
        out[start : start + len(rows)] = distances.argmin(1)

    starts = range(0, len(words), STEP)
    if len(starts) > 1:
        with ThreadPoolExecutor() as pool:  # NumPy lets go of the GIL
            list(pool.map(part, starts))
    else:
        part(0)
    return out


def _majority(words, assign, centroids):
    """Each cluster's centroid as the majority of its codes' bits; a cluster left empty keeps its centroid."""
    order = np.argsort(assign, kind="stable")
    held, starts = np.unique(assign[order], return_index=True)
    bits = np.unpackbits(words[order].view(np.uint8), axis=1)
    ones = np.add.reduceat(bits, starts, dtype=np.uint32)
    counts = np.diff(np.r_[starts, len(order)])[:, None]
    centroids = centroids.copy()
    centroids[held] = np.packbits(ones * 2 > counts, axis=1).view(np.uint64)
    return centroids


def _chain(centroids):
    """The centroids in an order where each is next to one close to it: a greedy walk from the first."""
    distance = np.bitwise_count(centroids[:, None, :] ^ centroids[None, :, :]).sum(2)
    walk, left = [0], np.ones(len(centroids), bool)
    left[0] = False
    while left.any():
        near = np.where(left, distance[walk[-1]], np.iinfo(distance.dtype).max)
        walk.append(int(near.argmin()))
        left[walk[-1]] = False
    return np.array(walk)
