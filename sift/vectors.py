"""The stores' vectors in memory as one bit per dimension, its sign: millions of passages searched in a sixteenth of
the memory their fp16 vectors take, and without the GPU. `top` finds candidates by Hamming distance, among the codes of
every collection searched, and scores the best of them exactly, with the vectors the stores hold: over 3.2 million
arXiv abstracts, scoring the 4,000 nearest exactly found 99.6% of exact search's top 10, and a known-item MRR of
0.9528 against 0.9530 (docs/benchmarks.md).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RESCORE = 4000  # candidates scored exactly, at least
BLOCK = 1 << 15  # codes per thread at a time: their XOR, 3 MB at 768 dimensions, stays in the cache
THREADS = min(16, os.cpu_count() or 1)
FAR = np.iinfo(np.uint16).max  # the distance of a code left out


def codes(matrix):
    """The signs of vectors, one bit per dimension (positive is 1), packed in whole 8-byte words: uint8, a row each."""
    bits = np.packbits(np.asarray(matrix) > 0, axis=1)
    pad = -bits.shape[1] % 8
    return np.pad(bits, ((0, 0), (0, pad))) if pad else bits


def top(held, query, k, exact, allowed=None, excluded=None):
    """The k chunks nearest `query`, a normalised vector, as [(id, cosine)], best first, ties by id, among the codes
    `held` (a Codes per collection): of those `allowed` (ids; None for all) but those `excluded`. At least RESCORE of
    the nearest by Hamming distance, of all collections together, are scored exactly with `exact(ids)`, which gives
    the vectors the stores hold of those of the ids they still have, as (ids, fp16 rows); so a filter applies before
    the candidates are picked."""
    query = np.asarray(query, np.float32)
    bits, want = codes(query[None, :])[0], max(RESCORE, k)
    nearest = [c.nearest(bits, want, allowed, excluded) for c in held]
    ids = np.concatenate([np.empty(0, np.int64), *(ids for ids, _ in nearest)])
    distance = np.concatenate([np.empty(0, np.uint16), *(d for _, d in nearest)])
    if len(ids) > want:
        ids = ids[np.argpartition(distance, want - 1)[:want]]
    if not len(ids):
        return []
    found, rows = exact(np.sort(ids))
    scores = rows.astype(np.float32) @ query
    intact = np.isfinite(scores)  # a vector the model broke (NaN) has no cosine
    found, scores = found[intact], scores[intact]
    order = np.lexsort((found, -scores))[:k]
    return [(int(found[i]), float(scores[i])) for i in order]


class Codes:
    """The codes of the vectors of chunks `ids`, rows of `bits`, of one collection. Patched codes keep these as they
    are and add to them."""

    def __init__(self, ids, bits, *, _parts=None, _alive=None):
        self.ids = np.asarray(ids, np.int64)
        self._parts = _parts or [np.ascontiguousarray(bits)]  # rows of codes, in the order of `ids`
        self._alive = _alive  # which of `ids` still count; None: all

    def __len__(self):
        return len(self.ids) if self._alive is None else int(self._alive.sum())

    @property
    def width(self):
        """The bytes of each code."""
        return self._parts[0].shape[1]

    def nearest(self, query_bits, n, allowed=None, excluded=None):
        """The n chunks whose codes are nearest `query_bits` by Hamming distance, of those `allowed` (ids; None for
        all) but those `excluded`: (their ids, the distances), in no order."""
        distance = self._distances(query_bits, self._keep(allowed, excluded))
        rows = np.flatnonzero(distance < FAR)
        if len(rows) > n:
            rows = rows[np.argpartition(distance[rows], n - 1)[:n]]
        return self.ids[rows], distance[rows]

    def live(self):
        """The ids that count and their codes, by id."""
        rows = np.flatnonzero(self._alive) if self._alive is not None else np.arange(len(self.ids))
        rows = rows[np.argsort(self.ids[rows], kind="stable")]
        return self.ids[rows], np.concatenate(self._parts)[rows]

    def patched(self, removed, ids, bits):
        """These codes without those of the chunks `removed`, and with those of the chunks `ids`, rows of `bits`, which
        replace any these hold under the same ids, as of a chunk embedded again. The codes loaded
        stay as they are, masked; the added ones form one more part, rebuilt on every patch, small."""
        ids, base = np.asarray(ids, np.int64), self._parts[0]
        gone = np.concatenate([np.asarray(removed, np.int64), ids])
        alive = ~np.isin(self.ids, gone)
        if self._alive is not None:
            alive &= self._alive
        added = np.flatnonzero(alive[len(base) :])
        added_bits = self._parts[1][added] if len(self._parts) > 1 else base[:0]
        return Codes(
            np.concatenate([self.ids[: len(base)], self.ids[len(base) :][added], ids]),
            None,
            _parts=[base, np.concatenate([added_bits, np.asarray(bits, np.uint8)])],
            _alive=np.concatenate([alive[: len(base)], np.ones(len(added) + len(ids), bool)]),
        )

    def _keep(self, allowed, excluded):
        """Which of `ids` may be found, or None for all."""
        keep = self._alive
        if allowed is not None:
            inside = np.isin(self.ids, np.asarray(allowed, np.int64))
            keep = inside if keep is None else keep & inside
        if excluded is not None and len(excluded):
            outside = ~np.isin(self.ids, np.asarray(excluded, np.int64))
            keep = outside if keep is None else keep & outside
        return keep

    def _distances(self, query_bits, keep):
        """The Hamming distance of every code to the query's, FAR for those `keep` leaves out; in blocks, each in a
        thread (NumPy lets go of the GIL)."""
        word = query_bits.view(np.uint64)
        out, starts, at = np.empty(len(self.ids), np.uint16), [], 0
        for part in self._parts:
            words = part.view(np.uint64)
            starts += [(words, at, start) for start in range(0, len(words), BLOCK)]
            at += len(words)

        def block(item):
            words, at, start = item
            rows = words[start : start + BLOCK]
            np.sum(np.bitwise_count(rows ^ word), axis=1, dtype=np.uint16, out=out[at + start : at + start + len(rows)])

        if len(starts) > 1:
            with ThreadPoolExecutor(THREADS) as pool:
                list(pool.map(block, starts))
        else:
            for item in starts:
                block(item)
        if keep is not None:
            out[~keep] = FAR
        return out
