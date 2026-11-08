"""The stores' vectors in memory as one bit per dimension, its sign: millions of passages searched in a sixteenth of
the memory their fp16 vectors take, and without the GPU. `top` finds candidates by Hamming distance, among the codes of
every collection searched, ranks the nearest again by the query's own floats against their bits, and scores the best of
those exactly, with the vectors the stores hold: over 3.2 million arXiv abstracts, 1,000 of the 10,000 nearest scored
exactly found 99.75% of exact search's top 10 and its known-item MRR, 0.9530 (docs/benchmarks.md).
"""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RESCORE = 1000  # candidates scored exactly, at least: reading their vectors is most of a search's time
SHORTLIST = 10  # times as many, the nearest by Hamming distance, ranked by the query's floats against their bits
BLOCK = 1 << 15  # codes per thread at a time: their XOR, 3 MB at 768 dimensions, stays in the cache
THREADS = min(16, os.cpu_count() or 1)
FAR = np.iinfo(np.uint16).max  # the distance of a code left out
SCAN_SHARE = 0.2  # of the codes searched: checking more candidates against a filter costs more than listing all it
# lets through (arXiv, 3.2M chunks: 4-5 us a check against 6.2 s a listing, warm; cold, 90 us against 56 s)
MARGIN = 1.25  # candidates checked beyond those that, at the rate let through so far, would be enough
_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32)  # each byte's bits


def codes(matrix):
    """The signs of vectors, one bit per dimension (positive is 1), packed in whole 8-byte words: uint8, a row each."""
    bits = np.packbits(np.asarray(matrix) > 0, axis=1)
    pad = -bits.shape[1] % 8
    return np.pad(bits, ((0, 0), (0, pad))) if pad else bits


class Filter(ABC):
    """Which chunks a search may find (see top), asked either way."""

    @abstractmethod
    def check(self, ids):
        """Which of these chunks it lets through, as a boolean array: a lookup each."""

    @abstractmethod
    def listed(self):
        """The ids of all the chunks it lets through: a pass over every document."""


def top(held, query, k, exact, allowed=None, excluded=None):
    """The k chunks nearest `query`, a normalised vector, as [(id, cosine)], best first, ties by id, among the codes
    `held` (a Codes per collection): of those `allowed` (ids, or a Filter; None for all) but those `excluded`. Of the
    SHORTLIST times RESCORE nearest by Hamming distance, of all collections together, the RESCORE best by the query's
    floats against their bits are scored exactly with `exact(ids)`, which gives the vectors the stores hold of those
    of the ids they still have, as (ids, fp16 rows); so a filter applies before the candidates are picked. A Filter is
    asked about the nearest candidates, a batch at a time, until enough pass, or, if it lets too few through for that,
    for all it lets through."""
    query = np.asarray(query, np.float32)
    bits, want = codes(query[None, :])[0], max(RESCORE, k)
    if isinstance(allowed, Filter):
        found = _checked(held, bits, SHORTLIST * want, allowed, excluded)
        if found is None:
            found = _nearest(held, bits, SHORTLIST * want, allowed.listed(), excluded)
    else:
        found = _nearest(held, bits, SHORTLIST * want, allowed, excluded)
    ids = _best_by_floats(held, query, found, want)
    if not len(ids):
        return []
    found, rows = exact(np.sort(ids))
    scores = rows.astype(np.float32) @ query
    intact = np.isfinite(scores)  # a vector the model broke (NaN) has no cosine
    found, scores = found[intact], scores[intact]
    order = np.lexsort((found, -scores))[:k]
    return [(int(found[i]), float(scores[i])) for i in order]


def _nearest(held, bits, want, allowed, excluded):
    """The `want` nearest codes of all collections, ties by id, of those `allowed` (ids; None for all) but those
    `excluded`, as _between gives them."""
    distances = [c.distances(bits, allowed, excluded) for c in held]
    within = _within(distances)
    high = int(np.searchsorted(within, min(want, within[-1])))  # the codes up to this distance are enough, or all
    return _first(_between(held, distances, 0, high), want)


def _checked(held, bits, want, allowed, excluded):
    """The `want` nearest codes of all collections, ties by id, that the Filter `allowed` lets through, but those
    `excluded`, as _between gives them: the nearest checked first, all codes at a distance together, more each time,
    as many as would be enough at the rate let through so far. None once that would check more than SCAN_SHARE of the
    codes."""
    distances = [c.distances(bits, excluded=excluded) for c in held]
    within = _within(distances)
    passed, found, low, n = [], 0, 0, want  # codes from distance `low` on are unchecked
    while n <= SCAN_SHARE * within[-1]:
        high = int(np.searchsorted(within, n))  # the codes up to this distance are at least n
        candidates = _between(held, distances, low, high)
        let = np.asarray(allowed.check(candidates[0]), bool)
        passed.append(tuple(part[let] for part in candidates))
        found, checked, low = found + int(let.sum()), int(within[high]), high + 1
        if found >= want:
            return _first(tuple(np.concatenate(part) for part in zip(*passed)), want)
        n = max(2 * checked, checked + int((want - found) * checked / max(found, 1) * MARGIN))
    return None


def _within(distances):
    """How many codes are within each distance, by distance, of all collections: those left out (FAR) not counted."""
    held = sum((np.bincount(d, minlength=FAR + 1)[:FAR] for d in distances), np.zeros(FAR, np.int64))
    return np.cumsum(held)


def _between(held, distances, low, high):
    """The codes of all collections at distances from low to high: (their ids, distances, which of `held` holds each,
    and where in it)."""
    rows = [np.flatnonzero((d >= low) & (d <= high)) for d in distances]
    return (
        np.concatenate([np.empty(0, np.int64), *(c.ids[r] for c, r in zip(held, rows))]),
        np.concatenate([np.empty(0, np.uint16), *(d[r] for d, r in zip(distances, rows))]),
        np.concatenate([np.empty(0, np.int64), *(np.full(len(r), i) for i, r in enumerate(rows))]),
        np.concatenate([np.empty(0, np.int64), *rows]),
    )


def _first(candidates, n):
    """The n nearest of candidates as _between gives them, ties by id."""
    ids, distance, *_ = candidates
    pick = np.lexsort((ids, distance))[:n]
    return tuple(part[pick] for part in candidates)


def _best_by_floats(held, query, candidates, n):
    """The ids of the n best candidates (as _between gives them) by the query's floats against their bits, ties by
    id: q . (2b - 1) = 2 q . b - sum(q), and per byte of a code, q . b is a sum looked up for its 256 values."""
    ids, _, owner, rows = candidates
    if len(ids) <= n:
        return ids
    width = held[0].width
    bits = np.empty((len(ids), width), np.uint8)
    for i, c in enumerate(held):
        mine = owner == i
        bits[mine] = c.rows(rows[mine])
    floats = np.zeros(8 * width, np.float32)
    floats[: len(query)] = query
    table = floats.reshape(width, 8) @ _BITS.T  # (byte of a code, its value): the query's floats where its bits are 1
    return ids[np.lexsort((ids, -table[np.arange(width), bits].sum(1)))[:n]]


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

    def rows(self, at):
        """The codes at these places in `ids`."""
        out, start = np.empty((len(at), self.width), np.uint8), 0
        for part in self._parts:
            mine = (at >= start) & (at < start + len(part))
            out[mine] = part[at[mine] - start]
            start += len(part)
        return out

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

    def distances(self, query_bits, allowed=None, excluded=None):
        """The Hamming distance of every code to the query's, by `ids`: FAR for those that don't count, those not
        `allowed` (ids; None for all) and those `excluded`; in blocks, each in a thread (NumPy lets go of the GIL)."""
        keep = self._keep(allowed, excluded)
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
