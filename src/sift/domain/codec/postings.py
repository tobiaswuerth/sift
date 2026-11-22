import itertools
import struct
import zlib
from collections.abc import Sequence
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from sift.domain.arrays import Ids, after, first, ints, largest, run_starts, sums_at, total

type Bytes = NDArray[np.uint8]

LEVEL = 6
RAW, PACKED = b"\x00", b"\x01"
PACKED_FROM = 64
BLOCK_HEAD = struct.Struct("<qI")
RANGES_HEAD = struct.Struct("<qI")
MOST_GAP = 1 << 32
GAP_BYTES = 4
LOW_BITS = 0x7F
MORE = 0x80
MOST_BYTES = 10


class Ranges(NamedTuple):
    ids: Ids
    starts: Ids
    ends: Ids
    best: Bytes


class Encoded(NamedTuple):
    postings: tuple[bytes, ...]
    positions: tuple[bytes, ...]


def _wrapped(body: bytes) -> bytes:
    return PACKED + zlib.compress(body, LEVEL) if len(body) >= PACKED_FROM else RAW + body


def _unwrapped(data: bytes) -> bytes:
    return zlib.decompress(data[1:]) if data[:1] == PACKED else data[1:]


def varints(values: Ids) -> tuple[bytes, list[int]]:
    sizes = np.ones(len(values), np.int64)
    for k in range(1, MOST_BYTES):
        sizes += values >= 1 << (7 * k)
    each = np.repeat(values, sizes)
    starts = np.cumsum(sizes) - sizes
    place = np.arange(len(each)) - np.repeat(starts, sizes)
    more = (place < np.repeat(sizes, sizes) - 1).astype(np.int64) * MORE
    written = (((each >> (7 * place)) & LOW_BITS) | more).astype(np.uint8).tobytes()
    return written, ints(after(0, np.cumsum(sizes)))


def numbers(data: bytes) -> Ids:
    raw = np.frombuffer(data, np.uint8)
    if not len(raw):
        return np.zeros(0, np.int64)
    starts = np.flatnonzero(np.concatenate((np.ones(1, np.bool_), raw[:-1] < MORE)))
    place = np.arange(len(raw)) - np.repeat(starts, np.diff(np.append(starts, len(raw))))
    return sums_at((raw & LOW_BITS).astype(np.int64) << (7 * place), starts)


def _gaps(ids: Ids, bounds: Sequence[int]) -> bytes:
    gaps = np.diff(ids)
    inside = np.ones(len(gaps), np.bool_)
    inside[np.array(bounds[1:-1], np.int64) - 1] = False
    if len(gaps) and inside.any() and (gaps[inside].min() < 1 or largest(gaps[inside]) >= MOST_GAP):
        raise ValueError("ids not ascending, or more than 2**32 apart: more than a block holds")
    return gaps.astype(np.uint32).tobytes()


def _position_gaps(positions: Ids, counts: Ids) -> Ids:
    starts = np.cumsum(counts) - counts
    gaps = positions.copy()
    gaps[1:] -= positions[:-1]
    gaps[starts] = positions[starts]
    return gaps


def encoded(ids: Ids, counts: Ids, positions: Ids, bounds: Sequence[int]) -> Encoded:
    gaps = _gaps(ids, bounds)
    counted, at_count = varints(counts)
    placed, at_place = varints(_position_gaps(positions, counts))
    by_posting = ints(after(0, np.cumsum(counts)))
    listed = ints(ids)
    postings: list[bytes] = []
    held: list[bytes] = []
    for start, end in itertools.pairwise(bounds):
        head = BLOCK_HEAD.pack(listed[start], end - start)
        body = gaps[GAP_BYTES * start : GAP_BYTES * (end - 1)] + counted[at_count[start] : at_count[end]]
        postings.append(_wrapped(head + body))
        held.append(_wrapped(placed[at_place[by_posting[start]] : at_place[by_posting[end]]]))
    return Encoded(tuple(postings), tuple(held))


def blocked(data: bytes) -> tuple[Ids, Ids]:
    raw = _unwrapped(data)
    head, n = cast("tuple[int, int]", BLOCK_HEAD.unpack_from(raw))
    at = BLOCK_HEAD.size
    gaps = np.frombuffer(raw, np.uint32, n - 1, at)
    return _ascending(head, gaps), numbers(raw[at + GAP_BYTES * (n - 1) :])


def blocked_all(each: Sequence[bytes]) -> tuple[Ids, Ids, list[int]]:
    starts: list[int] = []
    sizes: list[int] = []
    gaps: list[bytes] = []
    counts: list[bytes] = []
    for data in each:
        raw = _unwrapped(data)
        head, n = cast("tuple[int, int]", BLOCK_HEAD.unpack_from(raw))
        split = BLOCK_HEAD.size + GAP_BYTES * (n - 1)
        starts.append(head)
        sizes.append(n)
        gaps.append(raw[BLOCK_HEAD.size : split])
        counts.append(raw[split:])
    held = np.array(sizes, np.int64)
    firsts_at = np.cumsum(held) - held
    steps = np.empty(total(held), np.int64)
    inside = np.ones(len(steps), np.bool_)
    inside[firsts_at] = False
    steps[inside] = np.frombuffer(b"".join(gaps), np.uint32)
    steps[firsts_at] = starts
    summed = np.cumsum(steps)
    ids = summed - np.repeat(summed[firsts_at] - steps[firsts_at], held)
    return ids, numbers(b"".join(counts)), ints(after(0, np.cumsum(held)))


def positions_of(data: bytes, counts: Ids) -> Ids:
    gaps = numbers(_unwrapped(data))
    summed = np.cumsum(gaps)
    starts = np.cumsum(counts) - counts
    return summed - np.repeat(summed[starts] - gaps[starts], counts)


def _ascending(start: int, gaps: NDArray[np.uint32]) -> Ids:
    return start + after(0, np.cumsum(gaps, dtype=np.int64))


def ranged(ids: Ids, counts: Ids, best: Bytes) -> bytes:
    head = RANGES_HEAD.pack(first(ids), len(ids))
    gaps = np.diff(ids).astype(np.uint32).tobytes()
    return zlib.compress(head + gaps + (counts - 1).astype(np.uint8).tobytes() + best.astype(np.uint8).tobytes(), LEVEL)


def ranges_of(data: bytes) -> Ranges:
    raw = zlib.decompress(data)
    head, n = cast("tuple[int, int]", RANGES_HEAD.unpack_from(raw))
    at = RANGES_HEAD.size
    found = _ascending(head, np.frombuffer(raw, np.uint32, n - 1, at))
    at += GAP_BYTES * (n - 1)
    ends = np.cumsum(np.frombuffer(raw, np.uint8, n, at).astype(np.int64) + 1)
    best = np.frombuffer(raw, np.uint8, n, at + n).copy()
    return Ranges(found, after(0, ends[:-1]), ends, best)


def with_lengths(ids: Ids, lengths: Ids, shift: int) -> dict[int, bytes]:
    blocks = ids >> shift
    cuts = ints(np.append(run_starts(blocks), len(ids)))
    written: dict[int, bytes] = {}
    for start, end in itertools.pairwise(cuts):
        dense = np.zeros(1 << shift, np.uint32)
        dense[ids[start:end] & ((1 << shift) - 1)] = lengths[start:end]
        written[first(blocks[start:])] = zlib.compress(dense.tobytes(), LEVEL)
    return written


def lengths_of(data: bytes) -> Ids:
    return np.frombuffer(zlib.decompress(data), np.uint32).astype(np.int64)
