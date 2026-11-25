import itertools
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from sift.domain.arrays import STORED_U32, Ids, after, first, ints, largest, last, run_starts, sums_at, total
from sift.domain.errors import DamagedError

type Bytes = NDArray[np.uint8]

LEVEL = 6
RAW, PACKED = b"\x00", b"\x01"
PACKED_FROM = 64
BLOCK_HEAD = struct.Struct("<qI")
RANGES_HEAD = struct.Struct("<qI")
MOST_GAP = 1 << 32
MOST_POSITION = 1 << 32
GAP_BYTES = 4
LENGTH_BYTES = 4
COUNT_BYTES = 1
BEST_BYTES = 1
LOW_BITS = 0x7F
MORE = 0x80
MOST_BYTES = 9
MOST_INFLATED_BYTES = 64 << 20


def _sound(*, holds: bool, problem: str) -> None:
    if not holds:
        raise DamagedError(problem)


@dataclass(eq=False, frozen=True, slots=True)
class Ranges:
    ids: Ids
    starts: Ids
    ends: Ids
    best: Bytes


class Encoded(NamedTuple):
    postings: tuple[bytes, ...]
    positions: tuple[bytes, ...]


def _wrapped(body: bytes) -> bytes:
    return PACKED + zlib.compress(body, LEVEL) if len(body) >= PACKED_FROM else RAW + body


def _inflated(data: bytes) -> bytes:
    unpacking = zlib.decompressobj()
    try:
        found = unpacking.decompress(data, MOST_INFLATED_BYTES + 1)
    except zlib.error as error:
        raise DamagedError(f"a packed list does not unpack: {error}") from None
    _sound(
        holds=unpacking.eof, problem=f"a packed list unpacks to more than {MOST_INFLATED_BYTES:,} bytes, or ends early"
    )
    return found


def _inflated_to(data: bytes, size: int) -> bytes:
    unpacking = zlib.decompressobj()
    try:
        found = unpacking.decompress(data, size + 1)
    except zlib.error as error:
        raise DamagedError(f"a packed list does not unpack: {error}") from None
    _sound(holds=len(found) == size and unpacking.eof, problem=f"a packed list unpacks to other than {size:,} bytes")
    return found


def _unwrapped(data: bytes) -> bytes:
    _sound(holds=data[:1] in {RAW, PACKED}, problem="a list starts with an unknown flag byte")
    return _inflated(data[1:]) if data[:1] == PACKED else data[1:]


def varints(values: Ids) -> tuple[bytes, Ids]:
    sizes = np.ones(len(values), np.int64)
    for k in range(1, MOST_BYTES):
        sizes += values >= 1 << (7 * k)
    each = np.repeat(values, sizes)
    starts = np.cumsum(sizes) - sizes
    place = np.arange(len(each)) - np.repeat(starts, sizes)
    more = (place < np.repeat(sizes, sizes) - 1).astype(np.int64) * MORE
    written = (((each >> (7 * place)) & LOW_BITS) | more).astype(np.uint8).tobytes()
    return written, after(0, np.cumsum(sizes))


def numbers(data: bytes) -> Ids:
    raw = np.frombuffer(data, np.uint8)
    if not len(raw):
        return np.zeros(0, np.int64)
    _sound(holds=last(raw) < MORE, problem="a number runs past the end of its list")
    starts = np.flatnonzero(np.concatenate((np.ones(1, np.bool_), raw[:-1] < MORE))).astype(np.int64)
    sizes = np.diff(np.append(starts, len(raw)))
    _sound(holds=largest(sizes) <= MOST_BYTES, problem="a number takes more than nine bytes")
    place = np.arange(len(raw)) - np.repeat(starts, sizes)
    return sums_at((raw & LOW_BITS).astype(np.int64) << (7 * place), starts)


def _gaps(ids: Ids, bounds: Sequence[int]) -> bytes:
    gaps = np.diff(ids)
    inside = np.ones(len(gaps), np.bool_)
    inside[np.array(bounds[1:-1], np.int64) - 1] = False
    if len(gaps) and inside.any() and (gaps[inside].min() < 1 or largest(gaps[inside]) >= MOST_GAP):
        raise ValueError("ids not ascending, or more than 2**32 apart: more than a block holds")
    return gaps.astype(STORED_U32).tobytes()


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
    cut = np.array(bounds, np.int64)
    counts_at, places_at = ints(at_count[cut]), ints(at_place[after(0, np.cumsum(counts))[cut]])
    firsts_of = ints(ids[cut[:-1]])
    postings: list[bytes] = []
    held: list[bytes] = []
    for n, (start, end) in enumerate(itertools.pairwise(bounds)):
        head = BLOCK_HEAD.pack(firsts_of[n], end - start)
        body = gaps[GAP_BYTES * start : GAP_BYTES * (end - 1)] + counted[counts_at[n] : counts_at[n + 1]]
        postings.append(_wrapped(head + body))
        held.append(_wrapped(placed[places_at[n] : places_at[n + 1]]))
    return Encoded(tuple(postings), tuple(held))


def blocked(data: bytes) -> tuple[Ids, Ids]:
    places, counts, _ = blocked_all([data])
    return places, counts


def blocked_all(each: Sequence[bytes]) -> tuple[Ids, Ids, list[int]]:
    starts: list[int] = []
    sizes: list[int] = []
    gaps: list[bytes] = []
    counts: list[bytes] = []
    for data in each:
        raw = _unwrapped(data)
        _sound(holds=len(raw) >= BLOCK_HEAD.size, problem="a block is shorter than its head")
        head, n = cast("tuple[int, int]", BLOCK_HEAD.unpack_from(raw))
        split = BLOCK_HEAD.size + GAP_BYTES * (n - 1)
        _sound(holds=n >= 1 and split <= len(raw), problem="a block holds fewer ids than its head says")
        starts.append(head)
        sizes.append(n)
        gaps.append(raw[BLOCK_HEAD.size : split])
        counts.append(raw[split:])
    held = np.array(sizes, np.int64)
    firsts_at = np.cumsum(held) - held
    steps = np.empty(total(held), np.int64)
    inside = np.ones(len(steps), np.bool_)
    inside[firsts_at] = False
    steps[inside] = np.frombuffer(b"".join(gaps), STORED_U32)
    _sound(holds=bool((steps[inside] >= 1).all()), problem="a block holds ids out of order")
    steps[firsts_at] = starts
    summed = np.cumsum(steps)
    ids = summed - np.repeat(summed[firsts_at] - steps[firsts_at], held)
    counted = numbers(b"".join(counts))
    _sound(holds=len(counted) == len(ids), problem="a block holds counts other than its ids")
    _sound(holds=not len(counted) or int(counted.min()) >= 1, problem="a block holds a count below one")
    return ids, counted, ints(after(0, np.cumsum(held)))


def positions_of(data: bytes, counts: Ids) -> Ids:
    return positions_all([data], counts)


def positions_all(each: Sequence[bytes], counts: Ids) -> Ids:
    gaps = numbers(b"".join(map(_unwrapped, each)))
    _sound(holds=len(gaps) == total(counts), problem="a list holds positions other than its counts")
    summed = np.cumsum(gaps)
    starts = np.cumsum(counts) - counts
    found = summed - np.repeat(summed[starts] - gaps[starts], counts)
    within = not len(found) or max(largest(gaps), largest(found)) < MOST_POSITION
    _sound(holds=within, problem="a list holds a position past any passage")
    return found


def _ascending(start: int, gaps: NDArray[np.uint32]) -> Ids:
    _sound(holds=bool((gaps >= 1).all()), problem="ranges hold ids out of order")
    return start + after(0, np.cumsum(gaps, dtype=np.int64))


def ranged(ids: Ids, counts: Ids, best: Bytes) -> bytes:
    head = RANGES_HEAD.pack(first(ids), len(ids))
    steps = np.diff(ids)
    if len(steps) and (steps.min() < 1 or largest(steps) >= MOST_GAP):
        raise ValueError("ranges not ascending, or more than 2**32 apart: more than ranges hold")
    gaps = steps.astype(STORED_U32).tobytes()
    return zlib.compress(head + gaps + (counts - 1).astype(np.uint8).tobytes() + best.astype(np.uint8).tobytes(), LEVEL)


def ranges_of(data: bytes) -> Ranges:
    raw = _inflated(data)
    head, n = cast("tuple[int, int]", RANGES_HEAD.unpack_from(raw)) if len(raw) >= RANGES_HEAD.size else (0, 0)
    size = RANGES_HEAD.size + GAP_BYTES * (n - 1) + (COUNT_BYTES + BEST_BYTES) * n
    _sound(holds=n >= 1 and len(raw) == size, problem="ranges hold other than their head says")
    at = RANGES_HEAD.size
    found = _ascending(head, np.frombuffer(raw, STORED_U32, n - 1, at))
    at += GAP_BYTES * (n - 1)
    ends = np.cumsum(np.frombuffer(raw, np.uint8, n, at).astype(np.int64) + 1)
    best = np.frombuffer(raw, np.uint8, n, at + n).copy()
    return Ranges(found, after(0, ends[:-1]), ends, best)


def with_lengths(ids: Ids, lengths: Ids, shift: int) -> dict[int, bytes]:
    blocks = ids >> shift
    cuts = ints(np.append(run_starts(blocks), len(ids)))
    written: dict[int, bytes] = {}
    for start, end in itertools.pairwise(cuts):
        dense = np.zeros(1 << shift, STORED_U32)
        dense[ids[start:end] & ((1 << shift) - 1)] = lengths[start:end]
        written[first(blocks[start:])] = zlib.compress(dense.tobytes(), LEVEL)
    return written


def lengths_of(data: bytes, shift: int) -> Ids:
    raw = _inflated_to(data, LENGTH_BYTES << shift)
    return np.frombuffer(raw, STORED_U32).astype(np.int64)
