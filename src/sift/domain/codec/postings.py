import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from sift import _native
from sift.domain.arrays import STORED_I64, STORED_U32, Ids, after, whole
from sift.domain.codec.shape import Shape, shaped
from sift.domain.errors import DamagedError

type Bytes = NDArray[np.uint8]

RAW, PACKED = b"\x00", b"\x01"
RANGES_HEAD = struct.Struct("<qI")
GAP_BYTES = 4
COUNT_BYTES = 1
BEST_BYTES = 1
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


def _numbers(stored: bytes) -> Ids:
    return np.frombuffer(stored, STORED_I64).astype(np.int64, copy=False)


def blocked_all(each: Sequence[bytes]) -> tuple[Ids, Ids, list[int]]:
    try:
        ids, counts, bounds = _native.blocked_all(each)
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return _numbers(ids), _numbers(counts), bounds


def blocked_within(
    each: Sequence[bytes], numbers: Sequence[int], ranges: Ranges, shape: Shape
) -> list[tuple[Ids, Ids]]:
    try:
        found = _native.blocked_within(each, numbers, whole(ranges.ids), whole(ranges.ends), shaped(shape))
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return [(_numbers(places), _numbers(counts)) for places, counts in found]


def occurrences(
    each: Sequence[bytes], bounds: Sequence[int], held: tuple[Ids, Ids], wanted: Ids
) -> tuple[Ids, Ids, Ids]:
    places, counts = held
    try:
        found, times, positions = _native.occurrences(each, bounds, whole(places), whole(counts), whole(wanted))
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return _numbers(found), _numbers(times), _numbers(positions)


def _ascending(start: int, gaps: NDArray[np.uint32]) -> Ids:
    _sound(holds=bool((gaps >= 1).all()), problem="ranges hold ids out of order")
    return start + after(0, np.cumsum(gaps, dtype=np.int64))


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
    return dict(_native.with_lengths(whole(ids), whole(lengths), shift))


def lengths_all(each: Sequence[bytes], shift: int) -> NDArray[np.uint32]:
    try:
        found = _native.lengths_all(each, shift)
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return np.frombuffer(found, STORED_U32).astype(np.uint32, copy=False).reshape(len(each), 1 << shift)
