from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sift import _native
from sift.domain.arrays import STORED_U32, Ids, integers, whole
from sift.domain.codec.shape import Shape, shaped
from sift.domain.errors import DamagedError

RAW, PACKED = b"\x00", b"\x01"

type HeldRanges = _native.Ranges
type HeldBlock = _native.Block
type HeldTable = _native.LengthsTable
type Kept = HeldBlock | HeldTable


@dataclass(eq=False, frozen=True, slots=True)
class Ranges:
    ids: Ids
    starts: Ids
    ends: Ids
    best: NDArray[np.uint8]


def blocked_all(each: Sequence[bytes]) -> tuple[Ids, Ids, list[int]]:
    try:
        ids, counts, bounds = _native.blocked_all(each)
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return integers(ids), integers(counts), bounds


def blocked_within(each: Sequence[bytes], numbers: Sequence[int], ranges: HeldRanges, shape: Shape) -> list[HeldBlock]:
    try:
        return _native.blocked_within(each, numbers, ranges, shaped(shape))
    except ValueError as error:
        raise DamagedError(str(error)) from None


def postings_of(blocks: Sequence[HeldBlock]) -> tuple[Ids, Ids, list[int]]:
    ids, counts, bounds = _native.postings_of(blocks)
    return integers(ids), integers(counts), bounds


def ranges_kept(each: Sequence[bytes], bounds: Sequence[tuple[int, int, int]]) -> list[HeldRanges]:
    try:
        return _native.ranges_kept(each, bounds)
    except ValueError as error:
        raise DamagedError(str(error)) from None


def ranges_in(held: HeldRanges) -> Ranges:
    ids, starts, ends, best = _native.ranges_arrays(held)
    return Ranges(integers(ids), integers(starts), integers(ends), np.frombuffer(best, np.uint8))


def occurrences(
    each: Sequence[bytes], bounds: Sequence[int], held: tuple[Ids, Ids], wanted: Ids
) -> tuple[Ids, Ids, Ids]:
    places, counts = held
    try:
        found, times, positions = _native.occurrences(each, bounds, whole(places), whole(counts), whole(wanted))
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return integers(found), integers(times), integers(positions)


def with_lengths(ids: Ids, lengths: Ids, shift: int) -> dict[int, bytes]:
    return dict(_native.with_lengths(whole(ids), whole(lengths), shift))


def lengths_all(each: Sequence[bytes], shift: int) -> NDArray[np.uint32]:
    try:
        found = _native.lengths_all(each, shift)
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return np.frombuffer(found, STORED_U32).astype(np.uint32, copy=False).reshape(len(each), 1 << shift)


def kept_bytes(value: Kept) -> int:
    return _native.kept_bytes(value)
