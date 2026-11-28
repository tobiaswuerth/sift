from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from sift import _native
from sift.domain.arrays import Ids, read_only

type Floats = NDArray[np.float32]
type Halves = NDArray[np.float16]
type Bits = NDArray[np.uint8]
type Words = NDArray[np.uint64]
type Distances = NDArray[np.uint16]

WORD_BYTES = 8
MOST_BITS = int(np.iinfo(np.uint16).max)
BYTE_BITS = read_only(np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32))


def width_of(values: NDArray[np.generic]) -> int:
    return cast("int", values.shape[1])


def code_bytes(width: int) -> int:
    return -(-width // (8 * WORD_BYTES)) * WORD_BYTES


def codes(matrix: Floats) -> Bits:
    packed = np.packbits(matrix > 0, axis=1)
    pad = code_bytes(width_of(matrix)) - width_of(packed)
    return np.pad(packed, ((0, 0), (0, pad))) if pad else packed


def words(bits: Bits) -> Words:
    return np.ascontiguousarray(bits).view(np.uint64)


def code_of(query: Floats) -> Words:
    return words(codes(query.reshape(1, -1))).reshape(-1)


@dataclass(frozen=True, slots=True, eq=False)
class CodesFilling:
    native: _native.Filling
    rows: int
    size: int


@dataclass(frozen=True, slots=True, eq=False)
class HeldCodes:
    native: _native.Codes
    rows: int
    size: int


def filling(rows: int, size: int) -> CodesFilling:
    return CodesFilling(_native.Filling(rows, size), rows, size)


def filled(made: CodesFilling, more: bytes) -> None:
    made.native.extend(more)


def finished(made: CodesFilling) -> HeldCodes:
    return HeldCodes(_native.Codes(made.native), made.rows, made.size)


def packed(bits: Bits) -> HeldCodes:
    made = filling(len(bits), width_of(bits))
    filled(made, np.ascontiguousarray(bits).tobytes())
    return finished(made)


def measured(held: Sequence[HeldCodes], query: Words) -> tuple[Distances, Ids]:
    found, counted = _native.scanned_codes([each.native for each in held], np.ascontiguousarray(query).tobytes())
    return np.frombuffer(found, np.uint16), np.asarray(counted, np.int64)


def rows_of(held: HeldCodes, at: Ids) -> Bits:
    found = held.native.picked(np.ascontiguousarray(at, np.int64))
    return np.frombuffer(found, np.uint8).reshape(len(at), held.size)


def bit_scores(bits: Bits, query: Floats) -> Floats:
    width = width_of(bits)
    padded = np.zeros(8 * width, np.float32)
    padded[: len(query)] = query
    table = cast("Floats", padded.reshape(width, 8) @ BYTE_BITS.T)
    ones = table[np.arange(width), bits].sum(axis=1)
    return 2 * ones - float(padded.sum())
