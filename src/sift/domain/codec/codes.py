from typing import cast

import numpy as np
from numpy.typing import NDArray

type Floats = NDArray[np.float32]
type Bits = NDArray[np.uint8]
type Words = NDArray[np.uint64]
type Distances = NDArray[np.uint16]

WORD_BYTES = 8
MOST_BITS = int(np.iinfo(np.uint16).max)
SCANNED = 65_536
BYTE_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32)


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


def distances(held: Words, query: Words) -> Distances:
    found = np.empty(len(held), np.uint16)
    for start in range(0, len(held), SCANNED):
        found[start : start + SCANNED] = np.bitwise_count(held[start : start + SCANNED] ^ query).sum(
            axis=1, dtype=np.uint16
        )
    return found


def bit_scores(bits: Bits, query: Floats) -> Floats:
    width = width_of(bits)
    padded = np.zeros(8 * width, np.float32)
    padded[: len(query)] = query
    table = cast("Floats", padded.reshape(width, 8) @ BYTE_BITS.T)
    ones = table[np.arange(width), bits].sum(axis=1)
    return 2 * ones - float(padded.sum())
