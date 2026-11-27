import itertools
import struct
import zlib
from collections.abc import Sequence

import numpy as np

from sift.domain.arrays import Ids
from sift.domain.codec.postings import PACKED, RANGES_HEAD, RAW, occurrences

BLOCK_HEAD = struct.Struct("<qI")
PACKED_FROM = 64


def varint(number: int) -> bytes:
    low = [number >> shift & 0x7F for shift in range(0, max(number.bit_length(), 1), 7)]
    return bytes([*(each | 0x80 for each in low[:-1]), low[-1]])


def varints(numbers: Sequence[int]) -> bytes:
    return b"".join(map(varint, numbers))


def _gaps(ids: Sequence[int]) -> bytes:
    return b"".join((after - before).to_bytes(4, "little") for before, after in itertools.pairwise(ids))


def _wrapped(body: bytes) -> bytes:
    return RAW + body if len(body) < PACKED_FROM else PACKED + zlib.compress(body)


def block(ids: Sequence[int], counts: Sequence[int]) -> bytes:
    return _wrapped(BLOCK_HEAD.pack(ids[0], len(ids)) + _gaps(ids) + varints(counts))


def placed(positions: Sequence[Sequence[int]]) -> bytes:
    return _wrapped(
        b"".join(varints([after - before for before, after in itertools.pairwise([0, *each])]) for each in positions)
    )


def positions_of(data: bytes, counts: Ids) -> Ids:
    places = np.arange(len(counts), dtype=np.int64)
    return occurrences([data], [0, len(counts)], (places, counts), places)[2]


def ranges(ids: Sequence[int], sizes: Sequence[int], best: Sequence[int]) -> bytes:
    fewer = bytes(size - 1 for size in sizes)
    return zlib.compress(RANGES_HEAD.pack(ids[0], len(ids)) + _gaps(ids) + fewer + bytes(best))


type PackedEntry = tuple[str, int, bytes, bytes]


def _number(data: bytes, at: int) -> tuple[int, int]:
    found = shift = 0
    while True:
        byte = data[at]
        found |= (byte & 0x7F) << shift
        at, shift = at + 1, shift + 7
        if byte < 0x80:
            return found, at


def packed(lists: bytes, places: bytes) -> list[PackedEntry]:
    (count,) = struct.unpack_from("<I", lists)
    heads: list[tuple[str, int, int, int]] = []
    at = 4
    for _ in range(count):
        named, at = _number(lists, at)
        term, at = lists[at : at + named].decode(), at + named
        chunks, at = _number(lists, at)
        listed, at = _number(lists, at)
        held, at = _number(lists, at)
        heads.append((term, chunks, listed, held))
    found: list[PackedEntry] = []
    placed = 0
    for term, chunks, listed, held in heads:
        found.append((term, chunks, lists[at : at + listed], places[placed : placed + held]))
        at, placed = at + listed, placed + held
    assert (at, placed) == (len(lists), len(places))
    return found


def pack(entries: Sequence[PackedEntry]) -> tuple[bytes, bytes]:
    heads = b"".join(
        varint(len(term.encode())) + term.encode() + varints([chunks, len(postings), len(positions)])
        for term, chunks, postings, positions in entries
    )
    lists = struct.pack("<I", len(entries)) + heads + b"".join(each[2] for each in entries)
    return lists, b"".join(each[3] for each in entries)
