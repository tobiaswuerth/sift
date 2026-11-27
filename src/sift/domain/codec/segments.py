import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sift import _native
from sift.domain.arrays import STORED_U32, Ids, whole
from sift.domain.codec.shape import SHAPE, Shape
from sift.domain.errors import DamagedError
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS, TRANSLATION

GAP = ""
POINTS = 0x110000
DROPPED = 0xFFFFFFFF

type Building = _native.Building
type Merger = _native.Merger
type ListsRow = tuple[int, str, int, bytes | None, bytes | None, bytes | None]
type ListBlockRow = tuple[int, str, int, bytes, bytes]


@dataclass(frozen=True, slots=True)
class WrittenSegment:
    first: int
    last: int
    chunks: int
    tokens: int
    average: float
    lists: Sequence[ListsRow]
    blocks: Sequence[ListBlockRow]
    lengths: Sequence[tuple[int, int, bytes]]
    terms: Sequence[tuple[str, int]]


def stream(title: Sequence[str], text: Sequence[str], meta: Sequence[str]) -> list[str]:
    return [*title, GAP, *text, GAP, *meta]


@functools.cache
def _words() -> _native.Words:
    held = np.arange(POINTS, dtype=STORED_U32)
    for point, kept in TRANSLATION.items():
        held[point] = ord(kept) if kept else DROPPED
    return _native.Words(held.tobytes())


def building(most_tokens: int) -> Building:
    return _words().building(most_tokens, MOST_FIELD_CHARACTERS)


def _shaped(shape: Shape) -> tuple[int, int, int, int]:
    return shape.range_shift, shape.block, shape.long, shape.lengths_shift


@dataclass(eq=False, frozen=True, slots=True)
class MergedChunks:
    live: Ids
    lengths: Ids
    dead: Ids
    average: float


@dataclass(frozen=True, slots=True)
class MergeLimits:
    postings: int
    stored: int
    tokens: int


@dataclass(frozen=True, slots=True)
class MergeStep:
    lists: Sequence[ListsRow]
    blocks: Sequence[ListBlockRow]
    wanted: Sequence[int]
    done: bool


def merger(chunks: MergedChunks, shape: Shape, limits: MergeLimits, sources: int) -> Merger:
    live, lengths, dead = whole(chunks.live), whole(chunks.lengths), whole(chunks.dead)
    most = (limits.postings, limits.stored, limits.tokens)
    return _native.Merger(live, lengths, dead, chunks.average, _shaped(shape), most, sources)


def fed(held: Merger, source: int, rows: Sequence[tuple[object, ...]]) -> bool:
    try:
        return held.feed(source, rows)
    except (TypeError, OverflowError):
        raise DamagedError("the keyword index holds a row that is no list of a term") from None


def stepped(held: Merger, segment: int) -> MergeStep:
    try:
        (lists, blocks), wanted, done = held.step(segment)
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return MergeStep(lists, blocks, wanted, done)


class Collected(NamedTuple):
    first: int
    last: int
    chunks: int
    tokens: int


def gathered(built: Building) -> Collected:
    first, last, chunks, tokens = _native.gathered(built)
    if first is None or last is None:
        raise ValueError("a segment holds a chunk at least")
    return Collected(first, last, chunks, tokens)


def written(built: Building, segment: int, average: float, shape: Shape = SHAPE) -> WrittenSegment:
    held = gathered(built)
    (lists, blocks), lengths, terms = built.written(segment, average, _shaped(shape))
    return WrittenSegment(held.first, held.last, held.chunks, held.tokens, average, lists, blocks, lengths, terms)
