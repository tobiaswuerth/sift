import functools
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sift import _native
from sift.domain.arrays import STORED_U32, Ids, Scores, maxima_at, run_starts, sums_at, whole
from sift.domain.codec.postings import encoded, ranged
from sift.domain.codec.shape import SHAPE, Shape
from sift.domain.errors import DamagedError
from sift.domain.rank.bm25 import quantized, term_scores
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS, TRANSLATION

GAP = ""
POINTS = 0x110000
DROPPED = 0xFFFFFFFF

type Building = _native.Building
type Merging = _native.Merging
type _Row = tuple[str, int, bytes | None, bytes | None, bytes | None, list[tuple[bytes, bytes]]]


@dataclass(frozen=True, slots=True)
class ShortPostings:
    postings: bytes
    positions: bytes


@dataclass(frozen=True, slots=True)
class LongPostings:
    ranges: bytes
    blocks: tuple[ShortPostings, ...]


@dataclass(frozen=True, slots=True)
class Postings:
    term: str
    chunks: int
    held: ShortPostings | LongPostings


@dataclass(eq=False, frozen=True, slots=True)
class Lists:
    terms: tuple[str, ...]
    starts: tuple[int, ...]
    ids: Ids
    counts: Ids
    positions: Ids
    lengths: Ids


@dataclass(frozen=True, slots=True)
class Segment:
    first: int
    last: int
    chunks: int
    tokens: int
    average: float
    postings: tuple[Postings, ...]
    lengths: Mapping[int, bytes]


def stream(title: Sequence[str], text: Sequence[str], meta: Sequence[str]) -> list[str]:
    return [*title, GAP, *text, GAP, *meta]


@dataclass(eq=False, frozen=True, slots=True)
class RangesPart:
    ids: Ids
    sizes: Ids
    best: Scores


def ranges_part(ids: Ids, counts: Ids, lengths: Ids, average: float, shape: Shape) -> RangesPart:
    ranges = ids >> shape.range_shift
    starts = run_starts(ranges)
    best = maxima_at(term_scores(counts, lengths, average), starts)
    return RangesPart(ranges[starts], np.diff(np.append(starts, len(ids))), best)


def ranges_joined(parts: Sequence[RangesPart]) -> bytes:
    ids = np.concatenate([each.ids for each in parts])
    starts = run_starts(ids)
    sizes = sums_at(np.concatenate([each.sizes for each in parts]), starts)
    return ranged(ids[starts], sizes, quantized(maxima_at(np.concatenate([each.best for each in parts]), starts)))


def blocks_of(ids: Ids, counts: Ids, positions: Ids, shape: Shape) -> tuple[ShortPostings, ...]:
    if not len(ids):
        return ()
    made = encoded(ids, counts, positions, [*range(0, len(ids), shape.block), len(ids)])
    return tuple(ShortPostings(*each) for each in zip(made.postings, made.positions, strict=True))


def _cuts(term_starts: list[int], end: int, shape: Shape) -> list[int]:
    found: list[int] = []
    for start, stop in itertools.pairwise([*term_starts, end]):
        found += range(start, stop, shape.block) if stop - start >= shape.long else [start]
    return [*found, end]


def postings_of(lists: Lists, average: float, shape: Shape = SHAPE) -> tuple[Postings, ...]:
    end = len(lists.ids)
    coded = encoded(lists.ids, lists.counts, lists.positions, _cuts(list(lists.starts), end, shape))
    parts = iter(zip(coded.postings, coded.positions, strict=True))
    written: list[Postings] = []
    for term, (start, stop) in zip(lists.terms, itertools.pairwise([*lists.starts, end]), strict=True):
        if stop - start < shape.long:
            held: ShortPostings | LongPostings = ShortPostings(*next(parts))
        else:
            blocks = tuple(ShortPostings(*next(parts)) for _ in range(start, stop, shape.block))
            here = slice(start, stop)
            part = ranges_part(lists.ids[here], lists.counts[here], lists.lengths[here], average, shape)
            held = LongPostings(ranges_joined([part]), blocks)
        written.append(Postings(term, stop - start, held))
    return tuple(written)


@functools.cache
def _words() -> _native.Words:
    held = np.arange(POINTS, dtype=STORED_U32)
    for point, kept in TRANSLATION.items():
        held[point] = ord(kept) if kept else DROPPED
    return _native.Words(held.tobytes())


def building(most_tokens: int) -> Building:
    return _words().building(most_tokens, MOST_FIELD_CHARACTERS)


def _kept(row: _Row) -> ShortPostings | LongPostings:
    _, _, postings, positions, ranges, blocks = row
    if ranges is not None:
        return LongPostings(ranges, tuple(ShortPostings(*each) for each in blocks))
    if postings is None or positions is None:
        raise ValueError("a short list holds its postings and positions")
    return ShortPostings(postings, positions)


def merger(live: Ids, lengths: Ids, dead: Ids, average: float, shape: Shape) -> Merging:
    shaped = (shape.range_shift, shape.block, shape.long, shape.lengths_shift)
    return _native.Merging(whole(live), whole(lengths), whole(dead), average, shaped)


@dataclass(frozen=True, slots=True)
class MergedRows:
    terms: Sequence[str]
    postings: Sequence[bytes]
    positions: Sequence[bytes]


def lists_merged(
    held: Merging, rows: MergedRows, most_tokens: int
) -> tuple[tuple[Postings, ...], list[tuple[int, int]]]:
    try:
        made, large = held.lists(rows.terms, rows.postings, rows.positions, most_tokens)
    except ValueError as error:
        raise DamagedError(str(error)) from None
    return tuple(Postings(row[0], row[1], _kept(row)) for row in made), large


def written(built: Building, average: float, shape: Shape = SHAPE) -> Segment:
    chunks, tokens = built.chunks, built.tokens
    first, last, rows, lengths = built.written(average, shape.range_shift, shape.block, shape.long, shape.lengths_shift)
    postings = tuple(Postings(row[0], row[1], _kept(row)) for row in rows)
    return Segment(first, last, chunks, tokens, average, postings, dict(lengths))
