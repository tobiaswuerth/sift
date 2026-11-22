import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import Ids, first, ints, last, maxima_at, run_starts, total
from sift.domain.codec.postings import encoded, ranged, with_lengths
from sift.domain.rank.bm25 import quantized, term_scores

GAP = ""
FORMAT = 1
MOST_RANGE_SHIFT = 8
MOST_BLOCK = 1 << 20
MOST_LENGTHS_SHIFT = 20


@dataclass(frozen=True, slots=True)
class Shape:
    range_shift: int = 7
    block: int = 4096
    long: int = 4096
    lengths_shift: int = 10

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.range_shift, self.block, self.long, self.lengths_shift)):
            raise TypeError("a shape is whole numbers")
        if not 0 <= self.range_shift <= MOST_RANGE_SHIFT:
            raise ValueError(
                f"a range holds 2**range_shift places, at most 2**{MOST_RANGE_SHIFT}: its counts are bytes"
            )
        if min(self.block, self.long, self.lengths_shift) < 1:
            raise ValueError("blocks, long lists and blocks of lengths hold one entry at least")
        if max(self.block, self.long) > MOST_BLOCK or self.lengths_shift > MOST_LENGTHS_SHIFT:
            raise ValueError(
                f"blocks hold at most {MOST_BLOCK:,} entries and blocks of lengths 2**{MOST_LENGTHS_SHIFT}"
            )


SHAPE = Shape()


@dataclass(frozen=True, slots=True)
class Short:
    postings: bytes
    positions: bytes


@dataclass(frozen=True, slots=True)
class Long:
    ranges: bytes
    blocks: tuple[Short, ...]


@dataclass(frozen=True, slots=True)
class Postings:
    term: str
    chunks: int
    held: Short | Long


@dataclass(frozen=True, slots=True)
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


def _ranges(ids: Ids, counts: Ids, lengths: Ids, average: float, shape: Shape) -> bytes:
    ranges = ids >> shape.range_shift
    starts = run_starts(ranges)
    best = maxima_at(term_scores(counts, lengths, average), starts)
    return ranged(ranges[starts], np.diff(np.append(starts, len(ids))), quantized(best))


def _cuts(term_starts: list[int], end: int, shape: Shape) -> list[int]:
    found: list[int] = []
    for start, stop in itertools.pairwise([*term_starts, end]):
        found += range(start, stop, shape.block) if stop - start >= shape.long else [start]
    return [*found, end]


def _coded(streams: Sequence[Sequence[str]]) -> tuple[list[str], Ids, Ids, Ids]:
    flat = list(itertools.chain.from_iterable(streams))
    vocabulary = {GAP: 0}
    for token in sorted(set(flat) - vocabulary.keys()):
        vocabulary[token] = len(vocabulary)
    codes = np.fromiter(map(vocabulary.__getitem__, flat), np.int64, len(flat))
    sizes = np.fromiter(map(len, streams), np.int64, len(streams))
    owners = np.repeat(np.arange(len(streams)), sizes)
    places = np.arange(len(flat)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return list(vocabulary), codes, owners, places


def postings_of(lists: Lists, average: float, shape: Shape = SHAPE) -> tuple[Postings, ...]:
    end = len(lists.ids)
    coded = encoded(lists.ids, lists.counts, lists.positions, _cuts(list(lists.starts), end, shape))
    parts = iter(zip(coded.postings, coded.positions, strict=True))
    written: list[Postings] = []
    for term, (start, stop) in zip(lists.terms, itertools.pairwise([*lists.starts, end]), strict=True):
        if stop - start < shape.long:
            held: Short | Long = Short(*next(parts))
        else:
            blocks = tuple(Short(*next(parts)) for _ in range(start, stop, shape.block))
            here = slice(start, stop)
            held = Long(_ranges(lists.ids[here], lists.counts[here], lists.lengths[here], average, shape), blocks)
        written.append(Postings(term, stop - start, held))
    return tuple(written)


def segment(ids: Ids, streams: Sequence[Sequence[str]], average: float, shape: Shape = SHAPE) -> Segment:
    names, codes, owners, places = _coded(streams)
    kept = codes.astype(np.bool_)
    lengths = np.bincount(owners[kept], minlength=len(streams)).astype(np.int64)
    order = np.argsort(codes[kept], kind="stable")
    codes, owners, places = codes[kept][order], owners[kept][order], places[kept][order]
    posting_starts = run_starts(codes * len(streams) + owners)
    counts = np.diff(np.append(posting_starts, len(codes)))
    terms, holders = codes[posting_starts], owners[posting_starts]
    term_starts = ints(run_starts(terms))
    named = tuple(names[code] for code in ints(terms[term_starts]))
    lists = Lists(named, tuple(term_starts), ids[holders], counts, places, lengths[holders])
    written = postings_of(lists, average, shape)
    lengths_written = with_lengths(ids, lengths, shape.lengths_shift)
    return Segment(first(ids), last(ids), len(ids), total(lengths), average, written, lengths_written)
