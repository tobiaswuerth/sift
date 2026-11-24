import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import Ids, Scores, first, ints, last, maxima_at, run_starts, sums_at, total
from sift.domain.codec.postings import encoded, ranged, with_lengths
from sift.domain.codec.shape import SHAPE, Shape
from sift.domain.rank.bm25 import quantized, term_scores

GAP = ""


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
            held: ShortPostings | LongPostings = ShortPostings(*next(parts))
        else:
            blocks = tuple(ShortPostings(*next(parts)) for _ in range(start, stop, shape.block))
            here = slice(start, stop)
            part = ranges_part(lists.ids[here], lists.counts[here], lists.lengths[here], average, shape)
            held = LongPostings(ranges_joined([part]), blocks)
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
