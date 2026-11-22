import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import Ids, first, ints, last, maxima_at, run_starts, total
from sift.domain.codec.postings import encoded, ranged, with_lengths
from sift.domain.rank.bm25 import quantized, term_scores

GAP = ""


@dataclass(frozen=True, slots=True)
class Shape:
    range_shift: int = 7
    block: int = 4096
    long: int = 4096
    lengths_shift: int = 10


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
    for token in set(flat) - vocabulary.keys():
        vocabulary[token] = len(vocabulary)
    codes = np.fromiter(map(vocabulary.__getitem__, flat), np.int64, len(flat))
    sizes = np.fromiter(map(len, streams), np.int64, len(streams))
    owners = np.repeat(np.arange(len(streams)), sizes)
    places = np.arange(len(flat)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return list(vocabulary), codes, owners, places


def segment(ids: Ids, streams: Sequence[Sequence[str]], average: float, shape: Shape = SHAPE) -> Segment:
    names, codes, owners, places = _coded(streams)
    kept = codes.astype(np.bool_)
    lengths = np.bincount(owners[kept], minlength=len(streams)).astype(np.int64)
    order = np.argsort(codes[kept], kind="stable")
    codes, owners, places = codes[kept][order], owners[kept][order], places[kept][order]
    posting_starts = run_starts(codes * len(streams) + owners)
    counts = np.diff(np.append(posting_starts, len(codes)))
    terms, holders = codes[posting_starts], owners[posting_starts]
    chunk_ids, term_names, term_starts = ids[holders], ints(terms), ints(run_starts(terms))
    cuts = _cuts(term_starts, len(term_names), shape)
    lists = encoded(chunk_ids, counts, places, cuts)
    parts = iter(zip(lists.postings, lists.positions, strict=True))
    written: list[Postings] = []
    for start, end in itertools.pairwise([*term_starts, len(term_names)]):
        if end - start < shape.long:
            held: Short | Long = Short(*next(parts))
        else:
            blocks = tuple(Short(*next(parts)) for _ in range(start, end, shape.block))
            here = slice(start, end)
            held = Long(_ranges(chunk_ids[here], counts[here], lengths[holders[here]], average, shape), blocks)
        written.append(Postings(names[term_names[start]], end - start, held))
    lengths_written = with_lengths(ids, lengths, shape.lengths_shift)
    return Segment(first(ids), last(ids), len(ids), total(lengths), average, tuple(written), lengths_written)
