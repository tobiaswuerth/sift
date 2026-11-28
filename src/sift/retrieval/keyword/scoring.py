import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import integers, reals
from sift.domain.codec.postings import HeldBlock
from sift.domain.codec.shape import shaped
from sift.domain.query.expressions import Expression, Leaf
from sift.domain.rank.best import Best, offered
from sift.domain.rank.bm25 import slack
from sift.domain.rank.pruned import ExactPlan, LeafPlan, PartPlan, ScoringPlan, SegmentPlan, nodes_of
from sift.retrieval.allowed import AllowedDocuments
from sift.retrieval.keyword.leaves import Expired
from sift.retrieval.keyword.lists import (
    Opened,
    PostingArrays,
    Ranged,
    StoredList,
    Whole,
    dead,
    decoded_blocks,
    lengths_read,
    owners,
)


@dataclass(frozen=True, slots=True)
class Query:
    expression: Expression
    leaves: tuple[Leaf, ...]
    weight: Mapping[Leaf, float]
    bounding: Mapping[Leaf, float]
    average: float


def _owner(wanted: tuple[int, int]) -> int:
    return wanted[0]


@dataclass(eq=False, slots=True)
class ListReads:
    store: Opened
    parts: Sequence[Ranged]
    best: Best
    allowed: AllowedDocuments | None
    expired: Expired

    def read_blocks(self, wanted: list[tuple[int, int]]) -> list[HeldBlock]:
        found: list[HeldBlock] = []
        for part, each in itertools.groupby(wanted, key=_owner):
            numbers = [number for _, number in each]
            held = decoded_blocks(self.store, self.parts[part], numbers)
            found += [held[number] for number in numbers]
        return found

    def read_lengths(self, wanted: list[tuple[int, int]]) -> list[bytes]:
        found: list[bytes] = []
        for segment, each in itertools.groupby(wanted, key=_owner):
            found += lengths_read(self.store, self.store.segments[segment].id, [number for _, number in each])
        return found

    def offered(self, places: bytes, scores: bytes) -> tuple[float, int]:
        chunks, held = owners(self.store, integers(places))
        values = reals(scores)
        if self.allowed is not None:
            keep = self.allowed.documents(held)
            chunks, held, values = chunks[keep], held[keep], values[keep]
        self.best = offered(self.best, chunks, held, values)
        return self.best.threshold, len(self.best.chunks)

    def out_of_time(self) -> bool:
        return self.expired()


def _ranged(found: list[StoredList] | PostingArrays) -> list[Ranged]:
    return [] if isinstance(found, tuple) else [held for held in found if isinstance(held, Ranged)]


def _exact(found: list[StoredList] | PostingArrays, at: Mapping[int, int]) -> list[ExactPlan]:
    if isinstance(found, tuple):
        return [ExactPlan(*found, None)]
    return [ExactPlan(held.places, held.counts, at[held.segment]) for held in found if isinstance(held, Whole)]


def _stored(found: list[StoredList] | PostingArrays) -> list[StoredList]:
    return [] if isinstance(found, tuple) else found


def plan_of(
    store: Opened,
    query: Query,
    lists: Mapping[Leaf, list[StoredList] | PostingArrays],
    best: Best,
    batches: tuple[int, int],
) -> tuple[ScoringPlan, list[Ranged]]:
    at = {segment.id: number for number, segment in enumerate(store.segments)}
    owned = [_ranged(lists[leaf]) for leaf in query.leaves]
    starts = list(itertools.accumulate(map(len, owned), initial=0))
    leaves = [
        LeafPlan(_exact(lists[leaf], at), list(range(start, end)), query.bounding[leaf], query.weight[leaf])
        for leaf, start, end in zip(query.leaves, starts[:-1], starts[1:], strict=True)
    ]
    parts = [held for own in owned for held in own]
    held = {each.segment for leaf in query.leaves for each in _stored(lists[leaf])}
    none = np.zeros(0, np.int64)
    segments = [
        SegmentPlan(
            segment.first,
            segment.last,
            dead(store, segment.id).values if segment.id in held else none,
            store.decoded.lengths.get(segment.id),
        )
        for segment in store.segments
    ]
    averages = {segment.id: segment.average for segment in store.segments}
    told = [PartPlan(at[part.segment], part.ranges, slack(query.average, averages[part.segment])) for part in parts]
    numbers = (query.average, shaped(store.shape), *batches)
    nodes = nodes_of(query.expression, query.leaves)
    return ScoringPlan(segments, told, leaves, nodes, numbers, (best.threshold, len(best.chunks))), parts
