from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import Ids, Mask, Scores, among, first, last
from sift.domain.query.expressions import And, Expression, Leaf, Or, Phrase, Prefix, Term
from sift.domain.rank.bm25 import TOP, slack, term_scores, unquantized
from sift.retrieval.keyword.lists import (
    Opened,
    PostingArrays,
    Ranged,
    StoredList,
    Whole,
    lengths_at,
    live,
    postings_in,
    slices,
)


@dataclass(eq=False, frozen=True, slots=True)
class Exact:
    ranges: Ids
    places: Ids
    scores: Scores


@dataclass(frozen=True, slots=True)
class Bounded:
    held: Ranged
    slack: float


@dataclass(frozen=True, slots=True)
class Scored:
    exact: tuple[Exact, ...]
    bounded: tuple[Bounded, ...]
    size: int


@dataclass(frozen=True, slots=True)
class Query:
    expression: Expression
    leaves: tuple[Leaf, ...]
    weight: Mapping[Leaf, float]
    bounding: Mapping[Leaf, float]
    average: float

    @property
    def flat(self) -> bool:
        return isinstance(self.expression, Or) and all(
            isinstance(part, Term | Phrase | Prefix) for part in self.expression.parts
        )

    @property
    def conjunction(self) -> bool:
        return isinstance(self.expression, And) and all(
            isinstance(part, Term | Phrase | Prefix) for part in self.expression.parts
        )

    def most(self, leaf: Leaf) -> float:
        return self.bounding[leaf] * TOP


def _sorted(parts: Sequence[PostingArrays]) -> PostingArrays:
    places = np.concatenate([np.zeros(0, np.int64), *(here for here, _ in parts)])
    values = np.concatenate([np.zeros(0, np.int64), *(counts for _, counts in parts)])
    order = np.argsort(places, kind="stable")
    return places[order], values[order]


def _scores(store: Opened, places: Ids, counts: Ids, average: float) -> Scores:
    return term_scores(counts, lengths_at(store, places), average)


def _exact(store: Opened, places: Ids, counts: Ids, average: float) -> Exact:
    return Exact(places >> store.shape.range_shift, places, _scores(store, places, counts, average))


def scored(store: Opened, query: Query, found: list[StoredList] | PostingArrays) -> Scored:
    if isinstance(found, tuple):
        places, counts = found
        return Scored((_exact(store, places, counts, query.average),), (), len(places))
    whole = [live(store, held.segment, held.places, held.counts) for held in found if isinstance(held, Whole)]
    exact = (_exact(store, *_sorted(whole), query.average),) if whole else ()
    averages = {segment.id: segment.average for segment in store.segments}
    bounded = tuple(
        Bounded(held, slack(query.average, averages[held.segment])) for held in found if isinstance(held, Ranged)
    )
    size = sum(len(each.places) for each in exact) + sum(last(each.held.ranges.ends) for each in bounded)
    return Scored(exact, bounded, size)


def bounds(query: Query, lists: Mapping[Leaf, Scored]) -> tuple[int, Scores, dict[Leaf, Mask]]:
    found = [each.ranges for held in lists.values() for each in held.exact if len(each.ranges)]
    found += [each.held.ranges.ids for held in lists.values() for each in held.bounded]
    if not found:
        return 0, np.zeros(0), {leaf: np.zeros(0, np.bool_) for leaf in query.leaves}
    start = min(first(each) for each in found)
    span = max(last(each) for each in found) - start + 1
    bound, present = np.zeros(span), dict[Leaf, Mask]()
    for leaf in query.leaves:
        here = np.zeros(span)
        for each in lists[leaf].exact:
            np.maximum.at(here, each.ranges - start, each.scores)
        for each in lists[leaf].bounded:
            at = each.held.ranges.ids - start
            here[at] = np.maximum(here[at], unquantized(each.held.ranges.best) * each.slack)
        bound += query.bounding[leaf] * here
        present[leaf] = here > 0
    return start, bound, present


def _bounded_within(store: Opened, held: Scored, group: Ids) -> Iterator[PostingArrays]:
    for each in held.bounded:
        ids = each.held.ranges.ids
        at = np.minimum(np.searchsorted(ids, group), len(ids) - 1)
        hit = among(group, each.held.ids)
        if hit.any():
            yield postings_in(store, each.held, at[hit])


def within(store: Opened, query: Query, held: Scored, group: Ids) -> tuple[Ids, Scores]:
    places, scores = [np.zeros(0, np.int64)], [np.zeros(0)]
    for each in held.exact:
        take = slices(np.searchsorted(each.ranges, group), np.searchsorted(each.ranges, group, side="right"))
        places.append(each.places[take])
        scores.append(each.scores[take])
    read, counts = _sorted(list(_bounded_within(store, held, group)))
    if len(read):
        places.append(read)
        scores.append(_scores(store, read, counts, query.average))
    joined = np.concatenate(places)
    order = np.argsort(joined, kind="stable")
    return joined[order], np.concatenate(scores)[order]
