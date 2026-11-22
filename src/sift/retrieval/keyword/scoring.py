from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import Ids, Mask, among, first, last
from sift.domain.query.expressions import And, Expression, Leaf, Or, Phrase, Prefix, Term
from sift.domain.rank.bm25 import TOP, Scores, slack, term_scores, unquantized
from sift.retrieval.keyword.lists import Opened, Postings, Ranged, Stored, Whole, lengths_at, live, postings_in, slices


@dataclass(frozen=True, slots=True)
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


def _exact(store: Opened, places: Ids, counts: Ids, average: float) -> Exact:
    scores = term_scores(counts, lengths_at(store, places), average)
    return Exact(places >> store.shape.range_shift, places, scores)


def scored(store: Opened, query: Query, found: list[Stored] | Postings) -> Scored:
    if isinstance(found, tuple):
        places, counts = found
        return Scored((_exact(store, places, counts, query.average),), (), len(places))
    exact = tuple(
        _exact(store, *live(store, held.places, held.counts), query.average)
        for held in found
        if isinstance(held, Whole)
    )
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


def within(store: Opened, query: Query, held: Scored, group: Ids) -> Iterator[tuple[Ids, Scores]]:
    for each in held.exact:
        take = slices(np.searchsorted(each.ranges, group), np.searchsorted(each.ranges, group, side="right"))
        yield each.places[take], each.scores[take]
    for each in held.bounded:
        ids = each.held.ranges.ids
        at = np.minimum(np.searchsorted(ids, group), len(ids) - 1)
        hit = among(group, each.held.ids)
        if hit.any():
            places, counts = postings_in(store, each.held, at[hit])
            yield places, term_scores(counts, lengths_at(store, places), query.average)


def merged_within(store: Opened, query: Query, held: Scored, group: Ids) -> tuple[Ids, Scores]:
    parts = list(within(store, query, held, group)) if len(group) else []
    places = np.concatenate([np.zeros(0, np.int64), *(p for p, _ in parts)])
    scores = np.concatenate([np.zeros(0), *(s for _, s in parts)])
    order = np.argsort(places, kind="stable")
    return places[order], scores[order]
