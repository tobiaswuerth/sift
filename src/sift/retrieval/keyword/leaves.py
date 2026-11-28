from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NoReturn

import numpy as np

from sift.api.errors import InvalidRequestError
from sift.domain.arrays import Ascending, Ids, among, ascending, distinct, total
from sift.domain.codec.postings import ranges_in
from sift.domain.query.expressions import Phrase, Prefix
from sift.domain.rank.phrases import merged, phrase_counts
from sift.domain.text.wording import counted
from sift.retrieval.keyword.lists import (
    Opened,
    PostingArrays,
    Ranged,
    StoredList,
    Whole,
    everything,
    joined,
    positions_at,
    postings_in,
    stored_within,
)

type Expired = Callable[[], bool]

PREFIX_REMEDY = "write more of the word before the *"
PHRASE_REMEDY = "add a rarer word to the phrase, or search its words without quotes"
FEWER_REMEDY = "search fewer collections"


def never() -> bool:
    return False


@dataclass(eq=False, frozen=True, slots=True)
class Materialized:
    postings: PostingArrays
    cut: bool = False


def _union(store: Opened, lists: list[StoredList], expired: Expired = never) -> Materialized:
    parts: list[PostingArrays] = []
    cut = False
    for held in lists:
        if expired():
            cut = True
            break
        parts.append(everything(store, held))
    places = np.concatenate([np.zeros(0, np.int64), *(p for p, _ in parts)])
    counts = np.concatenate([np.zeros(0, np.int64), *(c for _, c in parts)])
    return Materialized(merged(places, counts), cut)


def size(lists: list[StoredList]) -> int:
    return sum(len(held.places) if isinstance(held, Whole) else held.chunks for held in lists)


def _within(store: Opened, held: StoredList, candidates: Ids) -> Ids:
    if isinstance(held, Whole):
        return held.places
    ranges = ranges_in(held.ranges)
    chosen = np.flatnonzero(among(ranges.ids, distinct(Ascending(candidates >> store.shape.range_shift))))
    return postings_in(store, held, ranges, chosen)[0]


def _rarest_first(phrase: Phrase, lists: dict[str, list[StoredList]]) -> list[str]:
    return sorted(set(phrase.tokens), key=lambda token: (size(lists[token]), token))


def _out_of_time() -> Materialized:
    nothing = np.zeros(0, np.int64)
    return Materialized((nothing, nothing), cut=True)


def _phrase(store: Opened, phrase: Phrase, lists: dict[str, list[StoredList]], expired: Expired) -> Materialized:
    order = _rarest_first(phrase, lists)
    first = _union(store, lists[order[0]], expired)
    candidates = first.postings[0]
    for token in order[1:]:
        if not len(candidates):
            break
        if expired():
            return _out_of_time()
        held = np.concatenate([np.zeros(0, np.int64), *(_within(store, part, candidates) for part in lists[token])])
        candidates = candidates[among(candidates, ascending(held))]
    if first.cut or (len(candidates) and expired()):
        return _out_of_time()
    if not len(candidates):
        return Materialized((candidates, candidates))
    none = (candidates[:0], candidates[:0], candidates[:0])
    wanted = Ascending(candidates)
    occurrences = {
        token: joined([none, *(positions_at(store, part, wanted) for part in lists[token])]) for token in order
    }
    counts = phrase_counts(wanted, [occurrences[token] for token in phrase.tokens])
    found = counts > 0
    return Materialized((candidates[found], counts[found]))


def _prefix(store: Opened, prefix: Prefix, expired: Expired) -> Materialized:
    if expired():
        return _out_of_time()
    lists = stored_within(store, prefix.text)
    return _union(store, [held for each in lists.values() for held in each], expired)


def materialized(
    store: Opened, leaf: Phrase | Prefix, lists: dict[str, list[StoredList]], expired: Expired = never
) -> Materialized:
    return _phrase(store, leaf, lists, expired) if isinstance(leaf, Phrase) else _prefix(store, leaf, expired)


def _ranges(store: Opened, held: StoredList) -> Ids:
    return ranges_in(held.ranges).ids if isinstance(held, Ranged) else held.places >> store.shape.range_shift


def _inside(store: Opened, held: StoredList, wanted: Ascending) -> int:
    if isinstance(held, Ranged):
        ranges = ranges_in(held.ranges)
        return total((ranges.ends - ranges.starts)[among(ranges.ids, wanted)])
    return int(np.count_nonzero(among(held.places >> store.shape.range_shift, wanted)))


def reads_of(store: Opened, phrase: Phrase, lists: dict[str, list[StoredList]]) -> int:
    rarest, *others = _rarest_first(phrase, lists)
    held = np.concatenate([np.zeros(0, np.int64), *(_ranges(store, part) for part in lists[rarest])])
    ranges = distinct(ascending(held))
    return size(lists[rarest]) + sum(_inside(store, part, ranges) for token in others for part in lists[token])


def _written(leaf: Phrase | Prefix) -> str:
    return f"{leaf.text}*" if isinstance(leaf, Prefix) else f'"{" ".join(leaf.tokens)}"'


@dataclass(frozen=True, slots=True)
class Limit:
    total: int
    most: int
    unit: str
    costs: str
    verb: str
    phrase_remedy: str = PHRASE_REMEDY
    stores: int = 1
    cut: bool = False


def refused(costs: Mapping[Phrase | Prefix, int], limit: Limit) -> NoReturn:
    widest, cost = max(costs.items(), key=lambda each: each[1])
    remedy = PREFIX_REMEDY if isinstance(widest, Prefix) else limit.phrase_remedy
    if limit.stores > 1:
        remedy = f"{remedy}, or {FEWER_REMEDY}"
    least = "at least " if limit.cut and cost > limit.most else ""
    rest = f"; with the rest of the query, {least}{counted(limit.total, limit.unit)}"
    said = f"{_written(widest)} {limit.costs} {least}{counted(cost, limit.unit)}{'' if limit.total == cost else rest}"
    raise InvalidRequestError(f"{said}, more than the {limit.most:,} a search {limit.verb}", remedy)
