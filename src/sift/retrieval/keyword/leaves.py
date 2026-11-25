from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

import numpy as np

from sift.api.errors import InvalidRequestError
from sift.domain.arrays import Ascending, Ids, among, ascending, distinct, last
from sift.domain.query.expressions import Phrase, Prefix
from sift.domain.rank.phrases import merged, phrase_counts
from sift.domain.text.wording import counted
from sift.retrieval.keyword.lists import (
    Opened,
    PostingArrays,
    StoredList,
    Whole,
    everything,
    joined,
    positions_at,
    postings_in,
    stored,
)
from sift.storage.keyword_lists import expansions, volume

PREFIX_REMEDY = "write more of the word before the *"
PHRASE_REMEDY = "add a rarer word to the phrase, or search its words without quotes"


def _all_of(store: Opened, lists: list[StoredList]) -> PostingArrays:
    parts = [everything(store, held) for held in lists]
    places = np.concatenate([np.zeros(0, np.int64), *(p for p, _ in parts)])
    counts = np.concatenate([np.zeros(0, np.int64), *(c for _, c in parts)])
    return merged(places, counts)


def _size(lists: list[StoredList]) -> int:
    return sum(len(held.places) if isinstance(held, Whole) else last(held.ranges.ends) for held in lists)


def _within(store: Opened, held: StoredList, candidates: Ids) -> Ids:
    if isinstance(held, Whole):
        return held.places
    chosen = np.flatnonzero(among(held.ranges.ids, distinct(Ascending(candidates >> store.shape.range_shift))))
    return postings_in(store, held, chosen)[0]


def _rarest_first(phrase: Phrase, lists: dict[str, list[StoredList]]) -> list[str]:
    return sorted(set(phrase.tokens), key=lambda token: (_size(lists[token]), token))


def _phrase(store: Opened, phrase: Phrase, lists: dict[str, list[StoredList]]) -> PostingArrays:
    order = _rarest_first(phrase, lists)
    candidates = _all_of(store, lists[order[0]])[0]
    for token in order[1:]:
        if not len(candidates):
            break
        held = np.concatenate([np.zeros(0, np.int64), *(_within(store, part, candidates) for part in lists[token])])
        candidates = candidates[among(candidates, ascending(held))]
    if not len(candidates):
        return candidates, candidates
    none = (candidates[:0], candidates[:0], candidates[:0])
    wanted = Ascending(candidates)
    occurrences = {
        token: joined([none, *(positions_at(store, part, wanted) for part in lists[token])]) for token in order
    }
    counts = phrase_counts(wanted, [occurrences[token] for token in phrase.tokens])
    found = counts > 0
    return candidates[found], counts[found]


def _prefix(store: Opened, prefix: Prefix) -> PostingArrays:
    terms = expansions(store.transaction, prefix.text)
    lists = stored(store, terms)
    return _all_of(store, [held for term in terms for held in lists[term]])


def materialized(store: Opened, leaf: Phrase | Prefix, lists: dict[str, list[StoredList]]) -> PostingArrays:
    return _phrase(store, leaf, lists) if isinstance(leaf, Phrase) else _prefix(store, leaf)


def reads_of(store: Opened, leaf: Phrase | Prefix, lists: dict[str, list[StoredList]]) -> int:
    if isinstance(leaf, Prefix):
        return volume(store.transaction, leaf.text).passages
    return sum(_size(lists[token]) for token in set(leaf.tokens))


def expansions_of(store: Opened, prefix: Prefix) -> int:
    return volume(store.transaction, prefix.text).words


def _written(leaf: Phrase | Prefix) -> str:
    return f"{leaf.text}*" if isinstance(leaf, Prefix) else f'"{" ".join(leaf.tokens)}"'


@dataclass(frozen=True, slots=True)
class Limit:
    total: int
    most: int
    unit: str
    costs: str
    verb: str


def refused(costs: Mapping[Phrase | Prefix, int], limit: Limit) -> NoReturn:
    widest, cost = max(costs.items(), key=lambda each: each[1])
    remedy = PREFIX_REMEDY if isinstance(widest, Prefix) else PHRASE_REMEDY
    whole = "" if limit.total == cost else f"; with the rest of the query, {counted(limit.total, limit.unit)}"
    said = f"{_written(widest)} {limit.costs} {counted(cost, limit.unit)}{whole}"
    raise InvalidRequestError(f"{said}, more than the {limit.most:,} a search {limit.verb}", remedy)
