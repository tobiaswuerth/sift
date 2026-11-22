import numpy as np

from sift.domain.arrays import Ids, last
from sift.domain.query.expressions import Phrase, Prefix
from sift.domain.rank.phrases import merged, phrase_counts
from sift.retrieval.keyword.lists import (
    Opened,
    Postings,
    Stored,
    Whole,
    everything,
    joined,
    positions_at,
    postings_in,
    stored,
)
from sift.storage.keyword_lists import expansions


def _all_of(store: Opened, lists: list[Stored]) -> Postings:
    parts = [everything(store, held) for held in lists]
    places = np.concatenate([np.zeros(0, np.int64), *(p for p, _ in parts)])
    counts = np.concatenate([np.zeros(0, np.int64), *(c for _, c in parts)])
    return merged(places, counts)


def _size(lists: list[Stored]) -> int:
    return sum(len(held.places) if isinstance(held, Whole) else last(held.ranges.ends) for held in lists)


def _within(store: Opened, held: Stored, candidates: Ids) -> Ids:
    if isinstance(held, Whole):
        return held.places
    chosen = np.flatnonzero(np.isin(held.ranges.ids, np.unique(candidates >> store.shape.range_shift)))
    return postings_in(store, held, chosen)[0]


def _phrase(store: Opened, phrase: Phrase, lists: dict[str, list[Stored]]) -> Postings:
    order = sorted(set(phrase.tokens), key=lambda token: _size(lists[token]))
    candidates = _all_of(store, lists[order[0]])[0]
    for token in order[1:]:
        if not len(candidates):
            break
        held = np.concatenate([np.zeros(0, np.int64), *(_within(store, part, candidates) for part in lists[token])])
        candidates = candidates[np.isin(candidates, held)]
    if not len(candidates):
        return candidates, candidates
    none = (candidates[:0], candidates[:0], candidates[:0])
    occurrences = {
        token: joined([none, *(positions_at(store, part, candidates) for part in lists[token])]) for token in order
    }
    counts = phrase_counts(candidates, [occurrences[token] for token in phrase.tokens])
    found = counts > 0
    return candidates[found], counts[found]


def _prefix(store: Opened, prefix: Prefix) -> Postings:
    terms = expansions(store.transaction, prefix.text)
    lists = stored(store, terms)
    return _all_of(store, [held for term in terms for held in lists[term]])


def materialized(store: Opened, leaf: Phrase | Prefix, lists: dict[str, list[Stored]]) -> Postings:
    return _phrase(store, leaf, lists) if isinstance(leaf, Phrase) else _prefix(store, leaf)
