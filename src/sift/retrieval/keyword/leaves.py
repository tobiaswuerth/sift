import numpy as np

from sift.api.errors import InvalidRequestError
from sift.domain.arrays import Ascending, Ids, among, ascending, distinct, last
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
from sift.storage.keyword_lists import expansions, volume


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
    chosen = np.flatnonzero(among(held.ranges.ids, distinct(Ascending(candidates >> store.shape.range_shift))))
    return postings_in(store, held, chosen)[0]


def _phrase(store: Opened, phrase: Phrase, lists: dict[str, list[Stored]]) -> Postings:
    order = sorted(set(phrase.tokens), key=lambda token: _size(lists[token]))
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


def _counted(n: int, what: str) -> str:
    return f"{n:,} {what}{'s' * (n != 1)}"


def _prefix(store: Opened, prefix: Prefix, most: int) -> Postings:
    held = volume(store.transaction, prefix.text)
    if held.passages > most:
        words, passages = _counted(held.words, "word"), _counted(held.passages, "passage")
        raise InvalidRequestError(
            f"{prefix.text}* stands for {words} in {passages}, more than a search reads",
            "write more of the word before the *",
        )
    terms = expansions(store.transaction, prefix.text)
    lists = stored(store, terms)
    return _all_of(store, [held for term in terms for held in lists[term]])


def materialized(store: Opened, leaf: Phrase | Prefix, lists: dict[str, list[Stored]], most_prefix: int) -> Postings:
    return _phrase(store, leaf, lists) if isinstance(leaf, Phrase) else _prefix(store, leaf, most_prefix)
