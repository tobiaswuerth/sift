from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sift.api.ids import ChunkId, DocumentId
from sift.domain.arrays import Ascending, Ids, Scores, among, distinct, ints
from sift.domain.query.expressions import Expression, Leaf, Phrase, Prefix, Term, leaves
from sift.domain.rank.best import Best, may_reach, offered, ranked, unreached_alone
from sift.domain.rank.bm25 import idf
from sift.domain.rank.boolean import counted, may_hold
from sift.retrieval.allowed import AllowedDocuments
from sift.retrieval.keyword.leaves import Expired, Limit, expansions_of, materialized, never, reads_of, refused
from sift.retrieval.keyword.lists import Opened, PostingArrays, StoredList, owners, stored
from sift.retrieval.keyword.scoring import Query, Scored, bounds, scored, within
from sift.storage.keyword_lists import frequencies


@dataclass(frozen=True, slots=True)
class Tuning:
    first_batch: int = 16
    most_batch: int = 4096
    most_read: int = 10_000_000
    most_expansions: int = 100_000


TUNING = Tuning()


@dataclass(frozen=True, slots=True)
class Searched:
    best: Best
    cut: bool


@dataclass(frozen=True, slots=True)
class FoundChunk:
    chunk: ChunkId
    document: DocumentId
    score: float


@dataclass(frozen=True, slots=True)
class Pass:
    store: Opened
    query: Query
    lists: Mapping[Leaf, Scored]
    allowed: AllowedDocuments | None


def _offer(work: Pass, best: Best, places: Ids, scores: Scores) -> Best:
    if not len(places):
        return best
    chunks, held = owners(work.store, places)
    if work.allowed is not None:
        keep = work.allowed.documents(held)
        chunks, held, scores = chunks[keep], held[keep], scores[keep]
    return offered(best, chunks, held, scores)


def _lazy(work: Pass, threshold: float) -> list[Leaf]:
    ascending = sorted(work.query.leaves, key=work.query.most) if work.query.flat else []
    return ascending[: unreached_alone(np.array([work.query.most(leaf) for leaf in ascending]), threshold)]


def _summed(query: Query, held: Mapping[Leaf, Scores], size: int) -> Scores:
    total = np.zeros(size)
    for leaf in query.leaves:
        if leaf in held:
            total = total + query.bounding[leaf] * held[leaf]
    return total


def _held_at(found: dict[Leaf, tuple[Ids, Scores]]) -> tuple[Ids, dict[Leaf, Scores]]:
    every = [here for here, _ in found.values()]
    places, where = np.unique(np.concatenate([np.zeros(0, np.int64), *every]), return_inverse=True)
    held: dict[Leaf, Scores] = {}
    at = 0
    for leaf, (here, scores) in found.items():
        held[leaf] = np.zeros(len(places))
        held[leaf][where[at : at + len(here)]] = scores
        at += len(here)
    return places, held


def _with_lazy(
    work: Pass, best: Best, places: Ids, held: dict[Leaf, Scores], lazy: list[Leaf]
) -> tuple[Ids, dict[Leaf, Scores]]:
    query, shift = work.query, work.store.shape.range_shift
    lazy = sorted(lazy, key=query.most, reverse=True)
    acc = _summed(query, held, len(places))
    for n, leaf in enumerate(lazy):
        near = np.flatnonzero(may_reach(acc + sum(query.most(other) for other in lazy[n:]), best.threshold))
        places, acc, held = places[near], acc[near], {key: value[near] for key, value in held.items()}
        found = np.zeros(len(places))
        here, scores = within(work.store, query, work.lists[leaf], distinct(Ascending(places >> shift)).values)
        at = np.minimum(np.searchsorted(here, places), max(len(here) - 1, 0))
        hit = np.flatnonzero(among(places, Ascending(here)))
        found[hit] = scores[at[hit]]
        held[leaf] = found
        acc = acc + query.bounding[leaf] * found
    return places, held


def _all_of(work: Pass, best: Best, group: Ids) -> Best:
    query, shift = work.query, work.store.shape.range_shift
    places, held = np.zeros(0, np.int64), {}
    for n, leaf in enumerate(sorted(query.leaves, key=lambda each: work.lists[each].size)):
        here, scores = within(
            work.store, query, work.lists[leaf], group if n == 0 else distinct(Ascending(places >> shift)).values
        )
        if n == 0:
            places, held = here, {leaf: scores}
            continue
        at = np.minimum(np.searchsorted(here, places), max(len(here) - 1, 0))
        hit = np.flatnonzero(among(places, Ascending(here)))
        places, held = places[hit], {key: value[hit] for key, value in held.items()} | {leaf: scores[at[hit]]}
    acc = _summed(query, held, len(places))
    keep = np.flatnonzero(acc >= best.threshold)
    return _offer(work, best, places[keep], acc[keep])


def _scored_group(work: Pass, best: Best, group: Ids) -> Best:
    query = work.query
    if query.conjunction:
        return _all_of(work, best, group)
    lazy = _lazy(work, best.threshold)
    eager = [leaf for leaf in query.leaves if leaf not in lazy]
    found = {leaf: within(work.store, query, work.lists[leaf], group) for leaf in eager}
    if not any(len(places) for places, _ in found.values()):
        return best
    places, held = _held_at(found)
    if query.flat:
        places, held = _with_lazy(work, best, places, held, lazy)
        acc = _summed(query, held, len(places))
        keep = np.flatnonzero((acc >= best.threshold) & (acc > 0))
        return _offer(work, best, places[keep], acc[keep])
    holds, counting = counted(query.expression, {leaf: scores > 0 for leaf, scores in held.items()})
    acc = np.zeros(len(places))
    for leaf, where in counting:
        acc += np.where(where, query.weight[leaf] * held[leaf], 0.0)
    keep = np.flatnonzero(holds & (acc >= best.threshold))
    return _offer(work, best, places[keep], acc[keep])


def _searched(work: Pass, best: Best, tuning: Tuning, expired: Expired) -> Searched:
    first, bound, present = bounds(work.query, work.lists)
    candidates = np.flatnonzero(may_hold(work.query.expression, present) & (bound > 0))
    remaining, bound = candidates + first, bound[candidates]
    batch, began = tuning.first_batch, False
    while len(remaining):
        beat = may_reach(bound, best.threshold)
        remaining, bound = remaining[beat], bound[beat]
        if began and len(remaining) and len(best.chunks) and expired():
            return Searched(best, cut=True)
        if len(remaining) > batch:
            pick = np.argpartition(-bound, batch - 1)[:batch]
            rest = np.ones(len(remaining), np.bool_)
            rest[pick] = False
            group, remaining, bound = remaining[pick], remaining[rest], bound[rest]
        else:
            group, remaining, bound = remaining, remaining[:0], bound[:0]
        if len(group):
            best, began = _scored_group(work, best, np.sort(group)), True
        batch = min(2 * batch, tuning.most_batch)
    return Searched(best, cut=False)


def _tokens(unique: Sequence[Leaf]) -> list[str]:
    found = [leaf.text if isinstance(leaf, Term) else leaf.tokens for leaf in unique if not isinstance(leaf, Prefix)]
    return sorted({token for each in found for token in ((each,) if isinstance(each, str) else each)})


type ReadLists = dict[Leaf, list[StoredList] | PostingArrays]


@dataclass(eq=False, frozen=True, slots=True)
class Prepared:
    store: Opened
    lists: ReadLists
    df: Mapping[Leaf, int]
    allowed: AllowedDocuments | None = None
    cut: bool = False


@dataclass(frozen=True, slots=True)
class Planned:
    store: Opened
    leaves: tuple[Leaf, ...]
    lists: dict[str, list[StoredList]]
    reads: Mapping[Phrase | Prefix, int]
    expansions: Mapping[Prefix, int]


def planned(store: Opened, expression: Expression) -> Planned:
    unique = tuple(dict.fromkeys(leaves(expression)))
    lists = stored(store, _tokens(unique))
    reads = {leaf: reads_of(store, leaf, lists) for leaf in unique if isinstance(leaf, Phrase | Prefix)}
    expanded = {leaf: expansions_of(store, leaf) for leaf in unique if isinstance(leaf, Prefix)}
    return Planned(store, unique, lists, reads, expanded)


def refuse_reading_too_much(plans: Sequence[Planned], tuning: Tuning = TUNING) -> None:
    reads: Counter[Phrase | Prefix] = Counter()
    expanded: Counter[Phrase | Prefix] = Counter()
    for plan in plans:
        reads.update(plan.reads)
        expanded.update(plan.expansions)
    if (total := expanded.total()) > tuning.most_expansions:
        refused(expanded, Limit(total, tuning.most_expansions, "word", "stands for", "expands"))
    if (total := reads.total()) > tuning.most_read:
        refused(reads, Limit(total, tuning.most_read, "passage", "reads", "reads"))


def prepared(plan: Planned, *, allowed: AllowedDocuments | None = None, expired: Expired = never) -> Prepared:
    store, lists = plan.store, plan.lists
    terms = frequencies(store.transaction, [leaf.text for leaf in plan.leaves if isinstance(leaf, Term)])
    found: ReadLists = {}
    df: dict[Leaf, int] = {}
    cut = False
    for leaf in plan.leaves:
        if isinstance(leaf, Term):
            found[leaf], df[leaf] = lists[leaf.text], terms.get(leaf.text, 0)
        else:
            read = materialized(store, leaf, lists, expired)
            found[leaf], df[leaf] = read.postings, len(read.postings[0])
            cut = cut or read.cut
    return Prepared(store, found, df, allowed, cut)


def query_of(expression: Expression, each: Sequence[Prepared]) -> Query | None:
    chunks = sum(held.store.chunks for held in each)
    if not chunks:
        return None
    named = leaves(expression)
    unique = tuple(dict.fromkeys(named))
    weight = {leaf: idf(chunks, sum(held.df[leaf] for held in each)) for leaf in unique}
    times = Counter(named)
    bounding = {leaf: weight[leaf] * times[leaf] for leaf in unique}
    return Query(expression, unique, weight, bounding, sum(held.store.tokens for held in each) / chunks)


def searched(held: Prepared, query: Query, best: Best, tuning: Tuning = TUNING, expired: Expired = never) -> Searched:
    lists = {leaf: scored(held.store, query, held.lists[leaf]) for leaf in query.leaves}
    return _searched(Pass(held.store, query, lists, held.allowed), best, tuning, expired)


def found_in(best: Best) -> tuple[FoundChunk, ...]:
    return tuple(
        FoundChunk(ChunkId(chunk), DocumentId(document), score)
        for (chunk, score), document in zip(ranked(best), ints(best.documents), strict=True)
    )
