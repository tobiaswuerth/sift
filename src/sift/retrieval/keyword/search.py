from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sift.api.ids import ChunkId, DocumentId
from sift.domain.arrays import ints
from sift.domain.errors import DamagedError
from sift.domain.query.expressions import Expression, Leaf, Phrase, Prefix, Term, leaves
from sift.domain.rank import pruned
from sift.domain.rank.best import Best, ranked
from sift.domain.rank.bm25 import OVERCOUNTED, idf
from sift.retrieval.allowed import AllowedDocuments
from sift.retrieval.keyword.leaves import (
    PHRASE_REMEDY,
    Expired,
    Limit,
    materialized,
    never,
    reads_of,
    refused,
    size,
)
from sift.retrieval.keyword.lists import Opened, PostingArrays, StoredList, stored
from sift.retrieval.keyword.scoring import ListReads, Query, plan_of
from sift.storage.keyword_lists import gone, living, prefixed


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
    words: Mapping[Prefix, Mapping[str, int]]


def planned(store: Opened, expression: Expression, tuning: Tuning = TUNING) -> Planned:
    unique = tuple(dict.fromkeys(leaves(expression)))
    lists = stored(store, _tokens(unique))
    most = tuning.most_expansions
    words = {leaf: prefixed(store.transaction, leaf.text, most) for leaf in unique if isinstance(leaf, Prefix)}
    reads: dict[Phrase | Prefix, int] = {leaf: sum(held.values()) for leaf, held in words.items()}
    reads |= {leaf: reads_of(store, leaf, lists) for leaf in unique if isinstance(leaf, Phrase)}
    return Planned(store, unique, lists, reads, words)


def refuse_reading_too_much(
    plans: Sequence[Planned], tuning: Tuning = TUNING, phrase_remedy: str = PHRASE_REMEDY
) -> None:
    reads: Counter[Phrase | Prefix] = Counter()
    expanded: Counter[Phrase | Prefix] = Counter()
    for plan in plans:
        reads.update(plan.reads)
        expanded.update({leaf: len(held) for leaf, held in plan.words.items()})
    if (total := expanded.total()) > tuning.most_expansions:
        limit = Limit(total, tuning.most_expansions, "word", "stands for", "expands", stores=len(plans), cut=True)
        refused(expanded, limit)
    if (total := reads.total()) > tuning.most_read:
        refused(reads, Limit(total, tuning.most_read, "passage", "reads", "reads", phrase_remedy, len(plans)))


def prepared(plan: Planned, *, allowed: AllowedDocuments | None = None, expired: Expired = never) -> Prepared:
    store, lists = plan.store, plan.lists
    deleted = gone(store.transaction, [leaf.text for leaf in plan.leaves if isinstance(leaf, Term)])
    found: ReadLists = {}
    df: dict[Leaf, int] = {}
    cut = False
    for leaf in plan.leaves:
        if isinstance(leaf, Term):
            found[leaf], df[leaf] = lists[leaf.text], living(size(lists[leaf.text]), deleted.get(leaf.text, 0))
        else:
            read = materialized(store, leaf, lists, expired)
            found[leaf], df[leaf] = read.postings, len(read.postings[0])
            cut = cut or read.cut
    if any(count > store.chunks for count in df.values()):
        raise DamagedError(OVERCOUNTED)
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
    plan, parts = plan_of(held.store, query, held.lists, best, (tuning.first_batch, tuning.most_batch))
    reads = ListReads(held.store, parts, best, held.allowed, expired)
    cut, tables = pruned.scored(plan, reads)
    for segment, table in zip(held.store.segments, tables, strict=True):
        held.store.decoded.lengths[segment.id] = table
    return Searched(reads.best, cut)


def found_in(best: Best) -> tuple[FoundChunk, ...]:
    return tuple(
        FoundChunk(ChunkId(chunk), DocumentId(document), score)
        for (chunk, score), document in zip(ranked(best), ints(best.documents), strict=True)
    )
