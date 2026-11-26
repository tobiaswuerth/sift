from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial

from sift.api.errors import field_at_fault
from sift.api.search import Options, SearchResult
from sift.app import hits
from sift.app.linked import Narrowing
from sift.app.resident import DecodedSource, decoded_each_search
from sift.app.searching import Open, Ranked, Wants, despite_losses, opened
from sift.domain.query.expressions import NOTHING, Expression
from sift.domain.rank.best import Best
from sift.retrieval.keyword.leaves import PHRASE_REMEDY, Expired
from sift.retrieval.keyword.search import (
    Prepared,
    Tuning,
    found_in,
    planned,
    prepared,
    query_of,
    refuse_reading_too_much,
    searched,
)


@dataclass(frozen=True, slots=True)
class Wanted:
    k: int
    by_document: bool
    tuning: Tuning
    expired: Expired
    field: str = "query"
    phrase_remedy: str = PHRASE_REMEDY


def found(stores: Open, wanted: Wanted, expression: Expression) -> Ranked:
    if expression == NOTHING:
        return Ranked(())
    plans = [
        (one, keywords, stores.guarded(one, partial(planned, keywords.store, expression)))
        for one, keywords in stores.keyworded()
        if expression not in keywords.prepared
    ]
    with field_at_fault(wanted.field):
        refuse_reading_too_much([plan for _, _, plan in plans], wanted.tuning, wanted.phrase_remedy)
    fresh: dict[int, Prepared] = {}
    for one, keywords, plan in plans:
        made = stores.guarded(one, partial(prepared, plan, allowed=keywords.allowed, expired=wanted.expired))
        fresh[one.collection.id] = made
        if not made.cut:
            keywords.prepared[expression] = made
    ready = [
        (one, keywords.prepared.get(expression) or fresh[one.collection.id]) for one, keywords in stores.keyworded()
    ]
    query = query_of(expression, [held for _, held in ready])
    cut = any(held.cut for _, held in ready)
    if query is None:
        return Ranked((), cut)
    best = Best(wanted.k, wanted.by_document)
    for one, held in ready:
        after = stores.guarded(one, partial(searched, held, query, best, wanted.tuning, wanted.expired))
        best, cut = after.best, cut or after.cut
    return Ranked(found_in(best), cut)


def _ranked(stores: Open, wanted: Wanted, expressions: tuple[Expression, Expression | None], k: int) -> Ranked:
    expression, looser = expressions
    first = found(stores, replace(wanted, k=k), expression)
    if first.found or looser is None:
        return first
    second = found(stores, replace(wanted, k=k), looser)
    return replace(second, loosened=bool(second.found))


def _in_time(expired: Expired) -> bool:
    return not expired()


def keyword_result(
    narrowing: Narrowing,
    request: Options,
    wanted: Wanted,
    expressions: tuple[Expression, Expression | None],
    decoded: DecodedSource = decoded_each_search,
) -> SearchResult:
    with ExitStack() as cleanup:
        stores = opened(cleanup, narrowing, Wants(keywords=True, decoded=decoded))
        rank = partial(_ranked, stores, wanted, expressions)
        more = partial(_in_time, wanted.expired)
        answered = despite_losses(partial(hits.answered, stores, rank, request, more=more))
        return hits.result(stores, answered, "keywords")
