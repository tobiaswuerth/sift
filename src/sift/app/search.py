import math
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from sift.api.errors import InvalidRequestError
from sift.api.search import Hit, SearchKeywords, SearchResult
from sift.app import searching
from sift.domain.query.expressions import NOTHING, Expression
from sift.domain.query.grammar import parsed, relaxed
from sift.domain.rank.best import Best
from sift.retrieval.keyword.lists import Opened, opened
from sift.retrieval.keyword.search import (
    TUNING,
    Expired,
    Found,
    Prepared,
    Tuning,
    found_in,
    never,
    prepared,
    query_of,
    searched,
)
from sift.storage.database import Reading
from sift.storage.keywords import refuse_stale

BUDGET_REMEDY = "give the seconds the search may take, e.g. --budget 0.5"


type Clock = Callable[[], float]


type Kept = searching.Kept[Opened]


@dataclass(frozen=True, slots=True)
class Asked:
    request: SearchKeywords
    expression: Expression
    looser: Expression | None
    expired: Expired
    tuning: Tuning


@dataclass(frozen=True, slots=True)
class Ranked:
    found: tuple[Found, ...]
    cut: bool


def _opened(transaction: Reading) -> Opened:
    refuse_stale(transaction)
    return opened(transaction)


def _found(kept: Kept, asked: Asked, expression: Expression) -> Ranked | None:
    ready: list[tuple[searching.Searching[Opened], Prepared]] = []
    for one in list(kept.searching):
        held = kept.guarded(one, partial(prepared, one.held, expression))
        if held is None:
            return None
        ready.append((one, held))
    query = query_of(expression, [held for _, held in ready])
    if query is None:
        return Ranked((), cut=False)
    best, cut = Best(asked.request.k, not asked.request.all_passages), False
    for one, held in ready:
        after = kept.guarded(one, partial(searched, held, query, best, asked.tuning, asked.expired))
        if after is None:
            return None
        best, cut = after.best, cut or after.cut
    return Ranked(found_in(best), cut)


def _searched(kept: Kept, asked: Asked) -> Ranked | None:
    found = Ranked((), cut=False) if asked.expression == NOTHING else _found(kept, asked, asked.expression)
    if found is None or found.found or asked.looser is None:
        return found
    return _found(kept, asked, asked.looser)


def _answered(kept: Kept, asked: Asked) -> tuple[list[tuple[int, Hit]], bool]:
    while True:
        ranked = _searched(kept, asked)
        hits = None if ranked is None else searching.hits(kept, ranked.found, compact=asked.request.compact)
        if ranked is not None and hits is not None:
            return hits, ranked.cut


def _expired(budget: float | None, clock: Clock) -> Expired:
    if budget is None:
        return never
    if not math.isfinite(budget) or budget <= 0:
        raise InvalidRequestError(f"--budget {budget} leaves no time to search", BUDGET_REMEDY)
    until = clock() + budget
    return lambda: clock() > until


def _asked(request: SearchKeywords, clock: Clock, tuning: Tuning) -> Asked:
    searching.checked_k(request.k)
    expired = _expired(request.budget, clock)
    return Asked(request, parsed(request.query), relaxed(request.query), expired, tuning)


def search(
    home: Path, request: SearchKeywords, clock: Clock = time.perf_counter, tuning: Tuning = TUNING
) -> SearchResult:
    asked = _asked(request, clock, tuning)
    collections = searching.collections_of(home, request.collection)
    with ExitStack() as cleanup:
        kept = searching.opened(cleanup, collections, _opened)
        answered, cut = _answered(kept, asked)
        return searching.result(kept, answered, partial=cut)
