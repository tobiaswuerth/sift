import math
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from sift.api.errors import InvalidRequestError
from sift.api.search import Hit, Options, SearchKeywords, SearchResult
from sift.app import searching
from sift.app.linked import Narrowing
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
from sift.storage.passing import Passing

BUDGET_REMEDY = "give the seconds the search may take, e.g. --budget 0.5"


type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class Keywords:
    store: Opened
    allowed: Passing | None


type Kept = searching.Kept[Keywords]


@dataclass(frozen=True, slots=True)
class Wanted:
    k: int
    by_document: bool
    tuning: Tuning
    expired: Expired


@dataclass(frozen=True, slots=True)
class Asked:
    request: Options
    expression: Expression
    looser: Expression | None
    wanted: Wanted
    narrowing: Narrowing


@dataclass(frozen=True, slots=True)
class Ranked:
    found: tuple[Found, ...]
    cut: bool


def keywords(transaction: Reading, allowed: Passing | None) -> Keywords:
    refuse_stale(transaction)
    return Keywords(opened(transaction), allowed)


def found[S](
    kept: searching.Kept[S], keywords_of: Callable[[S], Keywords], wanted: Wanted, expression: Expression
) -> Ranked | None:
    if expression == NOTHING:
        return Ranked((), cut=False)
    ready: list[tuple[searching.Searching[S], Prepared]] = []
    for one in list(kept.searching):
        store = keywords_of(one.held)
        held = kept.guarded(one, partial(prepared, store.store, expression, wanted.tuning, allowed=store.allowed))
        if held is None:
            return None
        ready.append((one, held))
    query = query_of(expression, [held for _, held in ready])
    if query is None:
        return Ranked((), cut=False)
    best, cut = Best(wanted.k, wanted.by_document), False
    for one, held in ready:
        after = kept.guarded(one, partial(searched, held, query, best, wanted.tuning, wanted.expired))
        if after is None:
            return None
        best, cut = after.best, cut or after.cut
    return Ranked(found_in(best), cut)


def _itself(store: Keywords) -> Keywords:
    return store


def _searched(kept: Kept, asked: Asked) -> Ranked | None:
    first = found(kept, _itself, asked.wanted, asked.expression)
    if first is None or first.found or asked.looser is None:
        return first
    return found(kept, _itself, asked.wanted, asked.looser)


def _ranked(kept: Kept, asked: Asked, wanted: int) -> Ranked | None:
    return _searched(kept, replace(asked, wanted=replace(asked.wanted, k=wanted)))


def _answered(kept: Kept, asked: Asked) -> tuple[list[tuple[int, Hit]], bool]:
    while True:
        found = searching.answered(kept, partial(_ranked, kept, asked), asked.request)
        if found is not None:
            return found


def _expired(budget: float | None, clock: Clock) -> Expired:
    if budget is None:
        return never
    if not math.isfinite(budget) or budget <= 0:
        raise InvalidRequestError(f"--budget {budget} leaves no time to search", BUDGET_REMEDY)
    until = clock() + budget
    return lambda: clock() > until


def _asked(home: Path, request: SearchKeywords, clock: Clock, tuning: Tuning) -> Asked:
    searching.checked(request)
    wanted = Wanted(request.k, not request.all_passages, tuning, _expired(request.budget, clock))
    expression, looser = parsed(request.query), relaxed(request.query)
    return Asked(request, expression, looser, wanted, searching.narrowed(home, request))


def search(
    home: Path, request: SearchKeywords, clock: Clock = time.perf_counter, tuning: Tuning = TUNING
) -> SearchResult:
    return searched_as(_asked(home, request, clock, tuning))


def searched_as(asked: Asked) -> SearchResult:
    with ExitStack() as cleanup:
        kept = searching.opened(cleanup, asked.narrowing, keywords)
        answered, cut = _answered(kept, asked)
        return searching.result(kept, answered, partial=cut)
