import math
import time
from collections.abc import Callable
from pathlib import Path

from sift.api.errors import InvalidRequestError
from sift.api.search import SearchKeywords, SearchResult
from sift.app import searching
from sift.app.keyword_candidates import Wanted, keyword_result
from sift.domain.query.grammar import parsed, relaxed
from sift.retrieval.keyword.search import TUNING, Expired, Tuning, never

BUDGET_REMEDY = "give the seconds the search may take, e.g. --budget 0.5"
EMPTY_REMEDY = "say what to find, e.g. sift search 'dark matter halo'"


type Clock = Callable[[], float]


def _expired(budget: float | None, clock: Clock) -> Expired:
    if budget is None:
        return never
    if not math.isfinite(budget) or budget <= 0:
        raise InvalidRequestError(f"--budget {budget} leaves no time to search", BUDGET_REMEDY)
    until = clock() + budget
    return lambda: clock() > until


def search(
    home: Path, request: SearchKeywords, clock: Clock = time.perf_counter, tuning: Tuning = TUNING
) -> SearchResult:
    searching.checked(request)
    searching.checked_query(request.query, EMPTY_REMEDY)
    wanted = Wanted(request.k, not request.all_passages, tuning, _expired(request.budget, clock))
    expression, looser = parsed(request.query), relaxed(request.query)
    return keyword_result(searching.narrowed(home, request), request, wanted, expression, looser)
