import time
from pathlib import Path

from sift.api.search import SearchKeywords, SearchResult
from sift.app import searching
from sift.app.keyword_candidates import Wanted, keyword_result
from sift.app.searching import Clock
from sift.domain.query.grammar import parsed, relaxed
from sift.retrieval.keyword.search import TUNING, Tuning

EMPTY_REMEDY = "say what to find, e.g. dark matter halo"


def search(
    home: Path, request: SearchKeywords, clock: Clock = time.perf_counter, tuning: Tuning = TUNING
) -> SearchResult:
    searching.checked(request)
    searching.checked_query(request.query, EMPTY_REMEDY)
    wanted = Wanted(request.wanted, not request.all_passages, tuning, searching.expired_after(request.budget, clock))
    expression, looser = parsed(request.query), relaxed(request.query)
    return keyword_result(searching.narrowed(home, request), request, wanted, expression, looser)
