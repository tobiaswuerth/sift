from pathlib import Path

from sift.api.search import SearchKeywords, SearchResult
from sift.app import searching
from sift.app.keyword_candidates import Wanted, keyword_result
from sift.app.searching import WATCHING, Watching
from sift.domain.query.grammar import parsed, relaxed
from sift.retrieval.keyword.search import TUNING, Tuning

EMPTY_REMEDY = "say what to find, e.g. dark matter halo"


def search(home: Path, request: SearchKeywords, watching: Watching = WATCHING, tuning: Tuning = TUNING) -> SearchResult:
    searching.checked(request)
    searching.checked_query(request.query, EMPTY_REMEDY)
    wanted = Wanted(
        request.wanted, not request.all_passages, tuning, searching.expired_after(request.budget, watching.clock)
    )
    expression, looser = parsed(request.query), relaxed(request.query)
    narrowing = searching.narrowed(home, request)
    return keyword_result(narrowing, request, wanted, (expression, looser), watching.decoded)
