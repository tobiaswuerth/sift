from sift.api.links import FindMentions, FindSimilar
from sift.api.search import SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app import mentions, query, search, similar, vsearch
from sift.app.searching import Watching
from sift.interfaces.acts.acting import Act, Acting


def _search(acting: Acting, request: SearchKeywords) -> SearchResult:
    return search.search(acting.settings.home, request, acting.clock)


def _watching(acting: Acting) -> Watching:
    return Watching(report=acting.report, codes=acting.codes)


def _vsearch(acting: Acting, request: SearchVectors) -> SearchResult:
    return vsearch.vsearch(acting.settings.home, request, acting.models(), watching=_watching(acting))


def _query(acting: Acting, request: SearchHybrid) -> SearchResult:
    return query.query(acting.settings.home, request, acting.models(), watching=_watching(acting))


def _mentions(acting: Acting, request: FindMentions) -> SearchResult:
    return mentions.mentions(acting.settings.home, request)


def _similar(acting: Acting, request: FindSimilar) -> SearchResult:
    return similar.similar(acting.settings.home, request, watching=_watching(acting))


SEARCH = Act(SearchKeywords, _search)
VSEARCH = Act(SearchVectors, _vsearch)
QUERY = Act(SearchHybrid, _query)
MENTIONS = Act(FindMentions, _mentions)
SIMILAR = Act(FindSimilar, _similar)
