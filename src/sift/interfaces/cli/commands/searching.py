from types import MappingProxyType

from sift.api.links import FindMentions, FindSimilar
from sift.api.search import SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app import mentions, query, search, similar, vsearch
from sift.app.searching import Watching
from sift.interfaces.cli.commands.making import needing_collections, searched
from sift.interfaces.cli.commands.models import models_of
from sift.interfaces.cli.context import Context


def _search(context: Context, request: SearchKeywords) -> SearchResult:
    return search.search(context.settings.home, request, context.clock)


def _vsearch(context: Context, request: SearchVectors) -> SearchResult:
    return vsearch.vsearch(context.settings.home, request, models_of(context), watching=Watching(report=context.report))


def _query(context: Context, request: SearchHybrid) -> SearchResult:
    return query.query(context.settings.home, request, models_of(context), watching=Watching(report=context.report))


def _mentions(context: Context, request: FindMentions) -> SearchResult:
    return mentions.mentions(context.settings.home, request)


def _similar(context: Context, request: FindSimilar) -> SearchResult:
    return similar.similar(context.settings.home, request)


COMMANDS = MappingProxyType(
    {
        "search": searched(SearchKeywords, _search, "keyword"),
        "vsearch": searched(SearchVectors, _vsearch, "semantic"),
        "query": searched(SearchHybrid, _query, "hybrid"),
        "mentions": needing_collections(searched(FindMentions, _mentions, "keyword")),
        "similar": needing_collections(searched(FindSimilar, _similar, "semantic")),
    }
)
