from dataclasses import dataclass
from typing import Annotated

from sift.api.described import Help, Read, Short
from sift.api.fields import FIELDS, Fields, listed
from sift.api.filters import Filter, written

MOST_HITS = 10_000
QUERY = (
    'what to find: words, "phrases", prefix*, AND, OR, NOT and (parentheses); words side by side must all be in a '
    "passage, and if no passage holds them all, any of them will do"
)
CANDIDATES = (
    "passages each of keyword and semantic search hands the reranker, raised to k if fewer, at most 10,000"
    " (default: 40)"
)
ADAPTIVE = "only the results before the largest drop in score, and five more, instead of always k"
BUDGET = (
    "seconds the search may take, roughly: once they are spent and anything is found, the best found so far, "
    "marked partial (default: no limit)"
)
PATH = 'only documents whose path in the collection matches this glob, e.g. "2024/*"'
FILTERS = (
    "only documents whose front matter passes: KEY=VALUE, KEY!=VALUE, or a range such as created>=2024-06; "
    "repeat for more: values of one key asked alike mean any (=) or none (!=) of them"
)
SORT = "order the k found by a front matter field, e.g. created, or --sort=-created for the newest first"


def _filter(text: str) -> tuple[Filter, ...]:
    return (written(text),)


@dataclass(frozen=True, slots=True, kw_only=True)
class Options:
    k: Annotated[int, Help(f"how many results, at most {MOST_HITS:,} (default: 10)"), Short("-k")] = 10
    collection: Annotated[str | None, Help("search this collection alone (default: every active one)"), Short("-c")] = (
        None
    )
    path: Annotated[str | None, Help(PATH)] = None
    filters: Annotated[tuple[Filter, ...], Help(FILTERS), Short("-f"), Read(_filter)] = ()
    fields: Annotated[tuple[str, ...], Help(FIELDS), Read(listed)] = ()
    sort: Annotated[str | None, Help(SORT)] = None
    all_passages: Annotated[bool, Help("every passage found, not only each document's best")] = False
    compact: Annotated[bool, Help("leave out the text of each passage")] = False


@dataclass(frozen=True, slots=True)
class SearchKeywords(Options):
    query: Annotated[str, Help(QUERY)]
    budget: Annotated[float | None, Help(BUDGET)] = None


@dataclass(frozen=True, slots=True)
class SearchVectors(Options):
    query: Annotated[str, Help("what to find, in words: passages near it in meaning are found")]


@dataclass(frozen=True, slots=True)
class SearchHybrid(Options):
    query: Annotated[str, Help("a question, or a description of what to find")]
    candidates: Annotated[int, Help(CANDIDATES)] = 40
    adaptive: Annotated[bool, Help(ADAPTIVE)] = False


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    title: str
    lines: str
    score: float
    text: str | None
    also: tuple[str, ...] = ()
    fields: Fields | None = None


@dataclass(frozen=True, slots=True)
class Unsearched:
    collection: str
    message: str
    remedy: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[Hit, ...]
    searched: int
    unsearched: tuple[Unsearched, ...]
    partial: bool = False
