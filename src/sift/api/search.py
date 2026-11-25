from dataclasses import dataclass
from typing import Annotated

from sift.api.described import (
    FIELD_LENGTH,
    LISTED,
    NAME_LENGTH,
    PATH_LENGTH,
    SECONDS,
    TEXT_LENGTH,
    Bounds,
    Composed,
    Help,
)
from sift.api.errors import Problem, SiftError
from sift.api.fields import FIELDS, Fields
from sift.api.filters import Filter
from sift.api.requests import Request

MOST_HITS = 10_000
MOST_LINKS = 100
SCORE_DIGITS = 7
QUERY = (
    'what to find: words, "phrases", prefix*, AND, OR, NOT and (parentheses); words side by side must all be in a '
    "passage, and if no passage holds them all, any of them will do"
)
CANDIDATES = "passages each of keyword and semantic search hands the reranker, raised to k if fewer"
ADAPTIVE = "only the results before the largest drop in score, and five more, instead of always k"
BUDGET = (
    "seconds the search may take, roughly: once they are spent and anything is found, the best found so far, "
    "marked partial (default: no limit)"
)
PATH = 'only documents whose path in the collection matches this glob, e.g. "2024/*"'
FILTERS = (
    "only documents whose front matter passes each filter: a field, an operator (=, !=, <, <=, >, >=) and its "
    "values; values of one field asked alike mean any (=) or none (!=) of them"
)
SORT = "order the k found by a front matter field, e.g. created, or -created for the newest first"
LINKS_TO = (
    'only documents linking this note: its id, or its name as links write it, e.g. "Acme AG"; '
    "several: documents linking any of them"
)
OFFSET = "leave out this many results first, to page through them: offset 10 and k 10 give results 11 to 20"
LINKED_FROM = "only documents this note links to, named by its id or as links write it; several: any of them"


@dataclass(frozen=True, slots=True, kw_only=True)
class Options(Request):
    k: Annotated[int, Help("how many results"), Bounds(1, MOST_HITS)] = 10
    collections: Annotated[
        tuple[str, ...],
        Help("search this collection; several: each of them (default: every active one)"),
        LISTED,
        NAME_LENGTH,
    ] = ()
    path: Annotated[str | None, Help(PATH), PATH_LENGTH] = None
    filters: Annotated[tuple[Filter, ...], Help(FILTERS), LISTED] = ()
    fields: Annotated[tuple[str, ...], Help(FIELDS), Composed(), LISTED, FIELD_LENGTH] = ()
    sort: Annotated[str | None, Help(SORT), Composed(), FIELD_LENGTH] = None
    links_to: Annotated[tuple[str, ...], Help(LINKS_TO), Bounds(0, MOST_LINKS), PATH_LENGTH] = ()
    linked_from: Annotated[tuple[str, ...], Help(LINKED_FROM), Bounds(0, MOST_LINKS), PATH_LENGTH] = ()
    all_passages: Annotated[
        bool, Help("every passage found, not only each document's best, and copies of one work apart")
    ] = False
    compact: Annotated[bool, Help("leave out the text of each passage")] = False
    adaptive: Annotated[bool, Help(ADAPTIVE)] = False
    offset: Annotated[int, Help(OFFSET), Bounds(0, MOST_HITS - 1)] = 0
    budget: Annotated[float | None, Help(BUDGET), SECONDS] = None

    @property
    def wanted(self) -> int:
        return self.offset + self.k


@dataclass(frozen=True, slots=True)
class SearchKeywords(Options):
    query: Annotated[str, Help(QUERY), TEXT_LENGTH]


@dataclass(frozen=True, slots=True)
class SearchVectors(Options):
    query: Annotated[str, Help("what to find, in words: passages near it in meaning are found"), TEXT_LENGTH]


@dataclass(frozen=True, slots=True)
class SearchHybrid(Options):
    query: Annotated[str, Help("a question, or a description of what to find"), TEXT_LENGTH]
    candidates: Annotated[int, Help(CANDIDATES), Bounds(1, MOST_HITS)] = 40


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    collection: str
    path: str
    title: str
    first_line: int
    last_line: int
    score: float
    text: str | None
    also: tuple[str, ...] = ()
    fields: Fields | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", float(f"{self.score:.{SCORE_DIGITS}g}"))

    @property
    def lines(self) -> str:
        return f"{self.first_line}-{self.last_line}"


@dataclass(frozen=True, slots=True)
class Unsearched:
    collection: str
    message: str
    remedy: str | None
    code: Problem


def unsearched(collection: str, error: SiftError) -> Unsearched:
    return Unsearched(collection, error.message, error.remedy, error.code)


@dataclass(frozen=True, slots=True)
class Left:
    collection: str
    message: str
    remedy: str | None
    code: Problem


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[Hit, ...]
    searched: int
    unsearched: tuple[Unsearched, ...]
    partial: bool = False
    ruled_out: int = 0
    left_out: tuple[Left, ...] = ()
