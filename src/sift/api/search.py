from dataclasses import dataclass
from typing import Annotated

from sift.api.described import Help, Short

MOST_HITS = 10_000
QUERY = (
    'what to find: words, "phrases", prefix*, AND, OR, NOT and (parentheses); words side by side must all be in a '
    "passage, and if no passage holds them all, any of them will do"
)


BUDGET = (
    "seconds the search may take, roughly: once they are spent and anything is found, the best found so far, "
    "marked partial (default: no limit)"
)


@dataclass(frozen=True, slots=True)
class SearchKeywords:
    query: Annotated[str, Help(QUERY)]
    k: Annotated[int, Help(f"how many results, at most {MOST_HITS:,} (default: 10)"), Short("-k")] = 10
    collection: Annotated[str | None, Help("search this collection alone (default: every active one)"), Short("-c")] = (
        None
    )
    all_passages: Annotated[bool, Help("every passage that matches, not only each document's best")] = False
    compact: Annotated[bool, Help("leave out the text of each passage")] = False
    budget: Annotated[float | None, Help(BUDGET)] = None


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    title: str
    lines: str
    score: float
    text: str | None


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
