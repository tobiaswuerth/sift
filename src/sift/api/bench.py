from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from sift.api.described import LISTED, NAME_LENGTH, PATH_LENGTH, AbsolutePath, Help
from sift.api.requests import Request

CASES = (
    'a JSON file of questions and the documents each should find: [{"query": "...", "relevant": ["collection/path", '
    '...]}, ...], or the documents with grades of relevance above 0, {"collection/path": 2, ...}'
)
SEARCHED = "search this collection; several: each of them (default: every active one)"


class Mode(StrEnum):
    KEYWORD = "search"
    SEMANTIC = "vsearch"
    HYBRID = "query"


@dataclass(frozen=True, slots=True)
class MeasureSearches(Request):
    file: Annotated[str, Help(CASES), AbsolutePath(), PATH_LENGTH]
    collections: Annotated[tuple[str, ...], Help(SEARCHED), LISTED, NAME_LENGTH] = ()


@dataclass(frozen=True, slots=True)
class ModeMeasured:
    mode: Mode
    ndcg: float
    mrr: float
    recall: float
    ms_per_query: float


@dataclass(frozen=True, slots=True)
class SearchesMeasured:
    cases: int
    modes: tuple[ModeMeasured, ...]
