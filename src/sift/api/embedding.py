from dataclasses import dataclass
from typing import Annotated, Literal

from sift.api.described import LISTED, NAME_LENGTH, Help
from sift.api.indexing import Skipped
from sift.api.requests import Request


@dataclass(frozen=True, slots=True)
class EmbedCollections(Request):
    names: Annotated[
        tuple[str, ...], Help("the collections to embed (default: every active one)"), LISTED, NAME_LENGTH
    ] = ()
    reset: Annotated[bool, Help("forget every collection's vectors and embed them again with the model now set")] = (
        False
    )


@dataclass(frozen=True, slots=True)
class Embedded:
    stored: int
    broken: int
    state: Literal["embedded"] = "embedded"


@dataclass(frozen=True, slots=True)
class CollectionEmbedded:
    name: str
    outcome: Embedded | Skipped


@dataclass(frozen=True, slots=True)
class EmbedResult:
    collections: tuple[CollectionEmbedded, ...]
