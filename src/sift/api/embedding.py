from dataclasses import dataclass
from typing import Annotated, Literal

from sift.api.described import Help
from sift.api.indexing import Skipped


@dataclass(frozen=True, slots=True)
class EmbedCollections:
    names: Annotated[tuple[str, ...], Help("the collections to embed (default: every active one)")] = ()
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
