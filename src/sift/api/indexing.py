from dataclasses import dataclass
from typing import Annotated, Literal

from sift.api.described import Help


@dataclass(frozen=True, slots=True)
class IndexCollections:
    names: Annotated[tuple[str, ...], Help("the collections to index (default: every active one)")] = ()


@dataclass(frozen=True, slots=True)
class Indexed:
    files: int
    added: int
    changed: int
    removed: int
    unreadable: int
    state: Literal["indexed"] = "indexed"


@dataclass(frozen=True, slots=True)
class Skipped:
    reason: str
    remedy: str | None
    state: Literal["skipped"] = "skipped"


@dataclass(frozen=True, slots=True)
class CollectionIndexed:
    name: str
    outcome: Indexed | Skipped


@dataclass(frozen=True, slots=True)
class IndexResult:
    collections: tuple[CollectionIndexed, ...]
