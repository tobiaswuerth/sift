import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sift.api.ids import CollectionId

if TYPE_CHECKING:
    from sift.retrieval.vector.nearest import Coded


@dataclass(frozen=True, slots=True)
class CodesAt:
    store: Path
    collection: CollectionId
    generation: int


class CodeSource(Protocol):
    def __call__(self, at: CodesAt, read: Callable[[], "Coded"], /) -> "Coded": ...


def read_each_time(_at: CodesAt, read: Callable[[], "Coded"], /) -> "Coded":
    return read()


@dataclass(slots=True, eq=False)
class ResidentCodes:
    held: dict[tuple[Path, CollectionId], tuple[int, "Coded"]] = field(
        default_factory=dict[tuple[Path, CollectionId], tuple[int, "Coded"]]
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, at: CodesAt, read: Callable[[], "Coded"], /) -> "Coded":
        with self.lock:
            found = self.held.get((at.store, at.collection))
            if found is not None and found[0] == at.generation:
                return found[1]
            self.held.pop((at.store, at.collection), None)
            fresh = read()
            self.held[at.store, at.collection] = (at.generation, fresh)
            return fresh

    def kept_only(self, searched: Collection[tuple[Path, CollectionId]]) -> None:
        with self.lock:
            for gone in self.held.keys() - searched:
                del self.held[gone]
