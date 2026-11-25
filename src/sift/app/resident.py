import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sift.api.ids import CollectionId

if TYPE_CHECKING:
    from sift.retrieval.keyword.lists import DecodedLists
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


@dataclass(frozen=True, slots=True)
class KeywordsAt:
    store: Path
    collection: CollectionId
    generation: int


class DecodedSource(Protocol):
    def __call__(self, at: KeywordsAt, /) -> "DecodedLists | None": ...


def decoded_each_search(_at: KeywordsAt, /) -> "DecodedLists | None":
    return None


type StoreKey = tuple[Path, CollectionId]


@dataclass(slots=True, eq=False)
class ResidentCodes:
    held: dict[StoreKey, tuple[int, "Coded"]] = field(default_factory=dict[StoreKey, tuple[int, "Coded"]])
    reading: dict[StoreKey, threading.Lock] = field(default_factory=dict[StoreKey, threading.Lock])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _current(self, key: StoreKey, generation: int) -> "Coded | None":
        found = self.held.get(key)
        return found[1] if found is not None and found[0] == generation else None

    def __call__(self, at: CodesAt, read: Callable[[], "Coded"], /) -> "Coded":
        key = (at.store, at.collection)
        with self.lock:
            found = self._current(key, at.generation)
            if found is not None:
                return found
            alone = self.reading.setdefault(key, threading.Lock())
        with alone:
            with self.lock:
                found = self._current(key, at.generation)
                if found is not None:
                    return found
                self.held.pop(key, None)
            fresh = read()
            with self.lock:
                self.held[key] = (at.generation, fresh)
            return fresh

    def kept_only(self, searched: Collection[StoreKey]) -> None:
        with self.lock:
            for gone in self.held.keys() - searched:
                del self.held[gone]
            for gone in self.reading.keys() - searched:
                del self.reading[gone]
