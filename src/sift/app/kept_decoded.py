import threading
from collections import OrderedDict
from collections.abc import Callable, Collection, Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from sift.api.ids import CollectionId
from sift.app.resident import KeywordsAt
from sift.domain.arrays import Ids
from sift.retrieval.keyword.lists import DecodedLists, Dense, PostingArrays, held_bytes

KEPT_BYTES = 512 << 20
ENTRY_BYTES = 512
LENGTHS = "lengths"
BLOCKS = "blocks"

type Decoded = Dense | PostingArrays
type EntryKey = tuple[Path, CollectionId, int, str, Hashable]


@dataclass(slots=True, eq=False)
class KeptDecoded:
    budget: int = KEPT_BYTES
    entries: OrderedDict[EntryKey, tuple[Decoded, int]] = field(
        default_factory=OrderedDict[EntryKey, tuple[Decoded, int]]
    )
    generations: dict[tuple[Path, CollectionId], int] = field(default_factory=dict[tuple[Path, CollectionId], int])
    size: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, at: KeywordsAt, /) -> DecodedLists:
        with self.lock:
            if self.generations.get((at.store, at.collection)) != at.generation:
                self._dropped(lambda key: key[:2] == (at.store, at.collection))
                self.generations[at.store, at.collection] = at.generation
        return DecodedLists(_KeptView(self, at, LENGTHS), _KeptView(self, at, BLOCKS))

    def got(self, key: EntryKey) -> Decoded | None:
        with self.lock:
            found = self.entries.get(key)
            if found is None:
                return None
            self.entries.move_to_end(key)
            return found[0]

    def put(self, at: KeywordsAt, key: EntryKey, value: Decoded) -> None:
        size = held_bytes(value) + ENTRY_BYTES
        with self.lock:
            if size > self.budget or self.generations.get((at.store, at.collection)) != at.generation:
                return
            replaced = self.entries.pop(key, None)
            self.size += size - (0 if replaced is None else replaced[1])
            self.entries[key] = (value, size)
            while self.size > self.budget:
                _, (_, dropped) = self.entries.popitem(last=False)
                self.size -= dropped

    def kept_only(self, searched: Collection[tuple[Path, CollectionId]]) -> None:
        with self.lock:
            for gone in self.generations.keys() - searched:
                del self.generations[gone]
            self._dropped(lambda key: (key[0], key[1]) not in searched)

    def _dropped(self, chosen: Callable[[EntryKey], bool]) -> None:
        for key in [key for key in self.entries if chosen(key)]:
            self.size -= self.entries.pop(key)[1]


@dataclass(frozen=True, slots=True)
class _KeptView[K: Hashable, V: (Dense, tuple[Ids, Ids])]:
    kept: KeptDecoded
    at: KeywordsAt
    kind: str

    def _key(self, key: K) -> EntryKey:
        return (self.at.store, self.at.collection, self.at.generation, self.kind, key)

    def get(self, key: K, /) -> V | None:
        return cast("V | None", self.kept.got(self._key(key)))

    def __setitem__(self, key: K, value: V, /) -> None:
        self.kept.put(self.at, self._key(key), value)
