from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Protocol

from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.ids import ChunkId, owner
from sift.api.search import MOST_HITS, Hit, SearchResult, Unsearched
from sift.app.collections import chosen_or_active
from sift.storage.catalog import Collection
from sift.storage.database import Reading, abandon, reading
from sift.storage.documents import passages, texts
from sift.storage.store import open_store, unavailable_if_damaged

FEW_REMEDY = "ask for one result or more, e.g. -k 10"
MANY_REMEDY = f"ask for {MOST_HITS:,} results at most, and narrow the query to reach the rest"


class Scored(Protocol):
    @property
    def chunk(self) -> int: ...

    @property
    def score(self) -> float: ...


@dataclass(frozen=True, slots=True)
class Searching[S]:
    collection: Collection
    transaction: Reading
    held: S
    resources: ExitStack


@dataclass(slots=True)
class Kept[S]:
    searching: list[Searching[S]] = field(default_factory=list[Searching[S]])
    unsearched: list[Unsearched] = field(default_factory=list[Unsearched])

    def said(self, collection: Collection, error: StoreUnavailableError) -> None:
        self.unsearched.append(Unsearched(collection.name, error.message, error.remedy))

    def guarded[T](self, each: Searching[S], work: Callable[[], T]) -> T | None:
        try:
            with unavailable_if_damaged(each.collection.store):
                return work()
        except StoreUnavailableError as error:
            self.searching.remove(each)
            self.said(each.collection, error)
            abandon(each.transaction)
            each.resources.close()
            return None


def checked_k(k: int) -> None:
    if k < 1:
        raise InvalidRequestError(f"-k {k:,} asks for no result", FEW_REMEDY)
    if k > MOST_HITS:
        raise InvalidRequestError(f"-k {k:,} asks for more than {MOST_HITS:,} results", MANY_REMEDY)


def collections_of(home: Path, name: str | None) -> tuple[Collection, ...]:
    return chosen_or_active(home, () if name is None else (name,))


def _searching[S](one: Collection, hold: Callable[[Reading], S]) -> Searching[S]:
    resources = ExitStack()
    with unavailable_if_damaged(one.store):
        try:
            connection = resources.enter_context(closing(open_store(one.store, one.id)))
            transaction = resources.enter_context(reading(connection))
            return Searching(one, transaction, hold(transaction), resources)
        except BaseException:
            resources.close()
            raise


def opened[S](cleanup: ExitStack, collections: Sequence[Collection], hold: Callable[[Reading], S]) -> Kept[S]:
    kept = Kept[S]()
    for one in collections:
        try:
            each = _searching(one, hold)
        except StoreUnavailableError as error:
            kept.said(one, error)
            continue
        cleanup.callback(each.resources.close)
        kept.searching.append(each)
    return kept


def _read[S](
    one: Searching[S], chunks: Sequence[ChunkId], scores: Mapping[int, float], *, compact: bool
) -> dict[int, Hit]:
    held = passages(one.transaction, chunks)
    whole = {} if compact else texts(one.transaction, sorted({passage.document for passage in held.values()}))
    return {
        chunk: Hit(
            f"{one.collection.name}/{passage.path}",
            passage.title,
            f"{passage.first_line}-{passage.last_line}",
            scores[chunk],
            None if compact else whole[passage.document][passage.start : passage.start + passage.length],
        )
        for chunk, passage in held.items()
    }


def hits[S](kept: Kept[S], found: Sequence[Scored], *, compact: bool) -> list[tuple[int, Hit]] | None:
    scores = {each.chunk: each.score for each in found}
    shown: dict[int, Hit] = {}
    for one in list(kept.searching):
        mine = [ChunkId(each.chunk) for each in found if owner(each.chunk) == one.collection.id]
        read = kept.guarded(one, partial(_read, one, mine, scores, compact=compact)) if mine else {}
        if read is None:
            return None
        shown |= read
    return [(each.chunk, shown[each.chunk]) for each in found]


def result[S](kept: Kept[S], answered: Sequence[tuple[int, Hit]], *, partial: bool) -> SearchResult:
    for one in list(kept.searching):
        kept.guarded(one, one.resources.close)
    live = {one.collection.id for one in kept.searching}
    found = tuple(hit for chunk, hit in answered if owner(chunk) in live)
    return SearchResult(found, len(kept.searching), tuple(kept.unsearched), partial=partial)
