from collections.abc import Callable, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.search import MOST_HITS, Options, Unsearched
from sift.app.collections import chosen_or_active
from sift.app.linked import LeftOut, Narrowing, condition_in, narrowing
from sift.domain.query import fields as shown
from sift.domain.query.expressions import Expression
from sift.retrieval.keyword import lists
from sift.retrieval.keyword.search import Prepared
from sift.retrieval.vector.nearest import Coded, coded
from sift.storage.catalog import Collection
from sift.storage.database import Reading, abandon, reading
from sift.storage.keywords import refuse_stale
from sift.storage.passing import Passing
from sift.storage.store import open_store, unavailable_if_unusable
from sift.storage.vectors import EMBED_REMEDY, codes_of, left, model_of, width_held

FEW_REMEDY = "ask for one result or more, e.g. -k 10"
MANY_REMEDY = f"ask for {MOST_HITS:,} results at most, and narrow the query to reach the rest"
UNEMBEDDED_REMEDY = "run sift embed"
EMPTY_QUERY = "an empty query finds nothing"


class ScoredChunk(Protocol):
    @property
    def chunk(self) -> int: ...

    @property
    def score(self) -> float: ...


@dataclass(frozen=True, slots=True)
class Ranked:
    found: Sequence[ScoredChunk]
    cut: bool = False


@dataclass(frozen=True, slots=True)
class Keywords:
    store: lists.Opened
    allowed: Passing | None
    prepared: dict[Expression, Prepared] = field(default_factory=dict[Expression, Prepared])


@dataclass(frozen=True, slots=True)
class Vectors:
    coded: Coded
    allowed: Passing | None = None


@dataclass(frozen=True, slots=True)
class Wants:
    keywords: bool = False
    model: str | None = None
    vectors_required: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class Searching:
    collection: Collection
    transaction: Reading
    keywords: Keywords | None
    vectors: Vectors | None
    resources: ExitStack


class StoreLostError(Exception):
    def __init__(self, collection: str) -> None:
        super().__init__(f"the store of {collection} was lost while searched")


@dataclass(slots=True)
class Open:
    searching: list[Searching] = field(default_factory=list[Searching])
    unsearched: list[Unsearched] = field(default_factory=list[Unsearched])
    ruled_out: int = 0

    def said(self, collection: Collection, error: StoreUnavailableError) -> None:
        self.unsearched.append(Unsearched(collection.name, error.message, error.remedy))

    def dropped(self, each: Searching, error: StoreUnavailableError) -> None:
        self.searching.remove(each)
        self.said(each.collection, error)
        abandon(each.transaction)
        each.resources.close()

    def guarded[T](self, each: Searching, work: Callable[[], T]) -> T:
        try:
            with unavailable_if_unusable(each.collection.store):
                return work()
        except StoreUnavailableError as error:
            self.dropped(each, error)
            raise StoreLostError(each.collection.name) from error

    def closed(self) -> None:
        for one in list(self.searching):
            self.guarded(one, one.resources.close)

    def keyworded(self) -> list[tuple[Searching, Keywords]]:
        return [(one, one.keywords) for one in self.searching if one.keywords is not None]

    def embedded(self) -> list[tuple[Searching, Vectors]]:
        return [(one, one.vectors) for one in self.searching if one.vectors is not None]


def despite_losses[T](work: Callable[[], T]) -> T:
    while True:
        try:
            return work()
        except StoreLostError:
            continue


def checked_k(k: int) -> None:
    if k < 1:
        raise InvalidRequestError(f"-k {k:,} asks for no result", FEW_REMEDY)
    if k > MOST_HITS:
        raise InvalidRequestError(f"-k {k:,} asks for more than {MOST_HITS:,} results", MANY_REMEDY)


def checked(request: Options) -> None:
    checked_k(request.k)
    shown.checked(request.fields, request.sort)


def checked_query(query: str, remedy: str) -> None:
    if not query.strip():
        raise InvalidRequestError(EMPTY_QUERY, remedy)


def narrowed(home: Path, request: Options, left_out: LeftOut | None = None) -> Narrowing:
    return narrowing(home, request, chosen_or_active(home, request.collections), left_out)


def no_vectors() -> Vectors:
    return Vectors(coded(np.zeros(0, np.int64), np.zeros((0, 0), np.uint8), 0))


def embedded_by(model: str, transaction: Reading) -> Vectors | None:
    found = model_of(transaction)
    if found is None:
        return None
    if found != model:
        raise StoreUnavailableError(f"its vectors are of {found}, not of {model}", EMBED_REMEDY.format(model=model))
    ids, bits = codes_of(transaction)
    return Vectors(coded(ids, bits, width_held(transaction) or 0))


def _vectors(model: str, transaction: Reading, wants: Wants) -> Vectors:
    found = embedded_by(model, transaction)
    if found is not None:
        return found
    if wants.vectors_required and left(transaction):
        raise StoreUnavailableError("it holds no vectors yet", UNEMBEDDED_REMEDY)
    return no_vectors()


def _held(transaction: Reading, allowed: Passing | None, wants: Wants) -> tuple[Keywords | None, Vectors | None]:
    if wants.keywords:
        refuse_stale(transaction)
    keywords = Keywords(lists.opened(transaction), allowed) if wants.keywords else None
    vectors = None if wants.model is None else replace(_vectors(wants.model, transaction, wants), allowed=allowed)
    return keywords, vectors


def _searching(one: Collection, asked: Narrowing, wants: Wants) -> Searching | None:
    resources = ExitStack()
    with unavailable_if_unusable(one.store):
        try:
            connection = resources.enter_context(closing(open_store(one.store, one.id)))
            transaction = resources.enter_context(reading(connection))
            found = condition_in(asked, one, transaction)
            if found is not None and found.lets_none:
                resources.close()
                return None
            allowed = None if found is None else Passing(transaction, found)
            return Searching(one, transaction, *_held(transaction, allowed, wants), resources)
        except BaseException:
            resources.close()
            raise


def opened(cleanup: ExitStack, asked: Narrowing, wants: Wants) -> Open:
    held = Open(ruled_out=asked.ruled_out)
    for one in asked.collections:
        try:
            each = _searching(one, asked, wants)
        except StoreUnavailableError as error:
            held.said(one, error)
            continue
        if each is None:
            held.ruled_out += 1
            continue
        cleanup.callback(each.resources.close)
        held.searching.append(each)
    return held
