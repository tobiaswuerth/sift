import math
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import Reporter
from sift.api.search import MOST_HITS, Left, Options, Unsearched, unsearched
from sift.app.collections import chosen_or_active
from sift.app.linked import LeftOut, Narrowing, condition_in, narrowing
from sift.app.loading import unreported
from sift.domain.query import fields as shown
from sift.domain.query.expressions import Expression
from sift.retrieval.keyword import lists
from sift.retrieval.keyword.search import Expired, Prepared, never
from sift.retrieval.vector.nearest import Coded, coded
from sift.storage.catalog import Collection
from sift.storage.database import Reading, abandon, reading
from sift.storage.keywords import refuse_stale
from sift.storage.passing import Passing
from sift.storage.store import connected, unavailable_if_unusable
from sift.storage.vector_meta import left, model_of, width_held
from sift.storage.vectors import EMBED_REMEDY, codes_of, refuse_recipe

UNEMBEDDED_REMEDY = "run sift embed"
EMPTY_QUERY = "an empty query finds nothing"
BUDGET_REMEDY = "give the seconds the search may take, e.g. 0.5"
OUT_OF_TIME = "not searched: the budget was spent before its vectors were read"
OUT_OF_TIME_REMEDY = "give the search more seconds, or none"

type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class Watching:
    report: Reporter = unreported
    clock: Clock = time.perf_counter


WATCHING = Watching()


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


@dataclass(eq=False, frozen=True, slots=True)
class Vectors:
    coded: Coded
    allowed: Passing | None = None
    unembedded: int = 0
    unlike: StoreUnavailableError | None = None


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
    left_out: list[Left] = field(default_factory=list[Left])
    cut: bool = False

    def said(self, collection: Collection, error: StoreUnavailableError) -> None:
        self.unsearched.append(unsearched(collection.name, error))

    def dropped(self, each: Searching, error: StoreUnavailableError) -> None:
        self.searching.remove(each)
        self.said(each.collection, error)
        abandon(each.transaction)
        each.resources.close()

    def by_keywords_alone(self, each: Searching, error: StoreUnavailableError) -> None:
        allowed = None if each.vectors is None else each.vectors.allowed
        vectors = replace(no_vectors(), allowed=allowed, unlike=error)
        self.searching[self.searching.index(each)] = replace(each, vectors=vectors)
        self.left_out.append(keywords_alone(each.collection.name, error))

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


def _paged(request: Options) -> None:
    if request.wanted > MOST_HITS:
        asked = f"offset {request.offset:,} and k {request.k:,} ask for {request.wanted:,} results"
        said = f"{asked}, more than {MOST_HITS:,}"
        raise InvalidRequestError(said, f"ask for at most {MOST_HITS:,} results in all", "offset")
    if request.offset and request.sort is not None:
        said = "a sorted search orders the best k alone, so it has no pages"
        raise InvalidRequestError(said, "ask for more with k instead, or page without sort", "offset")
    if request.offset and request.adaptive:
        said = "an adaptive search keeps what stands out of all it finds, so it has no pages"
        raise InvalidRequestError(said, "ask for more with k instead, or page without adaptive", "offset")


def checked(request: Options) -> None:
    shown.checked(request.fields, request.sort)
    _paged(request)


def expired_after(budget: float | None, clock: Clock) -> Expired:
    if budget is None:
        return never
    if not math.isfinite(budget) or budget <= 0:
        raise InvalidRequestError(f"{budget} seconds leave no time to search", BUDGET_REMEDY, "budget")
    until = clock() + budget
    return lambda: clock() > until


def checked_query(query: str, remedy: str) -> None:
    if not query.strip():
        raise InvalidRequestError(EMPTY_QUERY, remedy, "query")


def narrowed(home: Path, request: Options, left_out: LeftOut | None = None) -> Narrowing:
    return narrowing(home, request, chosen_or_active(home, request.collections), left_out)


def no_vectors() -> Vectors:
    return Vectors(coded(np.zeros(0, np.int64), np.zeros((0, 0), np.uint8), 0))


def embedded_by(model: str, transaction: Reading) -> Vectors | None:
    found = model_of(transaction)
    if found is None:
        return None
    if found != model:
        said = f"its vectors are of {found}, not of {model}"
        raise StoreUnavailableError(said, EMBED_REMEDY.format(model=model), code=Problem.OTHER_MODEL)
    refuse_recipe(transaction, model)
    ids, bits = codes_of(transaction)
    return Vectors(coded(ids, bits, width_held(transaction) or 0))


def _vectors(model: str, transaction: Reading, wants: Wants) -> Vectors:
    try:
        found = embedded_by(model, transaction)
    except StoreUnavailableError as error:
        if wants.vectors_required or error.code is not Problem.OTHER_MODEL:
            raise
        return replace(no_vectors(), unlike=error)
    if found is not None:
        return replace(found, unembedded=left(transaction))
    if wants.vectors_required and left(transaction):
        raise StoreUnavailableError("it holds no vectors yet", UNEMBEDDED_REMEDY, code=Problem.NOT_EMBEDDED)
    return replace(no_vectors(), unembedded=left(transaction))


def _held(transaction: Reading, allowed: Passing | None, wants: Wants) -> tuple[Keywords | None, Vectors | None]:
    if wants.keywords:
        refuse_stale(transaction)
    keywords = Keywords(lists.opened(transaction), allowed) if wants.keywords else None
    if wants.model is None:
        return keywords, None
    return keywords, replace(_vectors(wants.model, transaction, wants), allowed=allowed)


def keywords_alone(collection: str, error: StoreUnavailableError) -> Left:
    return Left(collection, f"{error.message}: searched by keyword alone", error.remedy, error.code)


def _unembedded(collection: Collection, count: int) -> Left:
    holds, passed = ("holds", "it") if count == 1 else ("hold", "them")
    said = f"{count:,} of its passages {holds} no vector yet: a search by meaning passes {passed} over"
    return Left(collection.name, said, UNEMBEDDED_REMEDY, Problem.NOT_EMBEDDED)


def _notes(collection: Collection, vectors: Vectors) -> list[Left]:
    if vectors.unlike is not None:
        return [keywords_alone(collection.name, vectors.unlike)]
    return [_unembedded(collection, vectors.unembedded)] if vectors.unembedded else []


def _out_of_time(held: Open, one: Collection, wants: Wants, expired: Expired) -> bool:
    if wants.model is None or not held.searching or not expired():
        return False
    held.left_out.append(Left(one.name, OUT_OF_TIME, OUT_OF_TIME_REMEDY, Problem.OUT_OF_TIME))
    held.cut = True
    return True


def _searching(one: Collection, asked: Narrowing, wants: Wants) -> Searching | None:
    resources = ExitStack()
    with unavailable_if_unusable(one.store):
        try:
            connection = resources.enter_context(connected(one.store, one.id))
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


def opened(cleanup: ExitStack, asked: Narrowing, wants: Wants, expired: Expired = never) -> Open:
    held = Open(ruled_out=asked.ruled_out)
    for one in asked.collections:
        if _out_of_time(held, one, wants, expired):
            continue
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
        if each.vectors is not None:
            held.left_out += _notes(one, each.vectors)
    return held
