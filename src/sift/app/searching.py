from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Protocol

from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.ids import ChunkId, owner
from sift.api.search import MOST_HITS, Hit, Options, SearchResult, Unsearched
from sift.app.collections import chosen_or_active
from sift.app.linked import LeftOut, Narrowing, condition_in, narrowing
from sift.domain.query import fields as shown
from sift.domain.rank.hybrid import work_of
from sift.domain.rank.works import one_per_work
from sift.domain.text.records import record_path
from sift.storage.catalog import Collection
from sift.storage.database import Reading, abandon, reading
from sift.storage.documents import fields_by_document, passages, texts
from sift.storage.passing import Passing
from sift.storage.store import open_store, unavailable_if_unusable

WIDER = 2
FEW_REMEDY = "ask for one result or more, e.g. -k 10"
MANY_REMEDY = f"ask for {MOST_HITS:,} results at most, and narrow the query to reach the rest"


class Scored(Protocol):
    @property
    def chunk(self) -> int: ...

    @property
    def score(self) -> float: ...


@dataclass(frozen=True, slots=True, eq=False)
class Searching[S]:
    collection: Collection
    transaction: Reading
    held: S
    resources: ExitStack


class StoreLostError(Exception):
    def __init__(self, collection: str) -> None:
        super().__init__(f"the store of {collection} was lost while searched")


@dataclass(slots=True)
class Kept[S]:
    searching: list[Searching[S]] = field(default_factory=list[Searching[S]])
    unsearched: list[Unsearched] = field(default_factory=list[Unsearched])
    ruled_out: int = 0

    def said(self, collection: Collection, error: StoreUnavailableError) -> None:
        self.unsearched.append(Unsearched(collection.name, error.message, error.remedy))

    def dropped(self, each: Searching[S], error: StoreUnavailableError) -> None:
        self.searching.remove(each)
        self.said(each.collection, error)
        abandon(each.transaction)
        each.resources.close()

    def guarded[T](self, each: Searching[S], work: Callable[[], T]) -> T:
        try:
            with unavailable_if_unusable(each.collection.store):
                return work()
        except StoreUnavailableError as error:
            self.dropped(each, error)
            raise StoreLostError(each.collection.name) from error

    def closed(self) -> None:
        for one in list(self.searching):
            self.guarded(one, one.resources.close)


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


def narrowed(home: Path, request: Options, left_out: LeftOut | None = None) -> Narrowing:
    return narrowing(home, request, collections_of(home, request.collection), left_out)


type Hold[S] = Callable[[Reading, Passing | None], S]


def fielded(request: Options) -> bool:
    return bool(request.fields) or request.sort is not None


def presented(answered: Sequence[tuple[int, Hit]], request: Options) -> list[tuple[int, Hit]]:
    held = [hit.fields or {} for _, hit in answered]
    order = range(len(answered)) if request.sort is None else shown.order(held, request.sort)
    wanted = request.fields
    return [
        (answered[at][0], replace(answered[at][1], fields=shown.picked(held[at], wanted) if wanted else None))
        for at in order
    ]


def collections_of(home: Path, name: str | None) -> tuple[Collection, ...]:
    return chosen_or_active(home, () if name is None else (name,))


def _searching[S](one: Collection, asked: Narrowing, hold: Hold[S]) -> Searching[S] | None:
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
            return Searching(one, transaction, hold(transaction, allowed), resources)
        except BaseException:
            resources.close()
            raise


def opened[S](cleanup: ExitStack, asked: Narrowing, hold: Hold[S]) -> Kept[S]:
    kept = Kept[S](ruled_out=asked.ruled_out)
    for one in asked.collections:
        try:
            each = _searching(one, asked, hold)
        except StoreUnavailableError as error:
            kept.said(one, error)
            continue
        if each is None:
            kept.ruled_out += 1
            continue
        cleanup.callback(each.resources.close)
        kept.searching.append(each)
    return kept


@dataclass(frozen=True, slots=True)
class Seen:
    chunk: int
    hit: Hit
    work: str | None


class Rounded(Protocol):
    @property
    def found(self) -> Sequence[Scored]: ...

    @property
    def cut(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class Round:
    found: Sequence[Scored]
    cut: bool = False


def _read[S](
    one: Searching[S], chunks: Sequence[ChunkId], scores: Mapping[int, float], request: Options
) -> dict[int, Seen]:
    held = passages(one.transaction, chunks)
    documents = sorted({passage.document for passage in held.values()})
    whole = {} if request.compact else texts(one.transaction, documents)
    fields = fields_by_document(one.transaction, documents)
    shown = fielded(request)
    return {
        chunk: Seen(
            chunk,
            Hit(
                f"{one.collection.name}/{record_path(passage.path, passage.key)}",
                passage.title,
                f"{passage.first_line}-{passage.last_line}",
                scores[chunk],
                None if request.compact else whole[passage.document][passage.start : passage.start + passage.length],
                fields=fields.get(passage.document) if shown else None,
            ),
            work_of(fields.get(passage.document, {}).get("doi")),
        )
        for chunk, passage in held.items()
    }


def _copied(best: Seen, copy: Seen) -> Seen:
    return replace(best, hit=replace(best.hit, also=(*best.hit.also, copy.hit.id)))


def _work(each: Seen) -> str | None:
    return each.work


def hits[S](
    kept: Kept[S], found: Sequence[Scored], request: Options, copy_of: str | None = None
) -> list[tuple[int, Hit]]:
    scores = {each.chunk: each.score for each in found}
    read_all: dict[int, Seen] = {}
    for one in list(kept.searching):
        mine = [ChunkId(each.chunk) for each in found if owner(each.chunk) == one.collection.id]
        read_all |= kept.guarded(one, partial(_read, one, mine, scores, request)) if mine else {}
    seen = [read_all[each.chunk] for each in found if copy_of is None or read_all[each.chunk].work != copy_of]
    works = seen if request.all_passages else one_per_work(seen, _work, _copied)
    return [(each.chunk, each.hit) for each in works]


def always() -> bool:
    return True


def answered[S](
    kept: Kept[S],
    rank: Callable[[int], Rounded],
    request: Options,
    copy_of: str | None = None,
    more: Callable[[], bool] = always,
) -> tuple[list[tuple[int, Hit]], bool]:
    wanted = request.k
    while True:
        found = rank(wanted)
        shown = hits(kept, found.found, request, copy_of)
        if len(shown) >= request.k or len(found.found) < wanted or found.cut:
            return presented(shown[: request.k], request), found.cut
        if not more():
            return presented(shown, request), True
        wanted *= WIDER


def result[S](kept: Kept[S], answered: Sequence[tuple[int, Hit]], *, partial: bool) -> SearchResult:
    despite_losses(kept.closed)
    live = {one.collection.id for one in kept.searching}
    found = tuple(hit for chunk, hit in answered if owner(chunk) in live)
    return SearchResult(found, len(kept.searching) + kept.ruled_out, tuple(kept.unsearched), partial=partial)


def also_unsearched(found: SearchResult, more: Sequence[Unsearched]) -> SearchResult:
    said = {each.collection for each in found.unsearched}
    return replace(found, unsearched=(*found.unsearched, *(each for each in more if each.collection not in said)))
