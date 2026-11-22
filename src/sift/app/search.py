from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.ids import ChunkId, owner
from sift.api.search import MOST_HITS, Hit, SearchKeywords, SearchResult, Unsearched
from sift.app.collections import chosen
from sift.domain.query.expressions import NOTHING, Expression
from sift.domain.query.grammar import parsed, relaxed
from sift.domain.rank.best import Best
from sift.retrieval.keyword.lists import Opened, opened
from sift.retrieval.keyword.search import Found, Prepared, found_in, prepared, query_of, searched
from sift.storage.catalog import Collection
from sift.storage.database import abandon, reading
from sift.storage.documents import passages, texts
from sift.storage.store import open_store, unavailable_if_damaged

K_REMEDY = "ask for one result or more, e.g. -k 10"
DECIMALS = 4


@dataclass(frozen=True, slots=True)
class Searching:
    collection: Collection
    store: Opened
    resources: ExitStack


@dataclass(slots=True)
class Kept:
    searching: list[Searching] = field(default_factory=list[Searching])
    unsearched: list[Unsearched] = field(default_factory=list[Unsearched])

    def said(self, collection: Collection, error: StoreUnavailableError) -> None:
        self.unsearched.append(Unsearched(collection.name, error.message, error.remedy))

    def guarded[T](self, each: Searching, work: Callable[[], T]) -> T | None:
        try:
            with unavailable_if_damaged(each.collection.store):
                return work()
        except StoreUnavailableError as error:
            self.searching.remove(each)
            self.said(each.collection, error)
            abandon(each.store.transaction)
            each.resources.close()
            return None


def _collections(home: Path, name: str | None) -> tuple[Collection, ...]:
    return chosen(home, (name,)) if name is not None else tuple(one for one in chosen(home, ()) if one.active)


def _searching(one: Collection) -> Searching:
    resources = ExitStack()
    with unavailable_if_damaged(one.store):
        try:
            connection = resources.enter_context(closing(open_store(one.store, one.id)))
            return Searching(one, opened(resources.enter_context(reading(connection))), resources)
        except BaseException:
            resources.close()
            raise


def _opened(cleanup: ExitStack, collections: Sequence[Collection]) -> Kept:
    kept = Kept()
    for one in collections:
        try:
            each = _searching(one)
        except StoreUnavailableError as error:
            kept.said(one, error)
            continue
        cleanup.callback(each.resources.close)
        kept.searching.append(each)
    return kept


def _found(kept: Kept, expression: Expression, k: int, *, by_document: bool) -> tuple[Found, ...]:
    ready: list[tuple[Searching, Prepared]] = []
    for one in list(kept.searching):
        held = kept.guarded(one, partial(prepared, one.store, expression))
        if held is not None:
            ready.append((one, held))
    query = query_of(expression, [held for _, held in ready])
    if query is None:
        return ()
    best = Best(k, by_document)
    for one, held in ready:
        after = kept.guarded(one, partial(searched, held, query, best))
        best = best if after is None else after
    return found_in(best)


def _searched(kept: Kept, request: SearchKeywords, k: int) -> tuple[Found, ...]:
    expression = parsed(request.query)
    by_document = not request.all_passages
    found = () if expression == NOTHING else _found(kept, expression, k, by_document=by_document)
    looser = relaxed(request.query) if not found else None
    return found if looser is None else _found(kept, looser, k, by_document=by_document)


def _read(one: Searching, chunks: Sequence[ChunkId], scores: Mapping[int, float], *, compact: bool) -> dict[int, Hit]:
    held = passages(one.store.transaction, chunks)
    whole = {} if compact else texts(one.store.transaction, sorted({passage.document for passage in held.values()}))
    return {
        chunk: Hit(
            f"{one.collection.name}/{passage.path}",
            passage.title,
            f"{passage.first_line}-{passage.last_line}",
            round(scores[chunk], DECIMALS),
            None if compact else whole[passage.document][passage.start : passage.start + passage.length],
        )
        for chunk, passage in held.items()
    }


def _hits(kept: Kept, found: Sequence[Found], *, compact: bool) -> tuple[Hit, ...]:
    scores = {each.chunk: each.score for each in found}
    shown: dict[int, Hit] = {}
    for one in list(kept.searching):
        mine = [ChunkId(each.chunk) for each in found if owner(each.chunk) == one.collection.id]
        read = kept.guarded(one, partial(_read, one, mine, scores, compact=compact)) if mine else {}
        if one in kept.searching:
            kept.guarded(one, one.resources.close)
        if read is not None and one in kept.searching:
            shown |= read
    return tuple(shown[each.chunk] for each in found if each.chunk in shown)


def search(home: Path, request: SearchKeywords) -> SearchResult:
    if request.k < 1:
        raise InvalidRequestError(f"-k {request.k} asks for no result", K_REMEDY)
    collections = _collections(home, request.collection)
    with ExitStack() as cleanup:
        kept = _opened(cleanup, collections)
        found = _searched(kept, request, min(request.k, MOST_HITS))
        hits = _hits(kept, found, compact=request.compact)
    return SearchResult(hits, len(collections), tuple(kept.unsearched))
