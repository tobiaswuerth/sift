from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.ids import owner
from sift.api.search import Hit, SearchResult, SearchVectors
from sift.app import searching
from sift.app.linked import Narrowing
from sift.domain.arrays import Ids, Mask
from sift.domain.codec.codes import Floats, Halves
from sift.domain.rank.best import Best, offered, ranked
from sift.inference.models import Models
from sift.retrieval.allowed import AllowedChunks
from sift.retrieval.vector.nearest import RESCORE, Coded, Near, Reads, coded, nearest
from sift.storage.catalog import Collection
from sift.storage.database import Reading
from sift.storage.documents import documents_of
from sift.storage.passing import Passing
from sift.storage.vectors import EMBED_REMEDY, codes_of, documents_embedded, left, model_of, vectors_of, width_held

EMPTY_REMEDY = "say what to find, e.g. sift vsearch 'how dark matter halos form'"
UNEMBEDDED_REMEDY = "run sift embed"
QUERY_REMEDY = "search again, or word the query otherwise"
WIDER = 4

type Exact = tuple[Ids, Halves]
type Read = Callable[[Reading, Ids], Exact]


@dataclass(frozen=True, slots=True)
class Held:
    coded: Coded
    allowed: Passing | None = None


type Kept = searching.Kept[Held]


@dataclass(frozen=True, slots=True)
class Asked:
    rescore: int
    read: Read
    count: Callable[[Reading], int] = documents_embedded


ASKED = Asked(RESCORE, vectors_of)


def no_vectors() -> Held:
    return Held(coded(np.zeros(0, np.int64), np.zeros((0, 0), np.uint8), 0))


def embedded_by(model: str, transaction: Reading) -> Held | None:
    found = model_of(transaction)
    if found is None:
        return None
    if found != model:
        raise StoreUnavailableError(f"its vectors are of {found}, not of {model}", EMBED_REMEDY.format(model=model))
    ids, bits = codes_of(transaction)
    return Held(coded(ids, bits, width_held(transaction) or 0))


def held(model: str, transaction: Reading) -> Held:
    found = embedded_by(model, transaction)
    if found is not None:
        return found
    if left(transaction):
        raise StoreUnavailableError("it holds no vectors yet", UNEMBEDDED_REMEDY)
    return no_vectors()


def held_within(model: str, asked: Narrowing, one: Collection, transaction: Reading) -> Held:
    return replace(held(model, transaction), allowed=searching.allowed(asked, one, transaction))


def query_vector(models: Models, text: str) -> Floats:
    vector = models.encode([text], query=True).astype(np.float32).reshape(-1)
    if not bool(np.isfinite(vector).all()):
        raise InvalidRequestError("the model gave no vector for the query", QUERY_REMEDY)
    return vector


def alike[S](kept: searching.Kept[S], held_of: Callable[[S], Held], model: str, query: Floats) -> None:
    for one in list(kept.searching):
        width = held_of(one.held).coded.width
        if len(held_of(one.held).coded.ids) and width != len(query):
            said = f"its vectors have {width:,} dimensions, the query {len(query):,}"
            kept.dropped(one, StoreUnavailableError(said, EMBED_REMEDY.format(model=model)))


def _nearest[S](
    kept: searching.Kept[S], held_of: Callable[[S], Held], query: Floats, want: int, read: Read
) -> tuple[Near, ...] | None:
    held = list(kept.searching)
    lost: list[bool] = []

    def exact(store: int, chunks: Ids) -> Exact:
        each = held[store]
        found = None if lost else kept.guarded(each, partial(read, each.transaction, chunks))
        if found is None:
            lost.append(True)
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float16)
        return found

    allowed = [_guarded(kept, each, held_of(each.held).allowed, lost) for each in held]
    found = nearest([held_of(each.held).coded for each in held], query, want, Reads(exact, allowed), rescore=want)
    return None if lost else found


@dataclass(frozen=True, slots=True)
class Guarded[S]:
    kept: searching.Kept[S]
    each: searching.Searching[S]
    allowed: AllowedChunks
    lost: list[bool]

    def checked(self, ids: Ids) -> Mask:
        found = None if self.lost else self.kept.guarded(self.each, partial(self.allowed.checked, ids))
        if found is None:
            self.lost.append(True)
            return np.zeros(len(ids), np.bool_)
        return found

    def listed(self) -> Ids:
        found = None if self.lost else self.kept.guarded(self.each, self.allowed.listed)
        if found is None:
            self.lost.append(True)
            return np.zeros(0, np.int64)
        return found


def _guarded[S](
    kept: searching.Kept[S], each: searching.Searching[S], allowed: Passing | None, lost: list[bool]
) -> Guarded[S] | None:
    return None if allowed is None else Guarded(kept, each, allowed, lost)


def _owned[S](kept: searching.Kept[S], near: tuple[Near, ...]) -> dict[int, int] | None:
    owners: dict[int, int] = {}
    for one in list(kept.searching):
        mine = [each.chunk for each in near if owner(each.chunk) == one.collection.id]
        found = kept.guarded(one, partial(documents_of, one.transaction, mine)) if mine else {}
        if found is None:
            return None
        owners |= found
    return owners


def _best(wanted: Best, near: tuple[Near, ...], owners: dict[int, int]) -> tuple[Near, ...]:
    chunks = np.array([each.chunk for each in near], np.int64)
    documents = np.array([owners[each.chunk] for each in near], np.int64)
    scores = np.array([each.score for each in near], np.float64)
    return tuple(Near(chunk, score) for chunk, score in ranked(offered(wanted, chunks, documents, scores)))


def _documents[S](kept: searching.Kept[S], count: Callable[[Reading], int]) -> int | None:
    total = 0
    for one in list(kept.searching):
        found = kept.guarded(one, partial(count, one.transaction))
        if found is None:
            return None
        total += found
    return total


def near[S](
    kept: searching.Kept[S], held_of: Callable[[S], Held], wanted: Best, query: Floats, asked: Asked
) -> tuple[Near, ...] | None:
    chunks = sum(len(held_of(each.held).coded.ids) for each in kept.searching)
    want, documents = max(asked.rescore, wanted.k), None
    while True:
        found = _nearest(kept, held_of, query, want, asked.read)
        owners = None if found is None else _owned(kept, found)
        if found is None or owners is None:
            return None
        best = _best(wanted, found, owners)
        if len(best) >= wanted.k or want >= chunks or len(found) < want:
            return best
        documents = _documents(kept, asked.count) if documents is None else documents
        if documents is None:
            return None
        if len(best) >= documents:
            return best
        want *= WIDER


def _itself(held: Held) -> Held:
    return held


def _answered(kept: Kept, request: SearchVectors, query: Floats, asked: Asked) -> list[tuple[int, Hit]]:
    while True:
        found = near(kept, _itself, Best(request.k, not request.all_passages), query, asked)
        shown = None if found is None else searching.hits(kept, found, request)
        if shown is not None:
            return shown


def vsearch(home: Path, request: SearchVectors, models: Models, asked: Asked = ASKED) -> SearchResult:
    searching.checked(request)
    if not request.query.strip():
        raise InvalidRequestError("an empty query finds nothing", EMPTY_REMEDY)
    narrowing = searching.narrowed(home, request)
    with ExitStack() as cleanup:
        hold = partial(held_within, models.name, narrowing)
        kept = searching.opened(cleanup, narrowing.collections, hold, narrowing.ruled_out)
        if not any(len(each.held.coded.ids) for each in kept.searching):
            return searching.result(kept, (), partial=False)
        query = query_vector(models, request.query)
        alike(kept, _itself, models.name, query)
        return searching.result(kept, _answered(kept, request, query, asked), partial=False)
