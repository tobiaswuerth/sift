from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from sift.api.errors import ModelUnavailableError, Problem, StoreUnavailableError
from sift.api.ids import ChunkId, owner
from sift.api.notices import Notice
from sift.api.search import Options
from sift.app import hits
from sift.app.searching import Open, Ranked, Searching, despite_losses
from sift.domain.arrays import Ids, Mask, ints
from sift.domain.codec.codes import Floats, Halves
from sift.domain.rank.best import Best, offered, ranked
from sift.domain.text.wording import counted
from sift.inference.models import Models
from sift.retrieval.allowed import AllowedChunks
from sift.retrieval.vector.nearest import RESCORE, Near, Reads, nearest
from sift.storage.database import Reading
from sift.storage.documents import documents_of
from sift.storage.passing import Passing
from sift.storage.vectors import EMBED_REMEDY, documents_embedded, vectors_of

QUERY_REMEDY = "word the query otherwise; if the model keeps failing, report it with its name"
WIDENED_REMEDY = "narrow the search with a path or filters, or ask for every passage"
WIDER = 4
MOST_WIDER = WIDER**2

type ExactRows = tuple[Ids, Halves]
type ExactReader = Callable[[Reading, Ids], ExactRows]


@dataclass(frozen=True, slots=True)
class Rescoring:
    rescore: int
    read: ExactReader
    count: Callable[[Reading], int] = documents_embedded


RESCORING = Rescoring(RESCORE, vectors_of)


def query_vector(models: Models, text: str) -> Floats:
    vector = models.encode([text], query=True).astype(np.float32).reshape(-1)
    if not bool(np.isfinite(vector).all()):
        raise ModelUnavailableError("the model gave no vector for the query", QUERY_REMEDY)
    if not bool(vector.any()):
        raise ModelUnavailableError("the model gave the query no direction to be near to", QUERY_REMEDY)
    return vector


def _unlike(stores: Open, model: str, query: Floats) -> list[tuple[Searching, StoreUnavailableError]]:
    return [
        (
            one,
            StoreUnavailableError(
                f"its vectors have {vectors.coded.width:,} dimensions, the query {len(query):,}",
                EMBED_REMEDY.format(model=model),
                code=Problem.OTHER_MODEL,
            ),
        )
        for one, vectors in stores.embedded()
        if len(vectors.coded.ids) and vectors.coded.width != len(query)
    ]


def alike(stores: Open, model: str, query: Floats) -> None:
    for one, refused in _unlike(stores, model, query):
        stores.dropped(one, refused)


def alike_or_by_keywords(stores: Open, model: str, query: Floats) -> None:
    for one, unlike in _unlike(stores, model, query):
        stores.by_keywords_alone(one, unlike)


@dataclass(frozen=True, slots=True)
class Guarded:
    stores: Open
    each: Searching
    allowed: AllowedChunks

    def checked(self, ids: Ids) -> Mask:
        return self.stores.guarded(self.each, partial(self.allowed.checked, ids))

    def listed(self) -> Ids:
        return self.stores.guarded(self.each, self.allowed.listed)


def _guarded(stores: Open, each: Searching, allowed: Passing | None) -> Guarded | None:
    return None if allowed is None else Guarded(stores, each, allowed)


def _nearest(stores: Open, query: Floats, want: int, read: ExactReader) -> tuple[Near, ...]:
    held = stores.embedded()

    def exact(store: int, chunks: Ids) -> ExactRows:
        one = held[store][0]
        return stores.guarded(one, partial(read, one.transaction, chunks))

    allowed = [_guarded(stores, one, vectors.allowed) for one, vectors in held]
    return nearest([vectors.coded for _, vectors in held], query, want, Reads(exact, allowed), rescore=want)


def _owned(stores: Open, near: tuple[Near, ...]) -> dict[int, int]:
    owners: dict[int, int] = {}
    for one in list(stores.searching):
        mine = np.array([each.chunk for each in near if owner(each.chunk) == one.collection.id], np.int64)
        found = stores.guarded(one, partial(documents_of, one.transaction, mine)) if len(mine) else mine
        owners |= {chunk: document for chunk, document in zip(ints(mine), ints(found), strict=True) if document >= 0}
    return owners


def _best(wanted: Best, near: tuple[Near, ...], owners: dict[int, int]) -> tuple[Near, ...]:
    chunks = np.array([each.chunk for each in near], np.int64)
    documents = np.array([owners[each.chunk] for each in near], np.int64)
    scores = np.array([each.score for each in near], np.float64)
    return tuple(Near(ChunkId(chunk), score) for chunk, score in ranked(offered(wanted, chunks, documents, scores)))


def _documents(stores: Open, count: Callable[[Reading], int]) -> int:
    return sum(stores.guarded(one, partial(count, one.transaction)) for one, _ in stores.embedded())


@dataclass(frozen=True, slots=True)
class Nearest:
    found: tuple[Near, ...]
    cut: bool = False


def _capped(stores: Open, read: int, best: tuple[Near, ...]) -> Nearest:
    said = f"the {read:,} nearest passages belong to {counted(len(best), 'document')}: a search reads no farther"
    stores.left_out.append(Notice(Problem.WIDENED_ENOUGH, said, WIDENED_REMEDY))
    return Nearest(best, cut=True)


def near(stores: Open, wanted: Best, query: Floats, asked: Rescoring) -> Nearest:
    chunks = sum(len(vectors.coded.ids) for _, vectors in stores.embedded())
    want, documents = max(asked.rescore, wanted.k), None
    farthest = want * MOST_WIDER
    while True:
        found = _nearest(stores, query, want, asked.read)
        best = _best(wanted, found, _owned(stores, found))
        if len(best) >= wanted.k or want >= chunks or len(found) < want:
            return Nearest(best)
        documents = _documents(stores, asked.count) if documents is None else documents
        if len(best) >= documents:
            return Nearest(best)
        if want >= farthest:
            return _capped(stores, want, best)
        if stores.expired():
            return Nearest(best, cut=True)
        want = min(want * WIDER, farthest)


def _round(stores: Open, request: Options, query: Floats, asked: Rescoring, wanted: int) -> Ranked:
    found = near(stores, Best(wanted, not request.all_passages), query, asked)
    return Ranked(found.found, found.cut)


def nearest_hits(
    stores: Open, request: Options, query: Floats, asked: Rescoring, copy_of: str | None = None
) -> hits.Answered:
    rank = partial(_round, stores, request, query, asked)
    return despite_losses(partial(hits.answered, stores, rank, request, copy_of))
