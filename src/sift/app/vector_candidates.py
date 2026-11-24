from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.ids import ChunkId, owner
from sift.api.search import Hit, Options
from sift.app import hits
from sift.app.searching import Open, Ranked, Searching, despite_losses
from sift.domain.arrays import Ids, Mask
from sift.domain.codec.codes import Floats, Halves
from sift.domain.rank.best import Best, offered, ranked
from sift.inference.models import Models
from sift.retrieval.allowed import AllowedChunks
from sift.retrieval.vector.nearest import RESCORE, Near, Reads, nearest
from sift.storage.database import Reading
from sift.storage.documents import documents_of
from sift.storage.passing import Passing
from sift.storage.vectors import EMBED_REMEDY, documents_embedded, vectors_of

QUERY_REMEDY = "word the query otherwise; if the model keeps failing, report it with its name"
WIDER = 4

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
        raise InvalidRequestError("the model gave no vector for the query", QUERY_REMEDY)
    if not bool(vector.any()):
        raise InvalidRequestError("the model gave the query no direction to be near to", QUERY_REMEDY)
    return vector


def alike(stores: Open, model: str, query: Floats) -> None:
    for one, vectors in stores.embedded():
        width = vectors.coded.width
        if len(vectors.coded.ids) and width != len(query):
            said = f"its vectors have {width:,} dimensions, the query {len(query):,}"
            refused = StoreUnavailableError(said, EMBED_REMEDY.format(model=model), code=Problem.OTHER_MODEL)
            stores.dropped(one, refused)


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
        mine = [each.chunk for each in near if owner(each.chunk) == one.collection.id]
        owners |= stores.guarded(one, partial(documents_of, one.transaction, mine)) if mine else {}
    return owners


def _best(wanted: Best, near: tuple[Near, ...], owners: dict[int, int]) -> tuple[Near, ...]:
    chunks = np.array([each.chunk for each in near], np.int64)
    documents = np.array([owners[each.chunk] for each in near], np.int64)
    scores = np.array([each.score for each in near], np.float64)
    return tuple(Near(ChunkId(chunk), score) for chunk, score in ranked(offered(wanted, chunks, documents, scores)))


def _documents(stores: Open, count: Callable[[Reading], int]) -> int:
    return sum(stores.guarded(one, partial(count, one.transaction)) for one, _ in stores.embedded())


def near(stores: Open, wanted: Best, query: Floats, asked: Rescoring) -> tuple[Near, ...]:
    chunks = sum(len(vectors.coded.ids) for _, vectors in stores.embedded())
    want, documents = max(asked.rescore, wanted.k), None
    while True:
        found = _nearest(stores, query, want, asked.read)
        best = _best(wanted, found, _owned(stores, found))
        if len(best) >= wanted.k or want >= chunks or len(found) < want:
            return best
        documents = _documents(stores, asked.count) if documents is None else documents
        if len(best) >= documents:
            return best
        want *= WIDER


def _round(stores: Open, request: Options, query: Floats, asked: Rescoring, wanted: int) -> Ranked:
    return Ranked(near(stores, Best(wanted, not request.all_passages), query, asked))


def nearest_hits(
    stores: Open, request: Options, query: Floats, asked: Rescoring, copy_of: str | None = None
) -> list[tuple[int, Hit]]:
    rank = partial(_round, stores, request, query, asked)
    return despite_losses(partial(hits.answered, stores, rank, request, copy_of))[0]
