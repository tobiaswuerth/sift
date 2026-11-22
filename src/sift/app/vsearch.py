from contextlib import ExitStack
from functools import partial
from pathlib import Path

import numpy as np

from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.ids import owner
from sift.api.search import Hit, SearchResult, SearchVectors
from sift.app import searching
from sift.domain.arrays import Ids
from sift.domain.codec.codes import Floats, Halves
from sift.domain.rank.best import Best, offered, ranked
from sift.inference.models import Models
from sift.retrieval.vector.nearest import RESCORE, Coded, Near, coded, nearest
from sift.storage.database import Reading
from sift.storage.documents import documents_of
from sift.storage.vectors import EMBED_REMEDY, codes_of, model_of, vectors_of, width_held

EMPTY_REMEDY = "say what to find, e.g. sift vsearch 'how dark matter halos form'"
UNEMBEDDED_REMEDY = "run sift embed"
QUERY_REMEDY = "search again, or word the query otherwise"

type Kept = searching.Kept[Coded]
type Exact = tuple[Ids, Halves]


def _held(model: str, transaction: Reading) -> Coded:
    found = model_of(transaction)
    if found is None:
        raise StoreUnavailableError("it holds no vectors yet", UNEMBEDDED_REMEDY)
    if found != model:
        raise StoreUnavailableError(f"its vectors are of {found}, not of {model}", EMBED_REMEDY.format(model=model))
    ids, bits = codes_of(transaction)
    return coded(ids, bits, width_held(transaction) or 0)


def _query(models: Models, text: str) -> Floats:
    if not text.strip():
        raise InvalidRequestError("an empty query finds nothing", EMPTY_REMEDY)
    vector = models.encode([text], query=True).astype(np.float32).reshape(-1)
    if not bool(np.isfinite(vector).all()):
        raise InvalidRequestError("the model gave no vector for the query", QUERY_REMEDY)
    return vector


def _nearest(kept: Kept, query: Floats, want: int) -> tuple[Near, ...] | None:
    held = list(kept.searching)
    lost: list[bool] = []

    def exact(store: int, chunks: Ids) -> Exact:
        each = held[store]
        found = kept.guarded(each, partial(vectors_of, each.transaction, chunks))
        if found is None:
            lost.append(True)
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float16)
        return found

    found = nearest([each.held for each in held], query, want, exact)
    return None if lost else found


def _owned(kept: Kept, near: tuple[Near, ...]) -> dict[int, int] | None:
    owners: dict[int, int] = {}
    for one in list(kept.searching):
        mine = [each.chunk for each in near if owner(each.chunk) == one.collection.id]
        found = kept.guarded(one, partial(documents_of, one.transaction, mine)) if mine else {}
        if found is None:
            return None
        owners |= found
    return owners


def _best(request: SearchVectors, near: tuple[Near, ...], owners: dict[int, int]) -> tuple[Near, ...]:
    kept = [each for each in near if each.chunk in owners]
    chunks = np.array([each.chunk for each in kept], np.int64)
    documents = np.array([owners[each.chunk] for each in kept], np.int64)
    scores = np.array([each.score for each in kept], np.float64)
    best = offered(Best(request.k, not request.all_passages), chunks, documents, scores)
    return tuple(Near(chunk, score) for chunk, score in ranked(best))


def _answered(kept: Kept, request: SearchVectors, query: Floats) -> list[tuple[int, Hit]]:
    while True:
        near = _nearest(kept, query, max(RESCORE, request.k))
        owners = None if near is None else _owned(kept, near)
        if near is None or owners is None:
            continue
        found = searching.hits(kept, _best(request, near, owners), compact=request.compact)
        if found is not None:
            return found


def vsearch(home: Path, request: SearchVectors, models: Models) -> SearchResult:
    searching.checked_k(request.k)
    query = _query(models, request.query)
    collections = searching.collections_of(home, request.collection)
    with ExitStack() as cleanup:
        kept = searching.opened(cleanup, collections, partial(_held, models.name))
        return searching.result(kept, _answered(kept, request, query), partial=False)
