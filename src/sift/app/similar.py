from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from sift.api.errors import InvalidRequestError
from sift.api.ids import DocumentId
from sift.api.links import FindSimilar
from sift.api.search import SearchResult
from sift.app import hits, searching
from sift.app.linked import LeftOut
from sift.app.notes import Note, note_of, read_in
from sift.app.searching import Wants
from sift.app.vector_candidates import RESCORING, Rescoring, alike, nearest_hits
from sift.domain.codec.codes import Floats
from sift.domain.rank.works import work_of_fields
from sift.storage.database import Reading
from sift.storage.documents import chunks_of, fields_by_document
from sift.storage.vectors import model_of, vectors_of

EMBED_REMEDY = "run sift embed"
NO_DIRECTION_REMEDY = "start from another document, or search its words with sift vsearch"


@dataclass(frozen=True, slots=True)
class Like:
    query: Floats
    model: str
    work: str | None


def _like(note: Note, whole: str, transaction: Reading) -> Like:
    document = note.find(transaction)
    chunks = np.array([each.id for each in chunks_of(transaction, DocumentId(document))], np.int64)
    _, vectors = vectors_of(transaction, chunks)
    model = model_of(transaction)
    if not len(vectors) or model is None:
        raise InvalidRequestError(f"{whole!r} has no vectors yet", EMBED_REMEDY)
    mean = vectors.astype(np.float32).mean(axis=0)
    length = float(np.linalg.norm(mean))
    if not np.isfinite(length) or length == 0:
        raise InvalidRequestError(
            f"{whole!r} has no direction to be near to: its passages cancel out", NO_DIRECTION_REMEDY
        )
    work = work_of_fields(fields_by_document(transaction, [document]).get(document, {}))
    return Like(mean / length, model, work)


def similar(home: Path, request: FindSimilar, asked: Rescoring = RESCORING) -> SearchResult:
    searching.checked(request)
    named = note_of(home, request.id)
    like = read_in(named.note.collection, partial(_like, named.note, request.id))
    narrowing = searching.narrowed(home, request, LeftOut(named.note, linking=False))
    with ExitStack() as cleanup:
        stores = searching.opened(cleanup, narrowing, Wants(model=like.model, vectors_required=True))
        alike(stores, like.model, like.query)
        found = hits.result(stores, nearest_hits(stores, request, like.query, asked, like.work), partial=False)
    return hits.also_unsearched(found, named.unsearched)
