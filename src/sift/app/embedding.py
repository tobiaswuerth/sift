from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path

import numpy as np

from sift.api.collections import Audience
from sift.api.embedding import CollectionEmbedded, EmbedCollections, Embedded, EmbedResult
from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.ids import CollectionId
from sift.api.indexing import Skipped
from sift.app.collections import chosen, chosen_or_active
from sift.app.loading import prepared
from sift.domain.text.embedding import embedded
from sift.domain.text.wording import counted
from sift.inference.models import Models
from sift.storage.catalog import Collection
from sift.storage.database import Connection, reading, writing
from sift.storage.store import connected, read_store
from sift.storage.vector_meta import left, model_of
from sift.storage.vectors import Pending, VectorsPut, forget_vectors, pending, put_vectors, refuse_model
from sift.storage.writer import sole_writer

RESET_ALL_REMEDY = "embed again with reset, naming no collection"
BROKEN_REMEDY = "embed again; if they stay, the model fails on their text"


@dataclass(frozen=True, slots=True)
class Embedding:
    chunks_a_write: int = 1024


EMBEDDING = Embedding()


type Failed = dict[CollectionId, StoreUnavailableError]


@dataclass(frozen=True, slots=True)
class Work:
    report: Reporter
    models: Models
    embedding: Embedding
    ready: Callable[[], None]


def _waiting(connection: Connection, after: int, most: int) -> list[Pending]:
    with reading(connection) as transaction:
        return pending(transaction, after, most)


def _written(connection: Connection, models: Models, waiting: list[Pending]) -> VectorsPut:
    matrix = models.encode([embedded(each.title, each.text, first=each.first) for each in waiting], query=False)
    with writing(connection) as transaction:
        return put_vectors(transaction, np.array([each.chunk for each in waiting], np.int64), matrix, models.name)


def _filled(connection: Connection, work: Work) -> Embedded:
    report = work.report
    with reading(connection) as transaction:
        refuse_model(transaction, work.models.name)
        total = left(transaction)
    if not total:
        return Embedded(0, 0)
    work.ready()
    report(PhaseStarted(Phase.EMBED, total))
    stored, broken, after = 0, 0, 0
    while waiting := _waiting(connection, after, work.embedding.chunks_a_write):
        kept = _written(connection, work.models, waiting)
        stored, broken, after = stored + kept.stored, broken + len(kept.broken), waiting[-1].chunk
        report(Advanced(stored + broken))
    report(PhaseDone(stored))
    if broken:
        report(Noted(f"{counted(broken, 'chunk')} could not be embedded: the model gave no vector", BROKEN_REMEDY))
    return Embedded(stored, broken)


def _skipped(one: Collection, report: Reporter, error: StoreUnavailableError) -> CollectionEmbedded:
    report(Noted(error.message, error.remedy, error.code))
    return CollectionEmbedded(one.name, Skipped(error.code, error.message, error.remedy))


def _embedded(one: Collection, work: Work, failed: Failed) -> CollectionEmbedded:
    work.report(CollectionStarted(one.name, str(one.folder)))
    if one.id in failed:
        return _skipped(one, work.report, failed[one.id])
    try:
        with sole_writer(one.store, "an embedding"), connected(one.store, one.id) as connection:
            return CollectionEmbedded(one.name, _filled(connection, work))
    except StoreUnavailableError as error:
        return _skipped(one, work.report, error)


def _forgotten(one: Collection) -> StoreUnavailableError | None:
    try:
        with (
            sole_writer(one.store, "an embedding"),
            connected(one.store, one.id) as connection,
            writing(connection) as transaction,
        ):
            forget_vectors(transaction)
    except StoreUnavailableError as error:
        return StoreUnavailableError(f"its vectors could not be forgotten: {error.message}", error.remedy)
    return None


def _reset(home: Path, report: Reporter, ready: Callable[[], None], embedding_now: tuple[Collection, ...]) -> Failed:
    ready()
    known = chosen(home, (), Audience.PEOPLE)
    failed = {one.id: error for one in known if (error := _forgotten(one)) is not None}
    now = {one.id for one in embedding_now}
    for one in known:
        if one.id in failed and one.id not in now:
            report(CollectionStarted(one.name, str(one.folder)))
            report(Noted(failed[one.id].message, failed[one.id].remedy, failed[one.id].code))
    return failed


def _model_held(one: Collection) -> str | None:
    try:
        with read_store(one.store, one.id) as transaction:
            return model_of(transaction)
    except StoreUnavailableError:
        return None


def embedded_by(home: Path, names: Sequence[str], model: str) -> tuple[str, ...]:
    return tuple(one.name for one in chosen(home, names, Audience.PEOPLE) if _model_held(one) == model)


def embed(
    home: Path, request: EmbedCollections, report: Reporter, models: Models, embedding: Embedding = EMBEDDING
) -> EmbedResult:
    if request.reset and request.names:
        said = "embeds every collection again, as all share one model"
        raise InvalidRequestError(said, RESET_ALL_REMEDY, "reset")
    chosen_now = chosen_or_active(home, request.names, Audience.PEOPLE)
    work = Work(report, models, embedding, cache(partial(prepared, report, (models.prepare,))))
    failed = _reset(home, report, work.ready, chosen_now) if request.reset else {}
    return EmbedResult(tuple(_embedded(one, work, failed) for one in chosen_now))
