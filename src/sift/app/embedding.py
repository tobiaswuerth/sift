import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from types import MappingProxyType

import numpy as np

from sift.api.collections import Audience
from sift.api.embedding import CollectionEmbedded, EmbedCollections, Embedded, EmbedResult
from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.ids import CollectionId
from sift.api.indexing import Skipped
from sift.app.collections import chosen, chosen_or_active
from sift.app.loading import compiled, prepared
from sift.domain.codec.codes import Halves
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
FEWEST_COMPILED = 20_000


@dataclass(frozen=True, slots=True)
class Embedding:
    chunks_a_call: int = 1024
    chunks_a_read: int = 16_384
    compile_from: int = FEWEST_COMPILED


EMBEDDING = Embedding()


type Failed = dict[CollectionId, StoreUnavailableError]


@dataclass(frozen=True, slots=True)
class Work:
    report: Reporter
    models: Models
    embedding: Embedding
    ready: Callable[[], None]
    compiled: Callable[[], None]


def _waiting(connection: Connection, after: int, most: int) -> list[Pending]:
    with reading(connection) as transaction:
        return pending(transaction, after, most)


def _length(waiting: Pending) -> int:
    return len(waiting.text)


def _encoded(models: Models, waiting: Sequence[Pending]) -> Halves:
    return models.encode([embedded(each.title, each.text, first=each.first) for each in waiting], query=False)


def _written(connection: Connection, models: Models, waiting: Sequence[Pending], matrix: Halves) -> VectorsPut:
    chunks = np.array([each.chunk for each in waiting], np.int64)
    order = np.argsort(chunks, kind="stable")
    with writing(connection) as transaction:
        return put_vectors(transaction, chunks[order], matrix[order], models.name)


def _filled(connection: Connection, work: Work) -> Embedded:
    report = work.report
    with reading(connection) as transaction:
        refuse_model(transaction, work.models.name)
        total = left(transaction)
    if not total:
        return Embedded(0, 0)
    work.ready()
    if total >= work.embedding.compile_from:
        work.compiled()
    report(PhaseStarted(Phase.EMBED, total))
    stored, broken, after, done = 0, 0, 0, 0
    while waiting := _waiting(connection, after, work.embedding.chunks_a_read):
        for alike in itertools.batched(sorted(waiting, key=_length, reverse=True), work.embedding.chunks_a_call):
            kept = _written(connection, work.models, alike, _encoded(work.models, alike))
            stored, broken, done = stored + kept.stored, broken + len(kept.broken), done + len(alike)
            report(Advanced(done))
        after = waiting[-1].chunk
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


def models_held(home: Path, names: Sequence[str]) -> Mapping[str, str]:
    held = ((one.name, _model_held(one)) for one in chosen(home, names, Audience.PEOPLE))
    return MappingProxyType({name: model for name, model in held if model is not None})


def embed(
    home: Path, request: EmbedCollections, report: Reporter, models: Models, embedding: Embedding = EMBEDDING
) -> EmbedResult:
    if request.reset and request.names:
        said = "embeds every collection again, as all share one model"
        raise InvalidRequestError(said, RESET_ALL_REMEDY, "reset")
    chosen_now = chosen_or_active(home, request.names, Audience.PEOPLE)
    ready, compiling = partial(prepared, report, (models.prepare,)), partial(compiled, report, models.prepare_many)
    work = Work(report, models, embedding, cache(ready), cache(compiling))
    failed = _reset(home, report, work.ready, chosen_now) if request.reset else {}
    return EmbedResult(tuple(_embedded(one, work, failed) for one in chosen_now))
