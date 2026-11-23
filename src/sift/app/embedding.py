from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sift.api.embedding import CollectionEmbedded, EmbedCollections, Embedded, EmbedResult
from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.ids import CollectionId
from sift.api.indexing import Skipped
from sift.app.collections import chosen, chosen_or_active
from sift.domain.text.embedding import embedded
from sift.inference.models import Models
from sift.storage.catalog import Collection
from sift.storage.database import Connection, reading, writing
from sift.storage.store import open_store, unavailable_if_damaged
from sift.storage.vectors import Kept, Pending, forget_vectors, left, pending, put_vectors, refuse_model

RESET_ALL_REMEDY = "run sift embed --reset without naming collections"
BROKEN_REMEDY = "run sift embed again; if they stay, the model fails on their text"


@dataclass(frozen=True, slots=True)
class Embedding:
    chunks_a_write: int = 1024


EMBEDDING = Embedding()


type Failed = dict[CollectionId, StoreUnavailableError]


def _prepared(models: Models, report: Reporter) -> None:
    report(PhaseStarted(Phase.LOAD, None))
    notices = models.prepare()
    report(PhaseDone(1))
    for notice in notices:
        report(Noted(notice))


def _counted(n: int, what: str) -> str:
    return f"{n:,} {what}{'s' * (n != 1)}"


def _waiting(connection: Connection, after: int, most: int) -> list[Pending]:
    with reading(connection) as transaction:
        return pending(transaction, after, most)


def _written(connection: Connection, models: Models, waiting: list[Pending]) -> Kept:
    matrix = models.encode([embedded(each.title, each.text, first=each.first) for each in waiting], query=False)
    with writing(connection) as transaction:
        return put_vectors(transaction, np.array([each.chunk for each in waiting], np.int64), matrix, models.name)


def _filled(connection: Connection, report: Reporter, models: Models, embedding: Embedding) -> Embedded:
    with reading(connection) as transaction:
        refuse_model(transaction, models.name)
        total = left(transaction)
    if not total:
        return Embedded(0, 0)
    _prepared(models, report)
    report(PhaseStarted(Phase.EMBED, total))
    stored, broken, after = 0, 0, 0
    while waiting := _waiting(connection, after, embedding.chunks_a_write):
        kept = _written(connection, models, waiting)
        stored, broken, after = stored + kept.stored, broken + len(kept.broken), waiting[-1].chunk
        report(Advanced(stored + broken))
    report(PhaseDone(stored))
    if broken:
        report(Noted(f"{_counted(broken, 'chunk')} could not be embedded: the model gave no vector", BROKEN_REMEDY))
    return Embedded(stored, broken)


def _skipped(one: Collection, report: Reporter, error: StoreUnavailableError) -> CollectionEmbedded:
    report(Noted(error.message, error.remedy))
    return CollectionEmbedded(one.name, Skipped(error.message, error.remedy))


def _embedded(
    one: Collection, report: Reporter, models: Models, embedding: Embedding, failed: Failed
) -> CollectionEmbedded:
    report(CollectionStarted(one.name, str(one.folder)))
    if one.id in failed:
        return _skipped(one, report, failed[one.id])
    try:
        with unavailable_if_damaged(one.store), closing(open_store(one.store, one.id)) as connection:
            return CollectionEmbedded(one.name, _filled(connection, report, models, embedding))
    except StoreUnavailableError as error:
        return _skipped(one, report, error)


def _forgotten(one: Collection) -> StoreUnavailableError | None:
    try:
        with (
            unavailable_if_damaged(one.store),
            closing(open_store(one.store, one.id)) as connection,
            writing(connection) as transaction,
        ):
            forget_vectors(transaction)
    except StoreUnavailableError as error:
        return StoreUnavailableError(f"its vectors could not be forgotten: {error.message}", error.remedy)
    return None


def _reset(home: Path, report: Reporter, models: Models, embedding_now: tuple[Collection, ...]) -> Failed:
    _prepared(models, report)
    known = chosen(home, ())
    failed = {one.id: error for one in known if (error := _forgotten(one)) is not None}
    now = {one.id for one in embedding_now}
    for one in known:
        if one.id in failed and one.id not in now:
            report(CollectionStarted(one.name, str(one.folder)))
            report(Noted(failed[one.id].message, failed[one.id].remedy))
    return failed


def embed(
    home: Path, request: EmbedCollections, report: Reporter, models: Models, embedding: Embedding = EMBEDDING
) -> EmbedResult:
    if request.reset and request.names:
        raise InvalidRequestError("--reset embeds every collection again, as all share one model", RESET_ALL_REMEDY)
    chosen_now = chosen_or_active(home, request.names)
    failed = _reset(home, report, models, chosen_now) if request.reset else {}
    return EmbedResult(tuple(_embedded(one, report, models, embedding, failed) for one in chosen_now))
