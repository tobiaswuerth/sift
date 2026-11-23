from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sift.api.embedding import CollectionEmbedded, EmbedCollections, Embedded, EmbedResult
from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.indexing import Skipped
from sift.app.collections import chosen, chosen_or_active
from sift.domain.text.embedding import embedded
from sift.inference.models import Models
from sift.storage.catalog import Collection
from sift.storage.database import Connection, reading, writing
from sift.storage.store import open_store, unavailable_if_damaged
from sift.storage.vectors import Pending, forget_vectors, left, pending, put_vectors, refuse_model

RESET_ALL_REMEDY = "run sift embed --reset without naming collections"
BROKEN_REMEDY = "run sift embed again; if they stay, the model fails on their text"


@dataclass(frozen=True, slots=True)
class Embedding:
    chunks_a_write: int = 1024


EMBEDDING = Embedding()


@dataclass(slots=True)
class Tally:
    stored: int = 0
    broken: int = 0


def _counted(n: int, what: str) -> str:
    return f"{n:,} {what}{'s' * (n != 1)}"


def _waiting(connection: Connection, after: int, most: int) -> list[Pending]:
    with reading(connection) as transaction:
        return pending(transaction, after, most)


def _written(connection: Connection, models: Models, waiting: list[Pending], tally: Tally) -> None:
    matrix = models.encode([embedded(each.title, each.text, first=each.first) for each in waiting], query=False)
    with writing(connection) as transaction:
        kept = put_vectors(transaction, np.array([each.chunk for each in waiting], np.int64), matrix, models.name)
    tally.stored, tally.broken = tally.stored + kept.stored, tally.broken + len(kept.broken)


def _filled(connection: Connection, report: Reporter, models: Models, embedding: Embedding) -> Embedded:
    with reading(connection) as transaction:
        refuse_model(transaction, models.name)
        total = left(transaction)
    if not total:
        return Embedded(0, 0)
    report(PhaseStarted(Phase.EMBED, total))
    tally, after = Tally(), 0
    try:
        while waiting := _waiting(connection, after, embedding.chunks_a_write):
            _written(connection, models, waiting, tally)
            after = waiting[-1].chunk
            report(Advanced(tally.stored + tally.broken))
    finally:
        report(PhaseDone(tally.stored))
    if tally.broken:
        report(
            Noted(f"{_counted(tally.broken, 'chunk')} could not be embedded: the model gave no vector", BROKEN_REMEDY)
        )
    return Embedded(tally.stored, tally.broken)


def _embedded(one: Collection, report: Reporter, models: Models, embedding: Embedding) -> CollectionEmbedded:
    report(CollectionStarted(one.name, str(one.folder)))
    try:
        with unavailable_if_damaged(one.store), closing(open_store(one.store, one.id)) as connection:
            return CollectionEmbedded(one.name, _filled(connection, report, models, embedding))
    except StoreUnavailableError as error:
        report(Noted(error.message, error.remedy))
        return CollectionEmbedded(one.name, Skipped(error.message, error.remedy))


def _forgotten(one: Collection) -> StoreUnavailableError | None:
    try:
        with (
            unavailable_if_damaged(one.store),
            closing(open_store(one.store, one.id)) as connection,
            writing(connection) as transaction,
        ):
            forget_vectors(transaction)
    except StoreUnavailableError as error:
        return error
    return None


def _reset(home: Path, report: Reporter, models: Models, embedding_now: tuple[Collection, ...]) -> None:
    models.prepare()
    for one in chosen(home, ()):
        error = _forgotten(one)
        if error is not None and one not in embedding_now:
            report(CollectionStarted(one.name, str(one.folder)))
            report(Noted(f"its vectors could not be forgotten: {error.message}", error.remedy))


def embed(
    home: Path, request: EmbedCollections, report: Reporter, models: Models, embedding: Embedding = EMBEDDING
) -> EmbedResult:
    if request.reset and request.names:
        raise InvalidRequestError("--reset embeds every collection again, as all share one model", RESET_ALL_REMEDY)
    chosen_now = chosen_or_active(home, request.names)
    if request.reset:
        _reset(home, report, models, chosen_now)
    return EmbedResult(tuple(_embedded(one, report, models, embedding) for one in chosen_now))
