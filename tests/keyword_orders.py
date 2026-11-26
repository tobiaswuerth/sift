from typing import NamedTuple

import numpy as np

from sift.storage.database import BATCHES, Batches, Connection, Count, IdRow, reading, rows, writing
from sift.storage.keyword_beside import built
from sift.storage.keyword_order import order_due, placed_anew
from sift.storage.keywords import current
from sift.storage.vectors import put_vectors

WIDTH = 64


class Placed(NamedTuple):
    id: int
    place: int


def placed(connection: Connection) -> list[Placed]:
    with reading(connection) as transaction:
        return rows(transaction, Placed, "SELECT id, place FROM chunks ORDER BY place")


def embedded(connection: Connection, chunks: list[int]) -> None:
    matrix = np.random.default_rng(len(chunks)).standard_normal((len(chunks), WIDTH)).astype(np.float16)
    with writing(connection) as transaction:
        put_vectors(transaction, np.array(chunks, np.int64), matrix, "fake")


def embedded_pending(connection: Connection) -> None:
    with reading(connection) as transaction:
        pending = rows(transaction, IdRow, "SELECT chunk AS id FROM unembedded ORDER BY chunk")
    embedded(connection, [each.id for each in pending])


def written(connection: Connection) -> int:
    with reading(connection) as transaction:
        (found,) = rows(transaction, Count, "SELECT vectors_written AS n FROM counts")
    return found.n


def placed_in(connection: Connection, order: list[int]) -> int:
    embedded(connection, [each.id for each in placed(connection)])
    return placed_anew(connection, np.array(order, np.int64), written(connection))


def finished(connection: Connection, batches: Batches = BATCHES) -> int:
    return max(built(connection, batches), default=0)


def current_of(connection: Connection) -> bool:
    with reading(connection) as transaction:
        return current(transaction)


def due(connection: Connection) -> bool:
    with reading(connection) as transaction:
        return order_due(transaction)
