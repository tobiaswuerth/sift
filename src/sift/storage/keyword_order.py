from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids, among, ascending, ints
from sift.domain.codec.codes import Bits
from sift.domain.codec.order import topical
from sift.storage.database import Connection, Count, IdRow, Reading, reading, rows, run, run_each, writing
from sift.storage.keywords import remake_above
from sift.storage.vectors import codes_of

ORDER_SHARE = 0.2

type Order = Callable[[Ids, Bits], Ids]


class _Held(NamedTuple):
    chunks: int
    vectors: int
    written: int
    ordered_at: int


def _held(transaction: Reading) -> _Held:
    sql = "SELECT chunks, vectors, vectors_written AS written, ordered_at FROM counts"
    (found,) = rows(transaction, _Held, sql)
    return found


def _due_since(held: _Held, written_at: int) -> bool:
    return held.vectors > 0 and held.written - written_at > ORDER_SHARE * held.chunks


def order_due(transaction: Reading) -> bool:
    held = _held(transaction)
    return _due_since(held, held.ordered_at)


def placed_anew(connection: Connection, order: Ids, made_at: int) -> int:
    with writing(connection) as transaction:
        held = _held(transaction)
        if held.ordered_at >= made_at or _due_since(held, made_at):
            return 0
        ids = np.fromiter((row.id for row in rows(transaction, IdRow, "SELECT id FROM chunks ORDER BY id")), np.int64)
        coded = order[among(order, ascending(ids))]
        sequence = np.concatenate([coded, ids[~among(ids, ascending(coded))]])
        (top,) = rows(transaction, Count, "SELECT places AS n FROM counts")
        places = range(top.n + 1, top.n + 1 + len(sequence))
        run_each(transaction, "UPDATE chunks SET place = ? WHERE id = ?", zip(places, ints(sequence), strict=True))
        remake_above(transaction, top.n)
        run(transaction, "UPDATE counts SET places = ?, ordered_at = ?", (top.n + len(sequence), made_at))
        return len(sequence)


def ordered(connection: Connection, order: Order = topical) -> int:
    with reading(connection) as transaction:
        made_at = _held(transaction).written
        ids, bits = codes_of(transaction)
    return placed_anew(connection, order(ids, bits), made_at)
