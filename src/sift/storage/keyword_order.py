from collections.abc import Callable, Iterator
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids, among, ascending, ints
from sift.domain.codec.codes import Bits
from sift.domain.codec.order import topical
from sift.storage.database import Connection, Count, IdRow, Reading, reading, rows, run, run_each, streamed, writing
from sift.storage.keyword_beside import begin
from sift.storage.keyword_meta import beside
from sift.storage.keywords import current, remake_above
from sift.storage.vectors import codes_of

ORDER_SHARE = 0.2
PLACED_AT_ONCE = 100_000

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
    return beside(transaction) is None and _due_since(held, held.ordered_at)


def _placed(first: int, sequence: Ids) -> Iterator[tuple[int, int]]:
    for at in range(0, len(sequence), PLACED_AT_ONCE):
        yield from enumerate(ints(sequence[at : at + PLACED_AT_ONCE]), first + at)


def placed_anew(connection: Connection, order: Ids, made_at: int) -> int:
    with writing(connection) as transaction:
        held = _held(transaction)
        if _due_since(held, made_at) or not order_due(transaction):
            return 0
        every = streamed(transaction, IdRow, "SELECT id FROM chunks ORDER BY id")
        ids = np.fromiter((row.id for row in every), np.int64)
        coded = order[among(order, ascending(ids))]
        sequence = np.concatenate([coded, ids[~among(ids, ascending(coded))]])
        (top,) = rows(transaction, Count, "SELECT places AS n FROM counts")
        if current(transaction):
            begin(transaction, top.n, _placed(top.n + 1, sequence), made_at)
        else:
            run_each(transaction, "UPDATE chunks SET place = ? WHERE id = ?", _placed(top.n + 1, sequence))
            remake_above(transaction, top.n)
            run(transaction, "UPDATE counts SET ordered_at = ?", (made_at,))
        run(transaction, "UPDATE counts SET places = ?", (top.n + len(sequence),))
        return len(sequence)


def ordered(connection: Connection, order: Order = topical) -> int:
    with reading(connection) as transaction:
        made_at = _held(transaction).written
        ids, bits = codes_of(transaction)
    return placed_anew(connection, order(ids, bits), made_at)
