import json
from typing import cast

import numpy as np

from sift.domain.arrays import Ids, ints
from sift.storage.database import Parameters, Reading

OWNERS_A_READ = 65_536
NO_DOCUMENT = -1
CHUNK_OWNERS = "SELECT id, document FROM chunks WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id"


def columns(transaction: Reading, sql: str, parameters: Parameters, width: int) -> tuple[Ids, ...]:
    fetched = cast("list[tuple[int, ...]]", transaction.connection.execute(sql, parameters).fetchall())
    table = np.array(fetched, np.int64).reshape(len(fetched), width)
    return tuple(np.ascontiguousarray(table[:, at]) for at in range(width))


def documents_of(transaction: Reading, chunks: Ids, part: int = OWNERS_A_READ) -> Ids:
    found = np.full(len(chunks), NO_DOCUMENT, np.int64)
    for start in range(0, len(chunks), part):
        asked = chunks[start : start + part]
        held, documents = columns(transaction, CHUNK_OWNERS, (json.dumps(ints(asked)),), 2)
        at = np.minimum(np.searchsorted(held, asked), max(len(held) - 1, 0))
        known = np.equal(held[at], asked) if len(held) else np.zeros(len(asked), np.bool_)
        found[start : start + part][known] = documents[at[known]]
    return found
