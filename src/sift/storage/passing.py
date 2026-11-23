import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from functools import cache
from typing import cast

import numpy as np

from sift.domain.arrays import Ascending, Ids, Mask, among, ints
from sift.domain.query.filters import Condition
from sift.storage.database import IdRow, Reading, rows, streamed
from sift.storage.documents import documents_of

QUOTED_NAME = ('{"a\\"b": 1}', '$."a\\"b"')
CHUNKS_OF = "SELECT id FROM chunks WHERE document IN (SELECT value FROM json_each(?)) ORDER BY id"
PASSING = """
WITH asked(id) AS (
  SELECT value FROM json_each(:documents)
  WHERE :within IS NULL OR value IN (SELECT value FROM json_each(:within))
  UNION ALL SELECT value FROM json_each(:within) WHERE :documents IS NULL
  UNION ALL SELECT id FROM documents WHERE :documents IS NULL AND :within IS NULL),
groups AS MATERIALIZED (
  SELECT g.key AS grp, json_extract(g.value, '$.path') AS field, json_extract(g.value, '$.any') AS any_of,
         json_extract(g.value, '$.none') AS none_of, json_array_length(g.value, '$.ranges') AS ranged
  FROM json_each(:fields) g),
linked AS MATERIALIZED (
  SELECT g.key AS grp, 'any' AS kind, CAST(l.value AS INTEGER) AS id
  FROM json_each(:fields) g, json_each(g.value, '$.linked_any') l
  UNION ALL
  SELECT g.key, 'none', CAST(l.value AS INTEGER) FROM json_each(:fields) g, json_each(g.value, '$.linked_none') l),
bounds AS MATERIALIZED (
  SELECT g.key AS grp, json_extract(r.value, '$.op') AS op, json_extract(r.value, '$.text') AS text,
         json_extract(r.value, '$.len') AS len, json_extract(r.value, '$.value') AS value
  FROM json_each(:fields) g, json_each(g.value, '$.ranges') r)
SELECT d.id FROM asked a JOIN documents d ON d.id = a.id JOIN sources s ON s.id = d.source
WHERE (:path IS NULL OR s.path GLOB :path)
  AND d.id NOT IN (SELECT value FROM json_each(:without))
  AND NOT EXISTS (
    SELECT 1 FROM groups g
    WHERE NOT (
      (g.any_of IS NULL AND g.ranged = 0
       OR EXISTS (
         SELECT 1 FROM json_each(d.fields, g.field) v
         WHERE (g.any_of IS NULL OR v.value IN (SELECT value FROM json_each(g.any_of)))
           AND NOT EXISTS (
             SELECT 1 FROM bounds b
             WHERE b.grp = g.grp AND NOT (
               CASE WHEN b.text
               THEN v.type IN ('text', 'integer', 'real') AND CASE b.op
                 WHEN '<' THEN substr(v.value, 1, b.len) < b.value
                 WHEN '<=' THEN substr(v.value, 1, b.len) <= b.value
                 WHEN '>' THEN substr(v.value, 1, b.len) > b.value
                 ELSE substr(v.value, 1, b.len) >= b.value END
               ELSE (v.type IN ('integer', 'real', 'true', 'false')
                     OR (v.type = 'text' AND (ltrim(v.value) GLOB '[0-9]*' OR ltrim(v.value) GLOB '[-+.][0-9]*'
                                              OR ltrim(v.value) GLOB '[-+].[0-9]*')))
                 AND CASE b.op
                   WHEN '<' THEN CAST(v.value AS REAL) < b.value
                   WHEN '<=' THEN CAST(v.value AS REAL) <= b.value
                   WHEN '>' THEN CAST(v.value AS REAL) > b.value
                   ELSE CAST(v.value AS REAL) >= b.value END
               END)))
       OR EXISTS (SELECT 1 FROM linked l WHERE l.grp = g.grp AND l.kind = 'any' AND l.id = d.id)
      )
      AND NOT EXISTS (SELECT 1 FROM linked l WHERE l.grp = g.grp AND l.kind = 'none' AND l.id = d.id)
      AND (g.none_of IS NULL
           OR NOT EXISTS (SELECT 1 FROM json_each(d.fields, g.field) v
                          WHERE v.value IN (SELECT value FROM json_each(g.none_of))))
    )
  )
"""


def _none() -> Ids:
    return np.zeros(0, np.int64)


def _no_mask() -> Mask:
    return np.zeros(0, np.bool_)


@cache
def reads_quotes() -> bool:
    with closing(sqlite3.connect(":memory:")) as database:
        (found,) = cast("tuple[object]", database.execute("SELECT json_extract(?, ?)", QUOTED_NAME).fetchone())
    return found == 1


def documents_passing(transaction: Reading, condition: Condition, documents: Ids | None) -> Ids:
    asked = None if documents is None else json.dumps(ints(documents))
    parameters = {
        "documents": asked,
        "path": condition.path,
        "fields": condition.fields,
        "within": condition.within,
        "without": condition.without,
    }
    return np.fromiter((row.id for row in streamed(transaction, IdRow, PASSING, parameters)), np.int64)


@dataclass(slots=True)
class Passing:
    transaction: Reading
    condition: Condition
    seen: Ids = field(default_factory=_none)
    passed: Mask = field(default_factory=_no_mask)
    every: Ids | None = None

    def documents(self, documents: Ids) -> Mask:
        asked = np.unique(documents)
        new = asked[~among(asked, Ascending(self.seen))]
        if len(new):
            held = np.concatenate([self.seen, new])
            order = np.argsort(held, kind="stable")
            let = np.isin(new, documents_passing(self.transaction, self.condition, new))
            self.seen, self.passed = held[order], np.concatenate([self.passed, let])[order]
        return self.passed[np.searchsorted(self.seen, documents)] if len(documents) else _no_mask()

    def checked(self, ids: Ids) -> Mask:
        owners = documents_of(self.transaction, ints(ids))
        documents = np.fromiter((owners.get(chunk, -1) for chunk in ints(ids)), np.int64, len(ids))
        known = documents >= 0
        let = np.zeros(len(ids), np.bool_)
        let[known] = self.documents(documents[known])
        return let

    def listed(self) -> Ids:
        if self.every is None:
            documents = documents_passing(self.transaction, self.condition, None)
            chunks = rows(self.transaction, IdRow, CHUNKS_OF, (json.dumps(ints(documents)),))
            self.every = np.array([row.id for row in chunks], np.int64)
        return self.every
