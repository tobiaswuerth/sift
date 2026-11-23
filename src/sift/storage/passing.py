import sqlite3
from contextlib import closing
from typing import NamedTuple, cast

import numpy as np

from sift.domain.arrays import Ids
from sift.domain.query.filters import Condition
from sift.storage.database import Reading, streamed

QUOTED_NAME = ('{"a\\"b": 1}', '$."a\\"b"')
PASSING = """
SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document JOIN sources s ON s.id = d.source
WHERE (:path IS NULL OR s.path GLOB :path)
  AND NOT EXISTS (
    SELECT 1 FROM json_each(:fields) g
    WHERE NOT (
      (json_extract(g.value, '$.any') IS NULL AND json_array_length(g.value, '$.ranges') = 0
       OR EXISTS (
         SELECT 1 FROM json_each(d.fields, json_extract(g.value, '$.path')) v
         WHERE (json_extract(g.value, '$.any') IS NULL
                OR v.value IN (SELECT value FROM json_each(g.value, '$.any')))
           AND NOT EXISTS (
             SELECT 1 FROM json_each(g.value, '$.ranges') r
             WHERE NOT (
               CASE WHEN json_extract(r.value, '$.text')
               THEN v.type IN ('text', 'integer', 'real') AND CASE json_extract(r.value, '$.op')
                 WHEN '<' THEN substr(v.value, 1, json_extract(r.value, '$.len')) < json_extract(r.value, '$.value')
                 WHEN '<=' THEN substr(v.value, 1, json_extract(r.value, '$.len')) <= json_extract(r.value, '$.value')
                 WHEN '>' THEN substr(v.value, 1, json_extract(r.value, '$.len')) > json_extract(r.value, '$.value')
                 ELSE substr(v.value, 1, json_extract(r.value, '$.len')) >= json_extract(r.value, '$.value') END
               ELSE (v.type IN ('integer', 'real', 'true', 'false')
                     OR (v.type = 'text' AND (ltrim(v.value) GLOB '[0-9]*' OR ltrim(v.value) GLOB '[-+.][0-9]*'
                                              OR ltrim(v.value) GLOB '[-+].[0-9]*')))
                 AND CASE json_extract(r.value, '$.op')
                   WHEN '<' THEN CAST(v.value AS REAL) < json_extract(r.value, '$.value')
                   WHEN '<=' THEN CAST(v.value AS REAL) <= json_extract(r.value, '$.value')
                   WHEN '>' THEN CAST(v.value AS REAL) > json_extract(r.value, '$.value')
                   ELSE CAST(v.value AS REAL) >= json_extract(r.value, '$.value') END
               END)))
      )
      AND (json_extract(g.value, '$.none') IS NULL
           OR NOT EXISTS (SELECT 1 FROM json_each(d.fields, json_extract(g.value, '$.path')) v
                          WHERE v.value IN (SELECT value FROM json_each(g.value, '$.none'))))
    )
  )
ORDER BY c.id
"""


class Chunk(NamedTuple):
    id: int


def reads_quotes() -> bool:
    with closing(sqlite3.connect(":memory:")) as database:
        (found,) = cast("tuple[object]", database.execute("SELECT json_extract(?, ?)", QUOTED_NAME).fetchone())
    return found == 1


def chunks_passing(transaction: Reading, condition: Condition) -> Ids:
    asked = {"path": condition.path, "fields": condition.fields}
    return np.fromiter((row.id for row in streamed(transaction, Chunk, PASSING, asked)), np.int64)
