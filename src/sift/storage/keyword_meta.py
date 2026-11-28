import dataclasses
import hashlib
import json
from enum import IntEnum
from typing import NamedTuple

from sift.domain.codec.shape import FORMAT, Shape
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS, TABLE_DIGEST
from sift.storage.database import Reading, Value, Writing, rows, run

SHAPE_KEY, THROUGH_KEY, FINGERPRINT_KEY = "keyword_shape", "keywords_through", "keyword_fingerprint"
FORMAT_KEY = "keyword_format"
NUMBERS_PROBE = """SELECT coalesce(group_concat(j.atom, ' '), '') FROM json_tree(
                       '[1e20, 0.1, 1.0, -0.0, 1.5e-7, 3.141592653589793, 12345678901234567890, true, false]') j
                   WHERE j.atom IS NOT NULL"""


def fingerprint(transaction: Reading) -> str:
    (rendered,) = rows(transaction, Value, NUMBERS_PROBE)
    held = f"{FORMAT}|{TABLE_DIGEST}|{MOST_FIELD_CHARACTERS}|{rendered.value}"
    return hashlib.blake2b(held.encode(), digest_size=16).hexdigest()


def keyword_meta(transaction: Reading, shape: Shape, start: int) -> tuple[tuple[str, str], ...]:
    return (
        (SHAPE_KEY, json.dumps(dataclasses.asdict(shape))),
        (THROUGH_KEY, str(start)),
        (FINGERPRINT_KEY, fingerprint(transaction)),
        (FORMAT_KEY, str(FORMAT)),
    )


class Side(IntEnum):
    SERVED = 0
    BUILT = 1


class Beside(NamedTuple):
    above: int
    through: int
    made_at: int


def beside(transaction: Reading) -> Beside | None:
    found = rows(transaction, Beside, "SELECT above, through, made_at FROM keyword_beside")
    return found[0] if found else None


def drop_beside(transaction: Writing) -> None:
    run(transaction, "DELETE FROM keyword_segments WHERE id IN (SELECT id FROM keyword_built)")
    run(transaction, "DELETE FROM keyword_places")
    run(transaction, "DELETE FROM keyword_beside")
