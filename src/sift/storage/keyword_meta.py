import dataclasses
import hashlib
import json

from sift.domain.codec.shape import FORMAT, Shape
from sift.domain.text.tokens import TABLE_DIGEST
from sift.storage.database import Reading, Value, rows

SHAPE_KEY, THROUGH_KEY, FINGERPRINT_KEY = "keyword_shape", "keywords_through", "keyword_fingerprint"
NUMBERS_PROBE = """SELECT coalesce(group_concat(j.atom, ' '), '') FROM json_tree(
                       '[1e20, 0.1, 1.0, -0.0, 1.5e-7, 3.141592653589793, 12345678901234567890, true, false]') j
                   WHERE j.atom IS NOT NULL"""


def fingerprint(transaction: Reading) -> str:
    (rendered,) = rows(transaction, Value, NUMBERS_PROBE)
    return hashlib.blake2b(f"{FORMAT}|{TABLE_DIGEST}|{rendered.value}".encode(), digest_size=16).hexdigest()


def keyword_meta(transaction: Reading, shape: Shape, start: int) -> tuple[tuple[str, str], ...]:
    return (
        (SHAPE_KEY, json.dumps(dataclasses.asdict(shape))),
        (THROUGH_KEY, str(start)),
        (FINGERPRINT_KEY, fingerprint(transaction)),
    )
