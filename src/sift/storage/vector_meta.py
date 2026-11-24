from sift.storage.database import Count, Reading, Value, first, rows

MODEL_KEY, WIDTH_KEY = "vector_model", "vector_width"
HELD_META = "SELECT value FROM meta WHERE key = ? AND (SELECT vectors FROM counts) > 0"


def _meta(transaction: Reading, key: str) -> str | None:
    found = first(transaction, Value, HELD_META, (key,))
    return None if found is None else found.value


def model_of(transaction: Reading) -> str | None:
    return _meta(transaction, MODEL_KEY)


def width_held(transaction: Reading) -> int | None:
    found = _meta(transaction, WIDTH_KEY)
    return None if found is None else int(found)


def left(transaction: Reading) -> int:
    (found,) = rows(transaction, Count, "SELECT chunks - vectors FROM counts")
    return found.n
