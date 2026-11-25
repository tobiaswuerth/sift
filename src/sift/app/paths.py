from pathlib import Path

from sift.api.errors import InvalidRequestError

ABSOLUTE_REMEDY = "give an absolute path"


def absolute(text: str, field: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        raise InvalidRequestError(f"{field} {text!r} is no absolute path", ABSOLUTE_REMEDY, field)
    return path
