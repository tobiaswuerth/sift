import re

from sift.api.errors import InvalidRequestError
from sift.domain.text.wording import counted

RANGE = re.compile(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?")
RANGE_REMEDY = "lines are a line number or a range such as 17-26"


def line_range(written: str) -> tuple[int, int]:
    matched = RANGE.fullmatch(written)
    first, last = (int(matched[1]), int(matched[2] or matched[1])) if matched else (0, 0)
    if not matched or last < first:
        raise InvalidRequestError(f"{written!r}: {RANGE_REMEDY}", "give a line number, or two joined by a dash")
    return first, last


def refuse_past_end(name: str, total: int, *, first: int) -> None:
    if first > total:
        said = f"{name} has {counted(total, 'line')}; line {first} is past its end"
        raise InvalidRequestError(said, f"ask for lines 1 to {total} at most")
