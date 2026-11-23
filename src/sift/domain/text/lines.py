import re

from sift.api.errors import InvalidRequestError

RANGE = re.compile(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?")
LINE = re.compile(r"[^\n]*\n|[^\n]+")
RANGE_REMEDY = "lines are a line number or a range such as 17-26"


def line_range(written: str) -> tuple[int, int]:
    matched = RANGE.fullmatch(written)
    first, last = (int(matched[1]), int(matched[2] or matched[1])) if matched else (0, 0)
    if not matched or last < first:
        raise InvalidRequestError(f"{written!r}: {RANGE_REMEDY}", "give a line number, or two joined by a dash")
    return first, last


def lines_of(text: str, first: int, last: int, name: str) -> str:
    every = [found.group() for found in LINE.finditer(text)]
    if first > len(every):
        noun = "line" if len(every) == 1 else "lines"
        said = f"{name} has {len(every)} {noun}; line {first} is past its end"
        raise InvalidRequestError(said, f"ask for lines 1 to {len(every)} at most")
    return "".join(every[first - 1 : last])
