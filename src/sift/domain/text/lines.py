from sift.api.errors import InvalidRequestError
from sift.domain.text.wording import counted


def refuse_past_end(name: str, total: int, *, first: int) -> None:
    if first > total:
        held = f"has {counted(total, 'line')}" if total else "holds no text"
        remedy = f"ask for lines 1 to {total} at most" if total else "ask for it without lines"
        raise InvalidRequestError(f"{name} {held}; line {first} is past its end", remedy)
