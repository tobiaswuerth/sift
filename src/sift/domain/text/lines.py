from sift.api.errors import InvalidRequestError
from sift.domain.text.wording import counted


def refuse_past_end(name: str, total: int, *, first: int) -> None:
    if first > total:
        said = f"{name} has {counted(total, 'line')}; line {first} is past its end"
        raise InvalidRequestError(said, f"ask for lines 1 to {total} at most")
