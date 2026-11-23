import dataclasses
from typing import Annotated, cast, get_args, get_origin, get_type_hints

from sift.api.described import Bounds
from sift.api.errors import InvalidRequestError


def _ranges(hint: object) -> tuple[Bounds, ...]:
    if get_origin(hint) is not Annotated:
        return ()
    return tuple(part for part in cast("tuple[object, ...]", get_args(hint))[1:] if isinstance(part, Bounds))


def validated(request: object) -> None:
    if not dataclasses.is_dataclass(request) or isinstance(request, type):
        raise TypeError(f"{request!r} is no request")
    hints = cast("dict[str, object]", get_type_hints(type(request), include_extras=True))
    for field in dataclasses.fields(request):
        value = cast("object", getattr(request, field.name))
        for bounds in _ranges(hints[field.name]):
            if isinstance(value, int) and not bounds.low <= value <= bounds.high:
                said = f"{value:,} is outside {bounds.low:,} to {bounds.high:,}"
                raise InvalidRequestError(said, f"give {bounds.low:,} to {bounds.high:,}", field.name)
