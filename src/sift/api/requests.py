import dataclasses
import math
import re
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from types import MappingProxyType
from typing import Annotated, Literal, TypeAliasType, cast, get_args, get_origin, get_type_hints

from sift.api.described import Bounds
from sift.api.errors import InvalidRequestError

SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")
HALF_PAIR_REMEDY = "send it as text in UTF-8"
KIND_NAMES: Mapping[object, str] = MappingProxyType(
    {str: "text", int: "a whole number", float: "a finite number", bool: "true or false"}
)


def _parts(hint: object) -> tuple[object, tuple[object, ...]]:
    if get_origin(hint) is not Annotated:
        return hint, ()
    kind, *metadata = cast("tuple[object, ...]", get_args(hint))
    return kind, tuple(metadata)


def _members(kind: object) -> tuple[object, ...]:
    return cast("tuple[object, ...]", get_args(kind)) if get_origin(kind) is types.UnionType else (kind,)


def _each(kind: object) -> object | None:
    found = cast("tuple[object, ...]", get_args(kind))
    return found[0] if get_origin(kind) is tuple and found[1:] == (Ellipsis,) else None


def _nothing(value: object) -> bool:
    return value is None


def _truth(value: object) -> bool:
    return isinstance(value, bool)


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


SCALARS: Mapping[object, Callable[[object], bool]] = MappingProxyType(
    {types.NoneType: _nothing, bool: _truth, int: _whole, float: _number}
)


def _one_fits(value: object, kind: object) -> bool:
    if isinstance(kind, TypeAliasType):
        return fits(value, cast("object", kind.__value__))
    scalar = SCALARS.get(kind)
    return scalar(value) if scalar is not None else _shaped_fits(value, kind)


def _shaped_fits(value: object, kind: object) -> bool:
    origin = get_origin(kind)
    if origin is Literal:
        return value in cast("tuple[object, ...]", get_args(kind))
    if origin is Mapping:
        return isinstance(value, Mapping)
    each = _each(kind)
    if each is not None:
        return isinstance(value, tuple) and all(fits(item, each) for item in cast("tuple[object, ...]", value))
    if isinstance(kind, type):
        return isinstance(value, kind)
    raise TypeError(f"a request holds no {kind!r}")


def fits(value: object, kind: object) -> bool:
    return any(_one_fits(value, member) for member in _members(kind))


def choices_of(kind: object) -> type[StrEnum] | None:
    return kind if isinstance(kind, type) and issubclass(kind, StrEnum) else None


def _one_named(kind: object) -> str:
    if get_origin(kind) is Literal:
        return " or ".join(str(each) for each in cast("tuple[object, ...]", get_args(kind)))
    each = _each(kind)
    if each is not None:
        return f"a list of {_plural(each)}"
    choices = choices_of(kind)
    if choices is not None:
        return ", ".join(member.value for member in choices)
    if kind is types.NoneType:
        return "nothing"
    return KIND_NAMES.get(kind, type_name(kind))


def type_name(kind: object) -> str:
    return kind.__name__ if isinstance(kind, type) else repr(kind)


def _plural(kind: object) -> str:
    return "text" if kind is str else f"{type_name(kind).lower()}s"


def named(kind: object) -> str:
    found = [_one_named(member) for member in _members(kind)]
    return found[0] if len(found) == 1 else f"{', '.join(found[:-1])} or {found[-1]}"


@cache
def _hints(kind: type) -> tuple[tuple[str, object, tuple[Bounds, ...]], ...]:
    hints = cast("dict[str, object]", get_type_hints(kind, include_extras=True))
    found: list[tuple[str, object, tuple[Bounds, ...]]] = []
    for field in dataclasses.fields(kind):
        hint, metadata = _parts(hints[field.name])
        found.append((field.name, hint, tuple(part for part in metadata if isinstance(part, Bounds))))
    return tuple(found)


def _bounded(name: str, value: object, bounds: Bounds) -> None:
    if isinstance(value, tuple):
        held = len(cast("tuple[object, ...]", value))
        if not bounds.low <= held <= bounds.high:
            said = f"{name} holds {held:,}, not {bounds.low:,} to {bounds.high:,}"
            raise InvalidRequestError(said, f"give {bounds.low:,} to {bounds.high:,} of them", name)
    elif isinstance(value, int) and not bounds.low <= value <= bounds.high:
        said = f"{value:,} is outside {bounds.low:,} to {bounds.high:,}"
        raise InvalidRequestError(said, f"give {bounds.low:,} to {bounds.high:,}", name)


def holds_half_a_pair(value: object) -> bool:
    if isinstance(value, str):
        return SURROGATE.search(value) is not None
    return isinstance(value, tuple) and any(holds_half_a_pair(each) for each in cast("tuple[object, ...]", value))


def checked(request: object) -> None:
    for name, kind, bounds in _hints(type(request)):
        value = cast("object", getattr(request, name))
        if not fits(value, kind):
            wanted = named(kind)
            raise InvalidRequestError(f"{name} is {value!r}, not {wanted}", f"give {name} as {wanted}", name)
        if holds_half_a_pair(value):
            raise InvalidRequestError(f"{name} holds half a surrogate pair", HALF_PAIR_REMEDY, name)
        for each in bounds:
            _bounded(name, value, each)


@dataclass(frozen=True, slots=True)
class Request:
    def __post_init__(self) -> None:
        checked(self)
