import dataclasses
import math
import re
import types
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from types import MappingProxyType
from typing import Annotated, Literal, TypeAliasType, cast, get_args, get_origin, get_type_hints

from sift.api.described import Bounds, Composed, Length, Within
from sift.api.errors import InvalidRequestError

SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")
HALF_PAIR_REMEDY = "send it as text in UTF-8"
HOLDS: Mapping[bool, str] = MappingProxyType({True: "holds a text", False: "is"})
EMPTY: Mapping[bool, str] = MappingProxyType({True: "holds an empty text", False: "is empty"})
KIND_NAMES: Mapping[object, str] = MappingProxyType(
    {str: "text", int: "a whole number", float: "a finite number", bool: "true or false"}
)
MOST_SHOWN = 100
WORD_START = re.compile(r"(?<=[a-z])(?=[A-Z])")
VOWELS = frozenset("aeiou")


def annotated_parts(hint: object) -> tuple[object, tuple[object, ...]]:
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


def plain_name(kind: type) -> str:
    return WORD_START.sub(" ", kind.__name__).lower()


def shown_value(value: object, most: int = MOST_SHOWN) -> str:
    text = f"{value:,}" if _whole(value) else repr(value)
    return text if len(text) <= most else f"{text[: most - 3]}..."


def _one_named(kind: object) -> str:
    if isinstance(kind, TypeAliasType):
        return named(cast("object", kind.__value__))
    if get_origin(kind) is Literal:
        return " or ".join(str(each) for each in cast("tuple[object, ...]", get_args(kind)))
    if get_origin(kind) is Mapping:
        return "an object"
    each = _each(kind)
    if each is not None:
        return f"a list of {_plural(each)}"
    choices = choices_of(kind)
    if choices is not None:
        return ", ".join(member.value for member in choices)
    return _simply_named(kind)


def _simply_named(kind: object) -> str:
    if kind is types.NoneType:
        return "nothing"
    if isinstance(kind, type) and kind not in KIND_NAMES:
        words = plain_name(kind)
        return f"{'an' if words[0] in VOWELS else 'a'} {words}"
    return KIND_NAMES.get(kind, type_name(kind))


def type_name(kind: object) -> str:
    return kind.__name__ if isinstance(kind, type) else repr(kind)


def _plural(kind: object) -> str:
    if kind is str:
        return "text"
    if isinstance(kind, type) and kind not in KIND_NAMES:
        return f"{plain_name(kind)}s"
    return f"values, each {named(kind)}"


def named(kind: object) -> str:
    found = [_one_named(member) for member in _members(kind)]
    return found[0] if len(found) == 1 else f"{', '.join(found[:-1])} or {found[-1]}"


@cache
def _hints(kind: type) -> tuple[tuple[str, object, tuple[object, ...]], ...]:
    hints = cast("dict[str, object]", get_type_hints(kind, include_extras=True))
    return tuple((field.name, *annotated_parts(hints[field.name])) for field in dataclasses.fields(kind))


def _bounded(name: str, value: object, bounds: Bounds) -> None:
    if isinstance(value, tuple):
        held = len(cast("tuple[object, ...]", value))
        if not bounds.low <= held <= bounds.high:
            said = f"{name} holds {held:,}, not {bounds.low:,} to {bounds.high:,}"
            raise InvalidRequestError(said, f"give {bounds.low:,} to {bounds.high:,} of them", name)
    elif isinstance(value, int) and not bounds.low <= value <= bounds.high:
        span = f"{bounds.shown(bounds.low)} to {bounds.shown(bounds.high)}"
        written = shown_value(value) if bounds.grouped else str(value)
        raise InvalidRequestError(f"{written} is outside {span}", f"give {span}", name)


def _sized(name: str, value: object, length: Length) -> None:
    texts = cast("tuple[object, ...]", value) if isinstance(value, tuple) else (value,)
    for text in (each for each in texts if isinstance(each, str)):
        if not length.low <= len(text) <= length.high:
            listed = isinstance(value, tuple)
            long = f"{name} {HOLDS[listed]} {len(text):,} characters long, not {length.low:,} to {length.high:,}"
            said = long if text else f"{name} {EMPTY[listed]}"
            raise InvalidRequestError(said, f"give {name} as {length.low:,} to {length.high:,} characters", name)


def _spanned(name: str, value: object, span: Within) -> None:
    if isinstance(value, int | float) and not span.low <= value <= span.high:
        said = f"{shown_value(value)} is outside {span.low:,g} to {span.high:,g}"
        raise InvalidRequestError(said, f"give {span.low:,g} to {span.high:,g}", name)


def _held_to(name: str, value: object, mark: object) -> None:
    if isinstance(mark, Bounds):
        _bounded(name, value, mark)
    elif isinstance(mark, Length):
        _sized(name, value, mark)
    elif isinstance(mark, Within):
        _spanned(name, value, mark)


def holds_half_a_pair(value: object) -> bool:
    if isinstance(value, str):
        return SURROGATE.search(value) is not None
    return isinstance(value, tuple) and any(holds_half_a_pair(each) for each in cast("tuple[object, ...]", value))


@cache
def _composed_names(kind: type) -> tuple[str, ...]:
    return tuple(name for name, _, metadata in _hints(kind) if any(isinstance(each, Composed) for each in metadata))


def composed(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, tuple):
        return tuple(composed(each) for each in cast("tuple[object, ...]", value))
    return value


def checked(request: object) -> None:
    for name, kind, metadata in _hints(type(request)):
        value = cast("object", getattr(request, name))
        if not fits(value, kind):
            wanted = named(kind)
            raise InvalidRequestError(f"{name} is {shown_value(value)}, not {wanted}", f"give {name} as {wanted}", name)
        if holds_half_a_pair(value):
            raise InvalidRequestError(f"{name} holds half a surrogate pair", HALF_PAIR_REMEDY, name)
        for each in metadata:
            _held_to(name, value, each)


@dataclass(frozen=True, slots=True)
class Request:
    def __post_init__(self) -> None:
        checked(self)
        for name in _composed_names(type(self)):
            object.__setattr__(self, name, composed(cast("object", getattr(self, name))))
