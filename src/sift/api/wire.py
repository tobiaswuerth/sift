import dataclasses
import types
from collections.abc import Mapping
from enum import Enum
from functools import cache
from types import MappingProxyType
from typing import Annotated, Literal, TypeAliasType, cast, get_args, get_origin, get_type_hints

from sift.api.errors import (
    InternalError,
    InvalidRequestError,
    JobStoppedError,
    ModelUnavailableError,
    NotFoundError,
    Problem,
    SiftError,
    StoreUnavailableError,
    SystemFailureError,
)
from sift.api.fields import Fields, FieldValue
from sift.api.requests import fits, named, type_name

type Json = str | int | float | bool | list[Json] | dict[str, Json] | None

WIRE_REMEDY = "send the request as sift's API describes it"
ERROR_KINDS: Mapping[str, type[SiftError]] = MappingProxyType(
    {
        "invalid_request": InvalidRequestError,
        "not_found": NotFoundError,
        "store_unavailable": StoreUnavailableError,
        "model_unavailable": ModelUnavailableError,
        "os": SystemFailureError,
        "stopped": JobStoppedError,
        "internal": InternalError,
    }
)
KIND_OF: Mapping[type[SiftError], str] = MappingProxyType({kind: name for name, kind in ERROR_KINDS.items()})
FREE: frozenset[object] = frozenset({Fields, FieldValue})


def encoded(value: object) -> Json:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: encoded(cast("object", getattr(value, field.name))) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return cast("Json", value.value)
    if isinstance(value, tuple | list):
        return [encoded(each) for each in cast("tuple[object, ...]", value)]
    if isinstance(value, Mapping):
        return {str(key): encoded(each) for key, each in cast("Mapping[object, object]", value).items()}
    if value is None or isinstance(value, str | int | float):
        return value
    raise TypeError(f"{value!r} has no JSON")


class WireError(ValueError):
    pass


def _refused(field: str, problem: str) -> InvalidRequestError:
    return InvalidRequestError(problem, WIRE_REMEDY, field)


def _plain(kind: object) -> object:
    while True:
        if get_origin(kind) is Annotated:
            kind = cast("tuple[object, ...]", get_args(kind))[0]
        elif isinstance(kind, TypeAliasType) and kind not in FREE:
            kind = cast("object", kind.__value__)
        else:
            return kind


@cache
def _fields(kind: type) -> tuple[tuple[str, object, bool], ...]:
    hints = cast("dict[str, object]", get_type_hints(kind, include_extras=True))
    return tuple(
        (
            field.name,
            _plain(hints[field.name]),
            field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING,
        )
        for field in dataclasses.fields(kind)
    )


def _object(kind: type, data: object, outer: str | None) -> object:
    if not isinstance(data, dict):
        raise WireError(f"a {type_name(kind).lower()} is an object")
    given = cast("dict[object, object]", data)
    known = {name: (hint, needed) for name, hint, needed in _fields(kind)}
    for key in given:
        if key not in known:
            raise _refused(outer or str(key), f"{kind.__name__} takes no {key}")
    for name, (_, needed) in known.items():
        if needed and name not in given:
            raise _refused(outer or name, f"{kind.__name__} needs {name}")
    built = {name: _value(hint, given[name], outer or name) for name, (hint, _) in known.items() if name in given}
    return cast("object", kind(**built))


def _discriminated(members: tuple[object, ...], data: object) -> object | None:
    if not isinstance(data, dict):
        return None
    state = cast("dict[object, object]", data).get("state")
    for member in members:
        if isinstance(member, type) and dataclasses.is_dataclass(member):
            hint = {name: hint for name, hint, _ in _fields(member)}.get("state")
            if hint is not None and state in cast("tuple[object, ...]", get_args(hint)):
                return member
    return None


def _one(kind: object, data: object, field: str) -> object:
    if kind in FREE:
        return data
    if get_origin(kind) is Literal:
        if data not in cast("tuple[object, ...]", get_args(kind)):
            raise _refused(field, f"{field} is {data!r}, not {named(kind)}")
        return data
    if get_origin(kind) is tuple and isinstance(data, list):
        (each, _) = cast("tuple[object, object]", get_args(kind))
        return tuple(_value(each, item, field) for item in cast("list[object]", data))
    if isinstance(kind, type) and issubclass(kind, Enum):
        try:
            return kind(data)
        except ValueError:
            choices = ", ".join(str(member) for member in kind)
            raise _refused(field, f"{data!r} is none of {choices}") from None
    if isinstance(kind, type) and dataclasses.is_dataclass(kind):
        return _object(kind, data, field)
    return data


def _first_taking(members: tuple[object, ...], data: object, field: str, whole: object) -> object:
    for member in members:
        try:
            found = _one(member, data, field)
        except (WireError, InvalidRequestError):
            continue
        if fits(found, member):
            return found
    raise _refused(field, f"{field} is {data!r}, not {named(whole)}")


def _value(kind: object, data: object, field: str) -> object:
    plain = _plain(kind)
    if get_origin(plain) is types.UnionType:
        members = tuple(_plain(member) for member in cast("tuple[object, ...]", get_args(plain)))
        if data is None and types.NoneType in members:
            return None
        chosen = _discriminated(members, data)
        if chosen is not None:
            return _one(chosen, data, field)
        rest = [member for member in members if member is not types.NoneType]
        if len(rest) > 1:
            return _first_taking(tuple(rest), data, field, plain)
        plain = rest[0]
    try:
        found = _one(plain, data, field)
    except WireError as error:
        raise _refused(field, str(error)) from None
    if not fits(found, plain):
        raise _refused(field, f"{field} is {data!r}, not {named(plain)}")
    return found


def decoded[T](kind: type[T], data: object) -> T:
    try:
        if get_origin(kind) is types.UnionType:
            return cast("T", _value(kind, data, ""))
        return cast("T", _object(kind, data, None))
    except WireError:
        raise InvalidRequestError("a request is an object of its fields", WIRE_REMEDY) from None


def encoded_error(error: SiftError) -> Json:
    field = error.field if isinstance(error, InvalidRequestError) else None
    return {
        "kind": KIND_OF[type(error)],
        "code": error.code.value,
        "message": error.message,
        "remedy": error.remedy,
        "field": field,
    }


def decoded_error(data: Json) -> SiftError:
    if not isinstance(data, dict):
        raise TypeError(f"{data!r} is no error")
    kind = ERROR_KINDS[str(data["kind"])]
    message, remedy, code = str(data["message"]), str(data["remedy"]), Problem(data["code"])
    if issubclass(kind, InvalidRequestError):
        field = data.get("field")
        return kind(message, remedy, None if field is None else str(field), code=code)
    return kind(message, remedy, code=code)
