import dataclasses
import types
from collections.abc import Collection, Mapping
from enum import Enum
from functools import cache
from types import MappingProxyType
from typing import Annotated, Literal, TypeAliasType, cast, get_args, get_origin, get_type_hints

from sift.api.errors import (
    CODE_KINDS,
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
from sift.api.requests import fits, named, shown_value

type Json = str | int | float | bool | list[Json] | dict[str, Json] | None

WIRE_REMEDY = "send the request as sift's API describes it"
CONTRACT = (2, 0)
CONTRACT_HEADER = "Sift-Contract"
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
PROBLEMS: Mapping[str, Problem] = MappingProxyType({each.value: each for each in Problem})
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


def _refused(field: str, problem: str, remedy: str) -> InvalidRequestError:
    return InvalidRequestError(problem, remedy, field)


def _not_of(field: str, data: object, kind: object) -> InvalidRequestError:
    wanted = named(kind)
    return _refused(field, f"{field} is {shown_value(data)}, not {wanted}", f"give {field} as {wanted}")


def _path(outer: str | None, name: str) -> str:
    return name if outer is None else f"{outer}.{name}"


def _within(outer: str, field: str | None, own: Collection[str]) -> str | None:
    base = outer.rsplit(".", 1)[-1].split("[", 1)[0]
    if field is None or field == base or field.startswith(f"{base}."):
        return outer + ("" if field is None else field[len(base) :])
    return f"{outer}.{field}" if field in own else field


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


def _refuse_unknown(known: Mapping[str, object], given: Mapping[object, object], outer: str | None) -> None:
    whole = "the request" if outer is None else outer
    for key in given:
        if key not in known:
            said = shown_value(str(key))
            raise _refused(
                _path(outer, str(key)), f"{whole} takes no {said}", f"leave out {said}: it takes {', '.join(known)}"
            )


def _object(kind: type, data: object, outer: str | None, *, strict: bool) -> object:
    if not isinstance(data, dict):
        raise WireError
    given = cast("dict[object, object]", data)
    known = {name: (hint, needed) for name, hint, needed in _fields(kind)}
    if strict:
        _refuse_unknown(known, given, outer)
    for name, (_, needed) in known.items():
        if needed and name not in given:
            raise _refused(_path(outer, name), f"{outer or 'the request'} needs {name}", f"give {_path(outer, name)}")
    built = {
        name: _value(hint, given[name], _path(outer, name), strict=strict)
        for name, (hint, _) in known.items()
        if name in given
    }
    try:
        return cast("object", kind(**built))
    except InvalidRequestError as error:
        if outer is not None:
            error.field = _within(outer, error.field, known)
        raise


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


def _one(kind: object, data: object, field: str, *, strict: bool) -> object:
    if kind in FREE:
        return data
    if get_origin(kind) is Literal:
        if data not in cast("tuple[object, ...]", get_args(kind)):
            raise _not_of(field, data, kind)
        return data
    if get_origin(kind) is tuple and isinstance(data, list):
        (each, _) = cast("tuple[object, object]", get_args(kind))
        items = enumerate(cast("list[object]", data))
        return tuple(_value(each, item, f"{field}[{at}]", strict=strict) for at, item in items)
    if isinstance(kind, type) and issubclass(kind, Enum):
        try:
            return kind(data)
        except ValueError:
            choices = ", ".join(str(member) for member in kind)
            raise _refused(
                field, f"{shown_value(data)} is none of {choices}", f"give {field} as one of {choices}"
            ) from None
    if isinstance(kind, type) and dataclasses.is_dataclass(kind):
        return _object(kind, data, field, strict=strict)
    return data


def _first_taking(members: tuple[object, ...], data: object, field: str, whole: object, *, strict: bool) -> object:
    for member in members:
        try:
            found = _one(member, data, field, strict=strict)
        except (WireError, InvalidRequestError):
            continue
        if fits(found, member):
            return found
    raise _not_of(field, data, whole)


def _value(kind: object, data: object, field: str, *, strict: bool) -> object:
    plain = whole = _plain(kind)
    if get_origin(plain) is types.UnionType:
        members = tuple(_plain(member) for member in cast("tuple[object, ...]", get_args(plain)))
        if data is None and types.NoneType in members:
            return None
        chosen = _discriminated(members, data)
        if chosen is not None:
            return _one(chosen, data, field, strict=strict)
        rest = [member for member in members if member is not types.NoneType]
        if len(rest) > 1:
            return _first_taking(tuple(rest), data, field, plain, strict=strict)
        plain = rest[0]
    try:
        found = _one(plain, data, field, strict=strict)
    except WireError:
        raise _not_of(field, data, whole) from None
    if not fits(found, plain):
        raise _not_of(field, data, whole)
    return found


def _decoded[T](kind: type[T], data: object, *, strict: bool) -> T:
    try:
        if get_origin(kind) is types.UnionType:
            return cast("T", _value(kind, data, "", strict=strict))
        return cast("T", _object(kind, data, None, strict=strict))
    except WireError:
        raise InvalidRequestError("a request is an object of its fields", WIRE_REMEDY) from None


def understood(contract: tuple[int, ...]) -> bool:
    return len(contract) == len(CONTRACT) and contract[0] == CONTRACT[0] and contract[1] >= CONTRACT[1]


def contract_text(contract: tuple[int, ...]) -> str:
    return ".".join(map(str, contract)) or "none"


def decoded[T](kind: type[T], data: object) -> T:
    return _decoded(kind, data, strict=True)


def read_answer[T](kind: type[T], data: object) -> T:
    return _decoded(kind, data, strict=False)


def encoded_error(error: SiftError) -> Json:
    return {
        "kind": next(name for name, kind in ERROR_KINDS.items() if isinstance(error, kind)),
        "code": error.code.value,
        "message": error.message,
        "remedy": error.remedy,
        "field": error.field,
    }


def decoded_error(data: Json) -> SiftError:
    if not isinstance(data, dict):
        raise TypeError(f"{data!r} is no error")
    kind = ERROR_KINDS.get(str(data.get("kind")), InternalError)
    message, remedy = str(data.get("message", "")), str(data.get("remedy") or "")
    said = PROBLEMS.get(str(data.get("code")), kind.usual)
    code = said if issubclass(kind, CODE_KINDS[said]) else kind.usual
    field = data.get("field")
    return kind(message, remedy, None if field is None else str(field), code=code)
