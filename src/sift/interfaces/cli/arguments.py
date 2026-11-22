import argparse
import dataclasses
import types
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, cast, get_args, get_origin, get_type_hints

from sift.api.described import Help, Place

type Placed = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    kind: object
    help: str
    place: bool
    given_by_place: bool


def _refused(request: type[object], name: str, problem: str) -> TypeError:
    return TypeError(f"{request.__name__}.{name}: {problem}")


def _described(request: type[object], name: str, hint: object) -> tuple[object, Help, bool]:
    kind, *metadata = cast("tuple[object, ...]", get_args(hint)) if get_origin(hint) is Annotated else (hint,)
    found = [part for part in metadata if isinstance(part, Help)]
    if not found:
        raise _refused(request, name, "no Help")
    return kind, found[0], any(isinstance(part, Place) for part in metadata)


def _left_out(kind: object) -> object | None:
    if get_origin(kind) is not types.UnionType:
        return None
    members = cast("tuple[object, ...]", get_args(kind))
    rest = [member for member in members if member is not types.NoneType]
    return rest[0] if len(rest) == 1 < len(members) else None


def _named(kind: object) -> str:
    return kind.__name__ if isinstance(kind, type) else repr(kind)


def _choices(kind: object) -> type[StrEnum] | None:
    return kind if isinstance(kind, type) and issubclass(kind, StrEnum) else None


def _argument(request: type[object], field: dataclasses.Field[object], hint: object) -> Argument:
    kind, described, place = _described(request, field.name, hint)
    default: object = field.default
    absent = default is dataclasses.MISSING
    if kind is bool:
        if default is not False:
            raise _refused(request, field.name, "a switch is off by default")
        return Argument(field.name, kind, described.text, place=False, given_by_place=False)
    if (optional := _left_out(kind)) is not None:
        if default is not None:
            raise _refused(request, field.name, "an option that may be left out has no default but None")
        kind = optional
    elif not absent:
        raise _refused(request, field.name, "an argument given by its place has no default")
    if kind is not str and _choices(kind) is None:
        raise _refused(request, field.name, f"the command line takes no {_named(kind)}")
    return Argument(field.name, kind, described.text, place=place, given_by_place=absent)


def arguments(request: type[object]) -> tuple[Argument, ...]:
    if not dataclasses.is_dataclass(request):
        raise TypeError(f"{request.__name__} is no dataclass")
    hints = cast("dict[str, object]", get_type_hints(request, include_extras=True))
    return tuple(_argument(request, field, hints[field.name]) for field in dataclasses.fields(request))


def _add(parser: argparse.ArgumentParser, argument: Argument, place: Placed) -> None:
    flag = argument.name if argument.given_by_place else f"--{argument.name.replace('_', '-')}"
    if argument.kind is bool:
        parser.add_argument(flag, action="store_true", help=argument.help)
    elif (choices := _choices(argument.kind)) is not None:
        parser.add_argument(flag, choices=[str(member) for member in choices], help=argument.help)
    else:
        parser.add_argument(flag, type=place if argument.place else str, help=argument.help)


def options(parser: argparse.ArgumentParser, request: type[object], place: Placed) -> None:
    for argument in arguments(request):
        _add(parser, argument, place)


def _value(argument: Argument, given: object) -> object:
    choices = _choices(argument.kind)
    return choices(given) if choices is not None and isinstance(given, str) else given


def request_of[R](request: type[R], namespace: argparse.Namespace) -> R:
    given = cast("dict[str, object]", vars(namespace))
    return request(**{argument.name: _value(argument, given[argument.name]) for argument in arguments(request)})
