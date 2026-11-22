import argparse
import dataclasses
import types
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Annotated, cast, get_args, get_origin, get_type_hints

from sift.api.described import Help, Place

type Placed = Callable[[str], str]


class How(Enum):
    POSITION = "position"
    OPTION = "option"
    SWITCH = "switch"
    ANY = "*"
    SOME = "+"


@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    kind: object
    help: str
    place: bool
    how: How


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


def _many(kind: object) -> bool:
    return get_origin(kind) is tuple and get_args(kind) == (str, ...)


def _choices(kind: object) -> type[StrEnum] | None:
    return kind if isinstance(kind, type) and issubclass(kind, StrEnum) else None


def _how(request: type[object], name: str, kind: object, default: object) -> How:
    absent = default is dataclasses.MISSING
    if kind is bool:
        if default is not False:
            raise _refused(request, name, "a switch is off by default")
        return How.SWITCH
    if _many(kind):
        if not absent and default != ():
            raise _refused(request, name, "a list given by place has no default but ()")
        return How.SOME if absent else How.ANY
    if _left_out(kind) is not None:
        if default is not None:
            raise _refused(request, name, "an option that may be left out has no default but None")
        return How.OPTION
    if not absent:
        raise _refused(request, name, "an argument given by its place has no default")
    return How.POSITION


def _argument(request: type[object], field: dataclasses.Field[object], hint: object) -> Argument:
    kind, described, place = _described(request, field.name, hint)
    default: object = field.default
    how = _how(request, field.name, kind, default)
    taken = _left_out(kind) if how is How.OPTION else str if how in (How.ANY, How.SOME) else kind
    if how is not How.SWITCH and taken is not str and _choices(taken) is None:
        raise _refused(request, field.name, f"the command line takes no {_named(kind)}")
    return Argument(field.name, taken, described.text, place, how)


def arguments(request: type[object]) -> tuple[Argument, ...]:
    if not dataclasses.is_dataclass(request):
        raise TypeError(f"{request.__name__} is no dataclass")
    hints = cast("dict[str, object]", get_type_hints(request, include_extras=True))
    return tuple(_argument(request, field, hints[field.name]) for field in dataclasses.fields(request))


def _add(parser: argparse.ArgumentParser, argument: Argument, place: Placed) -> None:
    named = f"--{argument.name.replace('_', '-')}" if argument.how in (How.OPTION, How.SWITCH) else argument.name
    choices = _choices(argument.kind)
    if argument.how is How.SWITCH:
        parser.add_argument(named, action="store_true", help=argument.help)
    elif argument.how in (How.ANY, How.SOME):
        convert = place if argument.place else str
        parser.add_argument(named, nargs=argument.how.value, type=convert, help=argument.help)
    elif choices is not None:
        parser.add_argument(named, choices=[str(member) for member in choices], help=argument.help)
    else:
        parser.add_argument(named, type=place if argument.place else str, help=argument.help)


def options(parser: argparse.ArgumentParser, request: type[object], place: Placed) -> None:
    for argument in arguments(request):
        _add(parser, argument, place)


def _value(argument: Argument, given: object) -> object:
    if argument.how in (How.ANY, How.SOME):
        return tuple(cast("list[str]", given))
    choices = _choices(argument.kind)
    return choices(given) if choices is not None and isinstance(given, str) else given


def request_of[R](request: type[R], namespace: argparse.Namespace) -> R:
    given = cast("dict[str, object]", vars(namespace))
    return request(**{argument.name: _value(argument, given[argument.name]) for argument in arguments(request)})
