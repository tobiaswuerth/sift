import argparse
import dataclasses
import re
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, cast, get_args, get_origin, get_type_hints, override

from sift.api.described import AbsolutePath, Bounds, Help
from sift.api.requests import choices_of, type_name
from sift.interfaces.cli.flags import RESERVED, Flag, flag_of

type Placing = Callable[[str], str]

PLAIN_KINDS = (str, int, float)
WHOLE_NUMBER = re.compile("-?[0-9]{1,18}")
DECIMAL = re.compile("-?[0-9]{1,9}([.][0-9]{1,9})?")


class How(Enum):
    POSITION = "position"
    OPTION = "option"
    SWITCH = "switch"
    TOGGLE = "toggle"
    READ = "read"
    ANY = "*"
    SOME = "+"


@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    kind: object
    help: str
    place: bool
    how: How
    flag: Flag
    default: object = None


@dataclass(frozen=True, slots=True)
class Described:
    kind: object
    help: Help
    place: bool
    flag: Flag
    bounds: Bounds | None = None


class Once(argparse.Action):
    @override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        given = f"given {self.dest}"
        if getattr(namespace, given, False):
            raise argparse.ArgumentError(self, "given twice: give it once")
        setattr(namespace, given, True)
        setattr(namespace, self.dest, values)


def _refused(request: type[object], name: str, problem: str) -> TypeError:
    return TypeError(f"{request.__name__}.{name}: {problem}")


def _described(request: type[object], name: str, hint: object) -> Described:
    kind, *metadata = cast("tuple[object, ...]", get_args(hint)) if get_origin(hint) is Annotated else (hint,)
    found = [part for part in metadata if isinstance(part, Help)]
    if not found:
        raise _refused(request, name, "no Help")
    if name in RESERVED:
        raise _refused(request, name, "sift's own option is named so")
    bounds = next((part for part in metadata if isinstance(part, Bounds)), None)
    return Described(kind, found[0], any(isinstance(part, AbsolutePath) for part in metadata), flag_of(name), bounds)


def limits(bounds: Bounds | None, default: object) -> str:
    if not isinstance(default, int) or isinstance(default, bool):
        return ""
    span = "" if bounds is None else f"{bounds.low:,} to {bounds.high:,}; "
    return f" ({span}default: {default:,})"


def _left_out(kind: object) -> object | None:
    if get_origin(kind) is not types.UnionType:
        return None
    members = cast("tuple[object, ...]", get_args(kind))
    rest = [member for member in members if member is not types.NoneType]
    return rest[0] if len(rest) == 1 < len(members) else None


def _many(kind: object) -> bool:
    return get_origin(kind) is tuple and get_args(kind) == (str, ...)


def _read(request: type[object], name: str, kind: object, default: object) -> How:
    if get_origin(kind) is not tuple:
        raise _refused(request, name, "only a list is read by a function")
    if default != ():
        raise _refused(request, name, "a list read by a function has no default but ()")
    return How.READ


def _how(request: type[object], name: str, described: Described, default: object) -> How:
    kind = described.kind
    if described.flag.read is not None:
        return _read(request, name, kind, default)
    absent = default is dataclasses.MISSING
    if kind is bool:
        if default is not False:
            raise _refused(request, name, "a switch is off by default")
        return How.SWITCH
    if _left_out(kind) is bool:
        if default is not None:
            raise _refused(request, name, "a yes or no that may be left out has no default but None")
        return How.TOGGLE
    if _many(kind):
        if not absent and default != ():
            raise _refused(request, name, "a list given by place has no default but ()")
        return How.SOME if absent else How.ANY
    if _left_out(kind) is not None:
        if default is not None:
            raise _refused(request, name, "an option that may be left out has no default but None")
        return How.OPTION
    return How.POSITION if absent else How.OPTION


def _argument(request: type[object], field: dataclasses.Field[object], hint: object) -> Argument:
    described = _described(request, field.name, hint)
    kind, default = described.kind, field.default
    how = _how(request, field.name, described, default)
    taken = (_left_out(kind) or kind) if how is How.OPTION else str if how in (How.ANY, How.SOME, How.READ) else kind
    if how not in (How.SWITCH, How.TOGGLE) and taken not in PLAIN_KINDS and choices_of(taken) is None:
        raise _refused(request, field.name, f"the command line takes no {type_name(kind)}")
    if described.flag.short is not None and how not in (How.OPTION, How.READ):
        raise _refused(request, field.name, "only an option given by name has a short flag")
    given = None if default is dataclasses.MISSING else default
    told = described.help.text + limits(described.bounds, given) + described.flag.syntax
    return Argument(field.name, taken, told, described.place, how, described.flag, given)


def arguments(request: type[object]) -> tuple[Argument, ...]:
    if not dataclasses.is_dataclass(request):
        raise TypeError(f"{request.__name__} is no dataclass")
    hints = cast("dict[str, object]", get_type_hints(request, include_extras=True))
    found = tuple(_argument(request, field, hints[field.name]) for field in dataclasses.fields(request))
    many = [argument.name for argument in found if argument.how in (How.ANY, How.SOME)]
    if len(many) > 1:
        raise _refused(request, many[1], "only one list is given by its places")
    return found


def _whole(written: str) -> int:
    if WHOLE_NUMBER.fullmatch(written) is None:
        raise argparse.ArgumentTypeError(f"{written!r} is no whole number: write up to 18 digits alone, e.g. 10")
    return int(written)


def _decimal(written: str) -> float:
    if DECIMAL.fullmatch(written) is None:
        raise argparse.ArgumentTypeError(f"{written!r} is no number: write digits with a point if need be, e.g. 0.5")
    return float(written)


def _converter(argument: Argument, place: Placing) -> Callable[[str], object]:
    if argument.place:
        return place
    return _whole if argument.kind is int else _decimal if argument.kind is float else str


def _add(parser: argparse.ArgumentParser, argument: Argument, place: Placing) -> None:
    named = [each for each in (argument.flag.short, argument.flag.long) if each is not None]
    choices = choices_of(argument.kind)
    if argument.how is How.SWITCH:
        parser.add_argument(*named, action="store_true", help=argument.help)
    elif argument.how is How.TOGGLE:
        parser.add_argument(
            *named, dest=argument.name, action=argparse.BooleanOptionalAction, default=None, help=argument.help
        )
    elif argument.how is How.READ:
        parser.add_argument(*named, dest=argument.name, action="append", help=argument.help)
    elif argument.how in (How.ANY, How.SOME):
        parser.add_argument(
            argument.name, nargs=argument.how.value, type=_converter(argument, place), help=argument.help
        )
    elif choices is not None:
        offered = [str(member) for member in choices]
        if argument.how is How.OPTION:
            parser.add_argument(
                *named, dest=argument.name, action=Once, choices=offered, default=argument.default, help=argument.help
            )
        else:
            parser.add_argument(argument.name, choices=offered, help=argument.help)
    elif argument.how is How.OPTION:
        converted = _converter(argument, place)
        parser.add_argument(
            *named, dest=argument.name, action=Once, type=converted, default=argument.default, help=argument.help
        )
    else:
        parser.add_argument(argument.name, type=_converter(argument, place), help=argument.help)


def flags(request: type[object]) -> dict[str, str]:
    return {
        argument.name: argument.flag.long or argument.flag.short or argument.name
        if argument.how in (How.OPTION, How.SWITCH, How.TOGGLE, How.READ)
        else argument.name
        for argument in arguments(request)
    }


def options(parser: argparse.ArgumentParser, request: type[object], place: Placing) -> None:
    for argument in arguments(request):
        _add(parser, argument, place)


def _value(argument: Argument, given: object) -> object:
    if argument.how in (How.ANY, How.SOME):
        return tuple(cast("list[str]", given))
    read = argument.flag.read
    if read is not None:
        return tuple(item for text in cast("list[str] | None", given) or () for item in read(text))
    choices = choices_of(argument.kind)
    return choices(given) if choices is not None and isinstance(given, str) else given


def request_of[R](request: type[R], namespace: argparse.Namespace) -> R:
    given = cast("dict[str, object]", vars(namespace))
    return request(**{argument.name: _value(argument, given[argument.name]) for argument in arguments(request)})
