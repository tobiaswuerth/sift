import argparse
import dataclasses
import re
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Annotated, cast, get_args, get_origin, get_type_hints, override

from sift.api.described import Help, Place, Read, Short

type Placing = Callable[[str], str]

PLAIN_KINDS = (str, int, float)
WHOLE_NUMBER = re.compile("-?[0-9]{1,18}")
DECIMAL = re.compile("-?[0-9]{1,9}([.][0-9]{1,9})?")


class How(Enum):
    POSITION = "position"
    OPTION = "option"
    SWITCH = "switch"
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
    short: str | None = None
    default: object = None
    read: Read | None = None


@dataclass(frozen=True, slots=True)
class Described:
    kind: object
    help: Help
    place: bool
    short: str | None
    read: Read | None


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
    short = next((f"-{part.letter}" for part in metadata if isinstance(part, Short)), None)
    read = next((part for part in metadata if isinstance(part, Read)), None)
    return Described(kind, found[0], any(isinstance(part, Place) for part in metadata), short, read)


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


def _read(request: type[object], name: str, kind: object, default: object) -> How:
    if get_origin(kind) is not tuple:
        raise _refused(request, name, "only a list is read by a function")
    if default != ():
        raise _refused(request, name, "a list read by a function has no default but ()")
    return How.READ


def _how(request: type[object], name: str, described: Described, default: object) -> How:
    kind = described.kind
    if described.read is not None:
        return _read(request, name, kind, default)
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
    return How.POSITION if absent else How.OPTION


def _argument(request: type[object], field: dataclasses.Field[object], hint: object) -> Argument:
    described = _described(request, field.name, hint)
    kind, default = described.kind, field.default
    how = _how(request, field.name, described, default)
    taken = (_left_out(kind) or kind) if how is How.OPTION else str if how in (How.ANY, How.SOME, How.READ) else kind
    if how is not How.SWITCH and taken not in PLAIN_KINDS and _choices(taken) is None:
        raise _refused(request, field.name, f"the command line takes no {_named(kind)}")
    if described.short is not None and how not in (How.OPTION, How.READ):
        raise _refused(request, field.name, "only an option given by name has a short flag")
    given = None if default is dataclasses.MISSING else default
    return Argument(
        field.name, taken, described.help.text, described.place, how, described.short, given, described.read
    )


def arguments(request: type[object]) -> tuple[Argument, ...]:
    if not dataclasses.is_dataclass(request):
        raise TypeError(f"{request.__name__} is no dataclass")
    hints = cast("dict[str, object]", get_type_hints(request, include_extras=True))
    return tuple(_argument(request, field, hints[field.name]) for field in dataclasses.fields(request))


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
    long = f"--{argument.name.replace('_', '-')}"
    named = [long] if argument.short is None else [argument.short, long]
    choices = _choices(argument.kind)
    if argument.how is How.SWITCH:
        parser.add_argument(long, action="store_true", help=argument.help)
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
        argument.name: argument.short or f"--{argument.name.replace('_', '-')}"
        if argument.how in (How.OPTION, How.SWITCH, How.READ)
        else argument.name
        for argument in arguments(request)
    }


def options(parser: argparse.ArgumentParser, request: type[object], place: Placing) -> None:
    for argument in arguments(request):
        _add(parser, argument, place)


def _value(argument: Argument, given: object) -> object:
    if argument.how in (How.ANY, How.SOME):
        return tuple(cast("list[str]", given))
    if argument.read is not None:
        return tuple(item for text in cast("list[str] | None", given) or () for item in argument.read.each(text))
    choices = _choices(argument.kind)
    return choices(given) if choices is not None and isinstance(given, str) else given


def request_of[R](request: type[R], namespace: argparse.Namespace) -> R:
    given = cast("dict[str, object]", vars(namespace))
    return request(**{argument.name: _value(argument, given[argument.name]) for argument in arguments(request)})
