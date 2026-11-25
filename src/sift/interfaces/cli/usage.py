import argparse
import contextlib
import difflib
import re
from collections.abc import Sequence
from typing import NoReturn, TextIO, cast, override

from sift.api.errors import InvalidRequestError
from sift.interfaces.cli.arguments import arguments, flags
from sift.interfaces.cli.entries import ENTRIES
from sift.interfaces.cli.shown.outcome import FAILED, Shown, failure
from sift.interfaces.console.style import PLAIN

ONE_VALUE = re.compile(r"argument (\S+): expected one argument")
ARGUMENT = re.compile(r"argument (\S+): (.+)", re.DOTALL)
NO_COMMAND = re.compile(r"argument COMMAND: invalid choice: '([^']*)'")
REQUIRED = "the following arguments are required: "
OWN_FLAGS = frozenset({"--json", "--color", "-h", "--help"})


class UsageError(Exception):
    message: str
    prog: str

    def __init__(self, message: str, prog: str) -> None:
        super().__init__(message)
        self.message = message
        self.prog = prog


class Parser(argparse.ArgumentParser):
    color: bool = False

    @override
    def error(self, message: str) -> NoReturn:
        raise UsageError(message, self.prog)


def asked_json(argv: Sequence[str]) -> bool:
    given = list(argv)
    return "--json" in (given[: given.index("--")] if "--" in given else given)


def _request(prog: str) -> type[object] | None:
    entry = ENTRIES.get(prog.removeprefix("sift").strip())
    return None if entry is None else entry.request


def _known(prog: str) -> frozenset[str]:
    request = _request(prog)
    arguments_of = () if request is None else arguments(request)
    found = {name for argument in arguments_of for name in (argument.flag.short, argument.flag.long)}
    return OWN_FLAGS | {name for name in found if name is not None}


def _field(request: type[object], names: Sequence[str]) -> str | None:
    named = set(names)
    held = (
        argument for argument in arguments(request) if {argument.flag.short, argument.flag.long, argument.name} & named
    )
    return next((argument.name for argument in held), None)


def _options(argv: Sequence[str]) -> list[str]:
    return list(argv)[: list(argv).index("--")] if "--" in argv else list(argv)


def _unknown(word: str, known: frozenset[str]) -> bool:
    return word.startswith("-") and word.split("=", 1)[0] not in known


def _unknown_dashed(prog: str, argv: Sequence[str]) -> bool:
    known = _known(prog)
    return any(_unknown(word, known) for word in _options(argv))


def _no_value(names: Sequence[str], prog: str, argv: Sequence[str]) -> str:
    words, known = _options(argv), _known(prog)
    typed = next((word for word in words if word in names), names[0])
    following = [words[at + 1] for at, word in enumerate(words[:-1]) if word in names]
    if any(_unknown(word, known) for word in following):
        return f"a value that starts with a dash is written {typed}=-value; see {prog} --help"
    return f"give {typed} a value; see {prog} --help"


def _nearest(written: str) -> str:
    nearest = difflib.get_close_matches(written, ENTRIES, n=1)
    return f"did you mean sift {nearest[0]}? " if nearest else ""


def _usage(wrong: UsageError, argv: Sequence[str]) -> str:
    unknown = NO_COMMAND.match(wrong.message)
    if unknown:
        return f"{_nearest(unknown.group(1))}see {wrong.prog} --help"
    asked = ONE_VALUE.fullmatch(wrong.message)
    if asked:
        return _no_value(asked.group(1).split("/"), wrong.prog, argv)
    if wrong.message.startswith(REQUIRED) and _unknown_dashed(wrong.prog, argv):
        return f"a value that starts with a dash goes after --, e.g. {wrong.prog} -- -value"
    return f"see {wrong.prog} --help"


def _refused_usage(wrong: UsageError, argv: Sequence[str]) -> Shown:
    remedy, request, found = _usage(wrong, argv), _request(wrong.prog), ARGUMENT.fullmatch(wrong.message)
    field = None if request is None or found is None else _field(request, found.group(1).split("/"))
    if request is None or found is None or field is None:
        return failure(InvalidRequestError(wrong.message, remedy), PLAIN, as_json=asked_json(argv))
    refused = InvalidRequestError(found.group(2), remedy, field)
    return failure(refused, PLAIN, as_json=asked_json(argv), flags=flags(request))


def parsed(
    built: argparse.ArgumentParser, argv: Sequence[str], out: TextIO | None, err: TextIO | None
) -> argparse.Namespace | Shown | int:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            namespace, extras = built.parse_known_args(argv)
        except SystemExit as stop:
            return stop.code if isinstance(stop.code, int) else FAILED
        except UsageError as wrong:
            return _refused_usage(wrong, argv)
    if extras:
        chosen = cast("dict[str, object]", vars(namespace)).get("command")
        prog = f"sift {chosen}" if isinstance(chosen, str) else "sift"
        return _refused_usage(UsageError(f"unrecognized arguments: {' '.join(extras)}", prog), argv)
    return namespace
