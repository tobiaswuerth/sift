import argparse
import contextlib
import re
from collections.abc import Sequence
from typing import NoReturn, TextIO, cast, override

from sift.api.errors import InvalidRequestError
from sift.interfaces.cli.arguments import arguments
from sift.interfaces.cli.entries import ENTRIES
from sift.interfaces.cli.shown.outcome import FAILED, Shown, failure
from sift.interfaces.console.style import PLAIN

ONE_VALUE = re.compile(r"argument (\S+?)(?:/\S+)?: expected one argument")
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


def _known(command: str) -> frozenset[str]:
    request = ENTRIES[command].request
    found = {name for argument in arguments(request) for name in (argument.flag.short, argument.flag.long)}
    return OWN_FLAGS | {name for name in found if name is not None}


def _unknown_dashed(prog: str, argv: Sequence[str]) -> bool:
    known = _known(prog.removeprefix("sift").strip())
    words = list(argv)[: list(argv).index("--")] if "--" in argv else list(argv)
    return any(word.startswith("-") and word.split("=", 1)[0] not in known for word in words)


def _usage(wrong: UsageError, argv: Sequence[str]) -> str:
    asked = ONE_VALUE.fullmatch(wrong.message)
    if asked:
        return f"a value that starts with a dash is written {asked.group(1)}=-value; see {wrong.prog} --help"
    if wrong.message.startswith(REQUIRED) and _unknown_dashed(wrong.prog, argv):
        return f"a value that starts with a dash goes after --, e.g. {wrong.prog} -- -value"
    return f"see {wrong.prog} --help"


def _refused_usage(wrong: UsageError, argv: Sequence[str]) -> Shown:
    return failure(InvalidRequestError(wrong.message, _usage(wrong, argv)), PLAIN, as_json=asked_json(argv))


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
