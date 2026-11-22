import argparse
import contextlib
import dataclasses
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from types import MappingProxyType
from typing import ClassVar, Protocol, TextIO, cast

from sift.api.collections import AddCollection, CollectionAdded, ShowStatus, Status
from sift.api.errors import SiftError
from sift.app import collections
from sift.interfaces.cli.arguments import options, request_of
from sift.interfaces.cli.shown import Shown, added, failure, status
from sift.interfaces.console.streams import ReaderLeftError, let_go, say, utf8, write_results
from sift.settings import Settings, from_environment, whole_path

DONE, FAILED, USAGE, INTERRUPTED = 0, 1, 2, 130
DESCRIPTION = "Fast local search over your documents and notes."


@dataclass(frozen=True, slots=True)
class Context:
    settings: Settings
    platform: str
    json: bool


@dataclass(frozen=True, slots=True)
class Command:
    help: str
    request: type[object]
    run: Callable[[argparse.Namespace, Context], Shown]


class Answer(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


def as_json(answer: Answer) -> str:
    return json.dumps(dataclasses.asdict(answer), ensure_ascii=False, indent=2)


def command[R, A: Answer](
    help_text: str, request: type[R], act: Callable[[Settings, R], A], show: Callable[[A, str], Shown]
) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        answer = act(context.settings, request_of(request, namespace))
        return Shown((as_json(answer),), ()) if context.json else show(answer, context.platform)

    return Command(help_text, request, run)


def _add(settings: Settings, request: AddCollection) -> CollectionAdded:
    return collections.add(settings.home, request)


def _status(settings: Settings, _request: ShowStatus) -> Status:
    return collections.status(settings.home)


def _status_shown(result: Status, _platform: str) -> Shown:
    return status(result)


COMMANDS = MappingProxyType(
    {
        "add": command("add a folder as a collection", AddCollection, _add, added),
        "status": command(
            "show each collection: its folder and what its store holds", ShowStatus, _status, _status_shown
        ),
    }
)


def _placed(settings: Settings, written: str) -> str:
    return str(whole_path(written, settings.user, settings.cwd))


def parser(settings: Settings) -> argparse.ArgumentParser:
    built = argparse.ArgumentParser(prog="sift", description=DESCRIPTION)
    built.add_argument("--version", action="version", version=f"sift {version('sift-rag')}")
    commands = built.add_subparsers(dest="command", metavar="COMMAND")
    for name, each in COMMANDS.items():
        chosen = commands.add_parser(name, help=each.help, description=each.help)
        options(chosen, each.request, partial(_placed, settings))
        chosen.add_argument("--json", action="store_true", help="answer in JSON")
    return built


def _parsed(
    built: argparse.ArgumentParser, argv: Sequence[str], out: TextIO | None, err: TextIO | None
) -> argparse.Namespace | int:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            return built.parse_args(argv)
        except SystemExit as stop:
            return stop.code if isinstance(stop.code, int) else USAGE


def _ran(argv: Sequence[str], settings: Settings, out: TextIO | None, err: TextIO | None, platform: str) -> int:
    built = parser(settings)
    namespace = _parsed(built, argv, out, err)
    if isinstance(namespace, int):
        return namespace
    given = cast("dict[str, object]", vars(namespace))
    chosen = given["command"]
    if not isinstance(chosen, str):
        write_results(out, (built.format_help().rstrip("\n"),))
        return DONE
    shown = COMMANDS[chosen].run(namespace, Context(settings, platform, json=given["json"] is True))
    write_results(out, shown.results)
    say(err, shown.messages)
    return DONE


def run(
    argv: Sequence[str], settings: Settings, out: TextIO | None, err: TextIO | None, platform: str = sys.platform
) -> int:
    for stream in (out, err):
        utf8(stream)
    try:
        return _ran(argv, settings, out, err, platform)
    except ReaderLeftError as left:
        let_go(left.stream)
        return FAILED
    except (SiftError, OSError) as error:
        say(err, failure(error).messages)
        return FAILED
    except KeyboardInterrupt:
        return INTERRUPTED


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(sys.argv[1:] if argv is None else argv, from_environment(), sys.stdout, sys.stderr))
