import argparse
import contextlib
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO, cast, override

from sift.api.errors import InvalidRequestError, SiftError, StoreUnavailableError
from sift.api.wire import CONTRACT, contract_text, understood
from sift.domain.text.decoding import encodable
from sift.interfaces.cli.arguments import flags, options
from sift.interfaces.cli.context import Context
from sift.interfaces.cli.entries import ENTRIES, command_of
from sift.interfaces.cli.flags import file_or_id
from sift.interfaces.cli.logs import FOLDER, LOGGED, Log, heard_too, kept
from sift.interfaces.cli.notice import notice
from sift.interfaces.cli.shown.outcome import FAILED, INTERRUPTED, Shown, failure, stopped_writing
from sift.interfaces.cli.usage import Parser, asked_json, parsed
from sift.interfaces.console.live import Live, ticking
from sift.interfaces.console.progress import Looks
from sift.interfaces.console.streams import (
    ReaderLeftError,
    columns,
    descriptor,
    is_terminal,
    let_go,
    say,
    utf8,
    write_results,
)
from sift.interfaces.console.style import PLAIN, Asked, Colour, Style, Terminal, style_of
from sift.interfaces.console.text import printable_text
from sift.interfaces.console.windows import console, escapes_work
from sift.interfaces.server.endpoint import Endpoint, found, unshared
from sift.settings import VARIABLES, Settings, from_environment, user_folder, whole_path

if TYPE_CHECKING:
    from sift.interfaces.cli.updates import Checking

DESCRIPTION = "Fast local search over your documents and notes."
SETTINGS_HEADING = "settings, read from the environment:"
NO_TEXT_REMEDY = "name files and folders in UTF-8, or rename this one"
STOPPED_WATCHING = "stopped watching: the home process goes on with it; sift jobs lists it and stops it"
OTHER_SIFT_REMEDY = "stop sift serve where it runs, and start it again with this sift"
SERVE = "serve"
JSON_HELP = "answer in JSON"
COLOR_HELP = "colour what sift says"
VERSION_HELP = "say which sift this is"


class Metadata(Protocol):
    def version(self, distribution_name: str) -> str: ...


def _as_metadata(module: object) -> Metadata:
    return cast("Metadata", module)


class VersionAction(argparse.Action):
    def __init__(self, option_strings: Sequence[str], dest: str) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, help=VERSION_HELP)

    @override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        write_results(sys.stdout, (f"sift {_version()}",))
        parser.exit()


def _version() -> str:
    return _as_metadata(import_module("importlib.metadata")).version("sift-rag")


class StoppedError(Exception):
    said: tuple[str, ...]

    def __init__(self, said: tuple[str, ...]) -> None:
        super().__init__(said)
        self.said = said


@dataclass(frozen=True, slots=True)
class SettingsSource:
    read: Callable[[], Settings]
    user: Path | None
    cwd: Path


def _placed(user: Path | None, cwd: Path, written: str) -> str:
    if not written:
        raise argparse.ArgumentTypeError("an empty path names nothing")
    try:
        return str(whole_path(written, user, cwd))
    except InvalidRequestError as error:
        raise argparse.ArgumentTypeError(f"{error.message}: {error.remedy}") from None


def _settings_help() -> str:
    width = max(map(len, VARIABLES)) + 2
    return "\n".join([SETTINGS_HEADING, *(f"  {name:<{width}}{said}" for name, said in VARIABLES.items())])


def _file_or_id(user: Path | None, cwd: Path, written: str) -> str:
    try:
        return file_or_id(user, cwd, written)
    except InvalidRequestError as error:
        raise argparse.ArgumentTypeError(f"{error.message}: {error.remedy}") from None


def parser(user: Path | None, cwd: Path) -> argparse.ArgumentParser:
    built = Parser(
        prog="sift",
        description=DESCRIPTION,
        epilog=_settings_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    built.color = False
    built.add_argument("--version", action=VersionAction)
    built.add_argument("--json", action="store_true", help=JSON_HELP)
    built.add_argument("--color", choices=list(Colour), default=Colour.AUTO, help=COLOR_HELP)
    commands = built.add_subparsers(dest="command", metavar="COMMAND")
    for name, each in ENTRIES.items():
        chosen = commands.add_parser(name, help=each.help, description=each.help, allow_abbrev=False)
        chosen.color = False
        options(chosen, each.request, partial(_placed, user, cwd), partial(_file_or_id, user, cwd))
        chosen.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=JSON_HELP)
        chosen.add_argument("--color", choices=list(Colour), default=argparse.SUPPRESS, help=COLOR_HELP)
    return built


def _encoding(stream: TextIO | None) -> str | None:
    found: object = stream.encoding if stream is not None else None
    return found if isinstance(found, str) else None


def _style(err: TextIO | None, settings: Settings, choice: Colour, encoding: str | None) -> Style:
    terminal = is_terminal(err)
    target = None if err is None else descriptor(err)
    escapes = terminal and target is not None and escapes_work(console(), target)
    return style_of(Terminal(terminal, escapes, encoding), Asked(choice, settings.no_color, settings.dumb))


def _live(err: TextIO | None, settings: Settings, style: Style, platform: str, clock: Callable[[], float]) -> Live:
    redraw = is_terminal(err) and not settings.dumb
    width = partial(columns, err) if err is not None else lambda: 0
    return Live(err, Looks(style, platform), redraw=redraw, columns=width, clock=clock)


def _local_now() -> datetime:
    return datetime.now().astimezone()


@dataclass(frozen=True, slots=True)
class Host:
    platform: str = sys.platform
    clock: Callable[[], float] = time.perf_counter
    now: Callable[[], datetime] = _local_now
    checking: "Checking | None" = None


HOST = Host()


@dataclass(frozen=True, slots=True)
class _Chosen:
    command: str
    namespace: argparse.Namespace
    json: bool
    style: Style


def _server(settings: Settings, command: str) -> Endpoint | None:
    if command == SERVE:
        return None
    running = found(settings)
    if running is None or understood(running.contract):
        return running
    spoken, ours = contract_text(running.contract), contract_text(CONTRACT)
    said = f"the home process runs sift {running.version}, whose contract {spoken} "
    said += f"this sift {_version()} ({ours}) cannot read"
    raise StoreUnavailableError(said, OTHER_SIFT_REMEDY)


def _commanded(chosen: _Chosen, settings: Settings, host: Host, live: Live, log: Log | None) -> Shown:
    told = live if log is None else heard_too(live, log, host.clock)
    server: Endpoint | None = None
    command = command_of(chosen.command)
    try:
        server = _server(settings, chosen.command)
        for noted in () if server is None else unshared(server, settings):
            told(noted)
        context = Context(settings, host.platform, chosen.json, chosen.style, told, host.clock, server)
        with ticking(told.tick):
            return command.run(chosen.namespace, context)
    except (SiftError, OSError) as error:
        return failure(error, chosen.style, as_json=chosen.json, flags=flags(ENTRIES[chosen.command].request))
    except KeyboardInterrupt:
        if log is not None:
            log.stopped()
        if command.writes:
            kept = stopped_writing(live.view, host.platform, chosen.style)
            raise StoppedError(kept if server is None else (STOPPED_WATCHING,)) from None
        raise
    except Exception:
        if log is not None:
            log.failed()
        raise
    finally:
        live.close()


def _heading(command: str) -> str:
    return f"sift {_version()}, {command}"


def _logged(asked: _Chosen, settings: Settings, host: Host, live: Live) -> Shown:
    if asked.command not in LOGGED:
        return _commanded(asked, settings, host, live, None)
    with contextlib.ExitStack() as held:
        try:
            log = held.enter_context(kept(settings.home / FOLDER, asked.command, host.now(), _heading(asked.command)))
        except OSError as error:
            live.close()
            return failure(error, asked.style, as_json=asked.json)
        found = _commanded(asked, settings, host, live, log)
        say(log, found.messages)
        log.ended(found.code)
        return found


def _ran(
    argv: Sequence[str],
    source: SettingsSource,
    streams: tuple[TextIO | None, TextIO | None],
    host: Host,
    encoding: str | None,
) -> Shown | int:
    out, err = streams
    platform = host.platform
    if (odd := next((argument for argument in argv if not encodable(argument)), None)) is not None:
        refused = InvalidRequestError(f"an argument is no text: {odd!r}", NO_TEXT_REMEDY)
        return failure(refused, PLAIN, as_json=asked_json([each for each in argv if encodable(each)]))
    built = parser(source.user, source.cwd)
    namespace = parsed(built, argv, out, err)
    if isinstance(namespace, (int, Shown)):
        return namespace
    given = cast("dict[str, object]", vars(namespace))
    chosen = given["command"]
    if not isinstance(chosen, str):
        return Shown((built.format_help().rstrip("\n"),), ())
    try:
        settings = source.read()
    except InvalidRequestError as error:
        return failure(error, PLAIN, as_json=given["json"] is True)
    style = _style(err, settings, Colour(str(given["color"])), encoding)
    asked = _Chosen(chosen, namespace, given["json"] is True, style)
    found = _logged(asked, settings, host, _live(err, settings, style, platform, host.clock))
    if not settings.update_check or asked.json or not is_terminal(err):
        return found
    _shown(found, out, err)
    say(err, notice(settings.home, _version(), style, host.checking))
    return found.code


def _shown(found: Shown, out: TextIO | None, err: TextIO | None) -> None:
    write_results(out, map(printable_text, found.results) if is_terminal(out) else found.results)
    say(err, found.messages)


def _given(settings: Settings) -> Settings:
    return settings


def _refused(error: InvalidRequestError) -> Settings:
    raise error


def run(argv: Sequence[str], settings: Settings, out: TextIO | None, err: TextIO | None, host: Host = HOST) -> int:
    return _answered(argv, SettingsSource(partial(_given, settings), settings.user, settings.cwd), (out, err), host)


def _answered(
    argv: Sequence[str], source: SettingsSource, streams: tuple[TextIO | None, TextIO | None], host: Host
) -> int:
    out, err = streams
    encoding = _encoding(err)
    for stream in (out, err):
        utf8(stream)
    try:
        found = _ran(argv, source, (out, err), host, encoding)
        if isinstance(found, int):
            return found
        _shown(found, out, err)
    except ReaderLeftError as left:
        let_go(left.stream)
        return FAILED
    except OSError as error:
        say(err, failure(error, PLAIN).messages)
        return FAILED
    except StoppedError as stopped:
        say(err, stopped.said)
        return INTERRUPTED
    except KeyboardInterrupt:
        return INTERRUPTED
    return found.code


def main(argv: Sequence[str] | None = None, read_settings: Callable[[], Settings] = from_environment) -> None:
    given = sys.argv[1:] if argv is None else argv
    try:
        settings = read_settings()
    except InvalidRequestError as error:
        source = SettingsSource(partial(_refused, error), user_folder(), Path.cwd())
        raise SystemExit(_answered(given, source, (sys.stdout, sys.stderr), HOST)) from None
    raise SystemExit(run(given, settings, sys.stdout, sys.stderr))
