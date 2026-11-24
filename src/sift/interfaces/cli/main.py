import argparse
import contextlib
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn, Protocol, TextIO, cast, override

from sift.api.collections import (
    ActivateCollections,
    AddCollection,
    DeactivateCollections,
    MoveCollections,
    RemoveCollection,
    ShowStatus,
)
from sift.api.documents import GetDocuments
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, SiftError, StoreUnavailableError
from sift.api.indexing import IndexCollections
from sift.api.jobs import ListJobs
from sift.api.links import FindMentions, FindSimilar, ListUnresolved, ShowLinks
from sift.api.search import SearchHybrid, SearchKeywords, SearchVectors
from sift.api.serving import ServeHome
from sift.domain.text.decoding import encodable
from sift.interfaces.cli.arguments import arguments, flags, options
from sift.interfaces.cli.context import Command, Commands, Context
from sift.interfaces.cli.logs import FOLDER, LOGGED, Log, heard_too, kept
from sift.interfaces.cli.notice import noticed
from sift.interfaces.cli.shown.outcome import FAILED, INTERRUPTED, Shown, failure
from sift.interfaces.console.live import Live, ticking
from sift.interfaces.console.progress import Looks
from sift.interfaces.console.streams import ReaderLeftError, columns, is_terminal, let_go, say, utf8, write_results
from sift.interfaces.console.style import PLAIN, Asked, Colour, Style, Terminal, style_of
from sift.interfaces.console.text import printable_text
from sift.interfaces.console.windows import console, escapes_work
from sift.interfaces.server.endpoint import Endpoint, found
from sift.settings import Settings, from_environment, whole_path

if TYPE_CHECKING:
    from sift.interfaces.cli.updates import Checking

DESCRIPTION = "Fast local search over your documents and notes."
NO_TEXT_REMEDY = "name files and folders in UTF-8, or rename this one"
ONE_VALUE = re.compile(r"argument (\S+?)(?:/\S+)?: expected one argument")
REQUIRED = "the following arguments are required: "
OWN_FLAGS = frozenset({"--json", "--color", "-h", "--help"})
STOPPED_WRITING = "stopped: what was written so far is kept"
STOPPED_WATCHING = "stopped watching: the home process goes on with it; sift jobs lists it and stops it"
OTHER_SIFT_REMEDY = "stop sift serve where it runs, and start it again with this sift"
SERVE = "serve"
JSON_HELP = "answer in JSON"
COLOR_HELP = "colour what sift says"
VERSION_HELP = "say which sift this is"
COMMANDS_PACKAGE = "sift.interfaces.cli.commands"


@dataclass(frozen=True, slots=True)
class CommandEntry:
    help: str
    request: type[object]
    module: str


ENTRIES = MappingProxyType(
    {
        "add": CommandEntry("add a folder as a collection", AddCollection, "collections"),
        "rm": CommandEntry(
            "remove a collection and its store; its files stay where they are", RemoveCollection, "collections"
        ),
        "move": CommandEntry(
            "move the stores of collections, or of all, to another folder or drive", MoveCollections, "collections"
        ),
        "activate": CommandEntry("search collections deactivated before again", ActivateCollections, "collections"),
        "deactivate": CommandEntry(
            "leave collections out of searches, and of index and embed unless named",
            DeactivateCollections,
            "collections",
        ),
        "index": CommandEntry("read what is new or changed in collections", IndexCollections, "indexing"),
        "query": CommandEntry(
            "find passages for a question: keyword and semantic, reranked", SearchHybrid, "searching"
        ),
        "search": CommandEntry("find passages by their words, ranked by BM25", SearchKeywords, "searching"),
        "embed": CommandEntry(
            "embed passages for semantic search, with the model SIFT_EMBED_MODEL names", EmbedCollections, "indexing"
        ),
        "vsearch": CommandEntry("find passages near a query in meaning", SearchVectors, "searching"),
        "get": CommandEntry("print documents by their ids", GetDocuments, "collections"),
        "similar": CommandEntry("find documents near one in meaning, by its own vectors", FindSimilar, "searching"),
        "mentions": CommandEntry("find documents naming a note without linking to it", FindMentions, "searching"),
        "links": CommandEntry("the links of a note: those it holds, and those into it", ShowLinks, "links"),
        "unresolved": CommandEntry("link targets no note answers to yet, most linked first", ListUnresolved, "links"),
        "jobs": CommandEntry("what sift serve does in the background, or stop one of its jobs", ListJobs, "jobs"),
        "status": CommandEntry("show each collection: its folder and what its store holds", ShowStatus, "collections"),
        "serve": CommandEntry(
            "serve this home: its models and codes held, commands answered at once", ServeHome, "serving"
        ),
    }
)


class Metadata(Protocol):
    def version(self, distribution_name: str) -> str: ...


def _as_metadata(module: object) -> Metadata:
    return cast("Metadata", module)


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
    said: str

    def __init__(self, said: str) -> None:
        super().__init__(said)
        self.said = said


def _as_commands(module: object) -> Commands:
    return cast("Commands", module)


def command_of(name: str) -> Command:
    return _as_commands(import_module(f"{COMMANDS_PACKAGE}.{ENTRIES[name].module}")).COMMANDS[name]


def _placed(settings: Settings, written: str) -> str:
    if not written:
        raise argparse.ArgumentTypeError("an empty path names nothing")
    try:
        return str(whole_path(written, settings.user, settings.cwd))
    except InvalidRequestError as error:
        raise argparse.ArgumentTypeError(f"{error.message}: {error.remedy}") from None


def parser(settings: Settings) -> argparse.ArgumentParser:
    built = Parser(prog="sift", description=DESCRIPTION, allow_abbrev=False)
    built.color = False
    built.add_argument("--version", action=VersionAction)
    built.add_argument("--json", action="store_true", help=JSON_HELP)
    built.add_argument("--color", choices=list(Colour), default=Colour.AUTO, help=COLOR_HELP)
    commands = built.add_subparsers(dest="command", metavar="COMMAND")
    for name, each in ENTRIES.items():
        chosen = commands.add_parser(name, help=each.help, description=each.help, allow_abbrev=False)
        chosen.color = False
        options(chosen, each.request, partial(_placed, settings))
        chosen.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=JSON_HELP)
        chosen.add_argument("--color", choices=list(Colour), default=argparse.SUPPRESS, help=COLOR_HELP)
    return built


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


def _parsed(
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


def _encoding(stream: TextIO | None) -> str | None:
    found: object = stream.encoding if stream is not None else None
    return found if isinstance(found, str) else None


def _style(err: TextIO | None, settings: Settings, choice: Colour, encoding: str | None) -> Style:
    terminal = is_terminal(err)
    escapes = terminal and err is not None and escapes_work(console(), err.fileno())
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
    running = found(settings.home)
    if running is None or running.version == _version():
        return running
    said = f"the home process runs sift {running.version}, this is sift {_version()}"
    raise StoreUnavailableError(said, OTHER_SIFT_REMEDY)


def _commanded(chosen: _Chosen, settings: Settings, host: Host, live: Live, log: Log | None) -> Shown:
    told = live if log is None else heard_too(live, log, host.clock)
    server: Endpoint | None = None
    command = command_of(chosen.command)
    try:
        server = _server(settings, chosen.command)
        context = Context(settings, host.platform, chosen.json, chosen.style, told, host.clock, server)
        with ticking(told.tick):
            return command.run(chosen.namespace, context)
    except (SiftError, OSError) as error:
        return failure(error, chosen.style, as_json=chosen.json, flags=flags(ENTRIES[chosen.command].request))
    except KeyboardInterrupt:
        if log is not None:
            log.stopped()
        if command.writes:
            raise StoppedError(STOPPED_WRITING if server is None else STOPPED_WATCHING) from None
        raise
    except Exception:
        if log is not None:
            log.failed()
        raise
    finally:
        live.close()


def _logged(asked: _Chosen, settings: Settings, host: Host, live: Live) -> Shown:
    if asked.command not in LOGGED:
        return _commanded(asked, settings, host, live, None)
    with kept(settings.home / FOLDER, asked.command, host.now(), f"sift {_version()}, {asked.command}") as log:
        found = _commanded(asked, settings, host, live, log)
        say(log, found.messages)
        log.ended(found.code)
        return found


def _ran(
    argv: Sequence[str],
    settings: Settings,
    streams: tuple[TextIO | None, TextIO | None],
    host: Host,
    encoding: str | None,
) -> Shown | int:
    out, err = streams
    platform = host.platform
    if (odd := next((argument for argument in argv if not encodable(argument)), None)) is not None:
        refused = InvalidRequestError(f"an argument is no text: {odd!r}", NO_TEXT_REMEDY)
        return failure(refused, PLAIN, as_json=asked_json([each for each in argv if encodable(each)]))
    built = parser(settings)
    namespace = _parsed(built, argv, out, err)
    if isinstance(namespace, (int, Shown)):
        return namespace
    given = cast("dict[str, object]", vars(namespace))
    chosen = given["command"]
    if not isinstance(chosen, str):
        return Shown((built.format_help().rstrip("\n"),), ())
    style = _style(err, settings, Colour(str(given["color"])), encoding)
    asked = _Chosen(chosen, namespace, given["json"] is True, style)
    found = _logged(asked, settings, host, _live(err, settings, style, platform, host.clock))
    if not settings.update_check or asked.json or not is_terminal(err):
        return found
    return noticed(found, settings.home, _version(), style, host.checking)


def run(argv: Sequence[str], settings: Settings, out: TextIO | None, err: TextIO | None, host: Host = HOST) -> int:
    encoding = _encoding(err)
    for stream in (out, err):
        utf8(stream)
    try:
        found = _ran(argv, settings, (out, err), host, encoding)
        if isinstance(found, int):
            return found
        write_results(out, map(printable_text, found.results) if is_terminal(out) else found.results)
        say(err, found.messages)
    except ReaderLeftError as left:
        let_go(left.stream)
        return FAILED
    except OSError as error:
        say(err, failure(error, PLAIN).messages)
        return FAILED
    except StoppedError as stopped:
        say(err, (stopped.said,))
        return INTERRUPTED
    except KeyboardInterrupt:
        return INTERRUPTED
    return found.code


def main(argv: Sequence[str] | None = None, read_settings: Callable[[], Settings] = from_environment) -> None:
    given = sys.argv[1:] if argv is None else argv
    try:
        settings = read_settings()
    except InvalidRequestError as error:
        refused = failure(error, PLAIN, as_json=asked_json(given))
        write_results(sys.stdout, refused.results)
        say(sys.stderr, refused.messages)
        raise SystemExit(refused.code) from None
    raise SystemExit(run(given, settings, sys.stdout, sys.stderr))
