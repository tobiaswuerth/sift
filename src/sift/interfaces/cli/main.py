import argparse
import contextlib
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from types import MappingProxyType
from typing import NoReturn, Protocol, TextIO, cast, override

from sift.api.collections import AddCollection, ShowStatus
from sift.api.documents import GetDocuments
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, SiftError
from sift.api.indexing import IndexCollections
from sift.api.links import FindMentions, FindSimilar, ListUnresolved, ShowLinks
from sift.api.search import SearchHybrid, SearchKeywords, SearchVectors
from sift.domain.text.decoding import encodable
from sift.interfaces.cli.arguments import flags, options
from sift.interfaces.cli.context import Command, Commands, Context
from sift.interfaces.cli.shown.outcome import FAILED, Shown, failure
from sift.interfaces.console.live import Live, ticking
from sift.interfaces.console.progress import Looks
from sift.interfaces.console.streams import ReaderLeftError, columns, is_terminal, let_go, say, utf8, write_results
from sift.interfaces.console.style import PLAIN, Asked, Colour, Style, Terminal, style_of
from sift.interfaces.console.text import printable_text
from sift.interfaces.console.windows import console, escapes_work
from sift.settings import Settings, from_environment, whole_path

INTERRUPTED = 130
DESCRIPTION = "Fast local search over your documents and notes."
NO_TEXT_REMEDY = "name files and folders in UTF-8, or rename this one"
ONE_VALUE = re.compile(r"argument (\S+?)(?:/\S+)?: expected one argument")
STOPPED_WRITING = "stopped: what was written so far is kept"
JSON_HELP = "answer in JSON"
COLOR_HELP = "colour what sift says"
VERSION_HELP = "say which sift this is"
WRITERS = frozenset({"index", "embed"})
COMMANDS_PACKAGE = "sift.interfaces.cli.commands"


@dataclass(frozen=True, slots=True)
class CommandEntry:
    help: str
    request: type[object]
    module: str


ENTRIES = MappingProxyType(
    {
        "add": CommandEntry("add a folder as a collection", AddCollection, "collections"),
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
        "status": CommandEntry("show each collection: its folder and what its store holds", ShowStatus, "collections"),
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
        metadata = _as_metadata(import_module("importlib.metadata"))
        write_results(sys.stdout, (f"sift {metadata.version('sift-rag')}",))
        parser.exit()


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
    built = Parser(prog="sift", description=DESCRIPTION)
    built.add_argument("--version", action=VersionAction)
    built.add_argument("--json", action="store_true", help=JSON_HELP)
    built.add_argument("--color", choices=list(Colour), default=Colour.AUTO, help=COLOR_HELP)
    commands = built.add_subparsers(dest="command", metavar="COMMAND")
    for name, each in ENTRIES.items():
        chosen = commands.add_parser(name, help=each.help, description=each.help)
        options(chosen, each.request, partial(_placed, settings))
        chosen.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=JSON_HELP)
        chosen.add_argument("--color", choices=list(Colour), default=argparse.SUPPRESS, help=COLOR_HELP)
    return built


def _usage(wrong: UsageError) -> str:
    asked = ONE_VALUE.fullmatch(wrong.message)
    dashed = f"a value that starts with a dash is written {asked.group(1)}=-value; " if asked else ""
    return f"{dashed}see {wrong.prog} --help"


def _parsed(
    built: argparse.ArgumentParser, argv: Sequence[str], out: TextIO | None, err: TextIO | None
) -> argparse.Namespace | Shown | int:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            return built.parse_args(argv)
        except SystemExit as stop:
            return stop.code if isinstance(stop.code, int) else FAILED
        except UsageError as wrong:
            return failure(InvalidRequestError(wrong.message, _usage(wrong)), PLAIN, as_json="--json" in argv)


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


@dataclass(frozen=True, slots=True)
class Host:
    platform: str = sys.platform
    clock: Callable[[], float] = time.perf_counter


HOST = Host()


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
        return failure(InvalidRequestError(f"an argument is no text: {odd!r}", NO_TEXT_REMEDY), PLAIN)
    built = parser(settings)
    namespace = _parsed(built, argv, out, err)
    if isinstance(namespace, (int, Shown)):
        return namespace
    given = cast("dict[str, object]", vars(namespace))
    chosen = given["command"]
    if not isinstance(chosen, str):
        return Shown((built.format_help().rstrip("\n"),), ())
    style = _style(err, settings, Colour(str(given["color"])), encoding)
    live = _live(err, settings, style, platform, host.clock)
    try:
        context = Context(settings, platform, given["json"] is True, style, live, host.clock)
        with ticking(live.tick):
            return command_of(chosen).run(namespace, context)
    except (SiftError, OSError) as error:
        return failure(error, style, as_json=given["json"] is True, flags=flags(ENTRIES[chosen].request))
    except KeyboardInterrupt:
        if chosen in WRITERS:
            raise StoppedError(STOPPED_WRITING) from None
        raise
    finally:
        live.close()


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
    try:
        settings = read_settings()
    except InvalidRequestError as error:
        refused = failure(error, PLAIN)
        say(sys.stderr, refused.messages)
        raise SystemExit(refused.code) from None
    raise SystemExit(run(sys.argv[1:] if argv is None else argv, settings, sys.stdout, sys.stderr))
