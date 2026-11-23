import argparse
import contextlib
import dataclasses
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from types import MappingProxyType
from typing import ClassVar, Protocol, TextIO, cast

from sift.api.collections import AddCollection, CollectionAdded, ShowStatus, Status
from sift.api.documents import Documents, GetDocuments
from sift.api.embedding import EmbedCollections, EmbedResult
from sift.api.errors import InvalidRequestError, SiftError
from sift.api.events import Reporter
from sift.api.indexing import IndexCollections, IndexResult, Skipped
from sift.api.links import Links, ListUnresolved, ShowLinks, Unresolved
from sift.api.search import SearchHybrid, SearchKeywords, SearchVectors
from sift.app import collections, documents, embedding, indexing, links, query, search, vsearch
from sift.inference.loading import Deferred, deferred
from sift.interfaces.cli import shown
from sift.interfaces.cli.arguments import options, request_of
from sift.interfaces.cli.shown import Shown
from sift.interfaces.console.live import Live
from sift.interfaces.console.progress import Looks
from sift.interfaces.console.streams import ReaderLeftError, columns, is_terminal, let_go, say, utf8, write_results
from sift.interfaces.console.style import PLAIN, Asked, Colour, Style, Terminal, style_of
from sift.interfaces.console.windows import console, escapes_work
from sift.settings import Settings, from_environment, whole_path

DONE, FAILED, USAGE, INTERRUPTED = 0, 1, 2, 130
DESCRIPTION = "Fast local search over your documents and notes."
NO_TEXT_REMEDY = "name files and folders in UTF-8, or rename this one"


@dataclass(frozen=True, slots=True)
class Context:
    settings: Settings
    platform: str
    json: bool
    style: Style
    report: Reporter
    clock: Callable[[], float] = time.perf_counter


class Answer(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


@dataclass(frozen=True, slots=True)
class Command:
    help: str
    request: type[object]
    run: Callable[[argparse.Namespace, Context], Shown]


def as_json(answer: Answer) -> str:
    return json.dumps(dataclasses.asdict(answer), ensure_ascii=False, indent=2)


def command[R, A: Answer](
    help_text: str,
    request: type[R],
    act: Callable[[Context, R], A],
    show: Callable[[A, str, Style], Shown],
    failed: Callable[[A], bool] = lambda _: False,
) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        answer = act(context, request_of(request, namespace))
        found = Shown((as_json(answer),), ()) if context.json else show(answer, context.platform, context.style)
        return found._replace(failed=failed(answer))

    return Command(help_text, request, run)


def _add(context: Context, request: AddCollection) -> CollectionAdded:
    return collections.add(context.settings.home, request)


def _status(context: Context, request: ShowStatus) -> Status:
    return collections.status(context.settings.home, request)


def _index(context: Context, request: IndexCollections) -> IndexResult:
    return indexing.index(context.settings.home, request, context.report)


def _get(context: Context, request: GetDocuments) -> Documents:
    return documents.documents(context.settings.home, request)


def _search(namespace: argparse.Namespace, context: Context) -> Shown:
    started = context.clock()
    answer = search.search(context.settings.home, request_of(SearchKeywords, namespace), context.clock)
    if context.json:
        return Shown((as_json(answer),), (), failed=bool(answer.unsearched))
    return shown.hits(answer, context.clock() - started, context.platform, context.style)


def _models(context: Context) -> Deferred:
    return deferred(context.settings.embed_model, context.settings.rerank_model)


def _embed(context: Context, request: EmbedCollections) -> EmbedResult:
    return embedding.embed(context.settings.home, request, context.report, _models(context))


def _vsearch(namespace: argparse.Namespace, context: Context) -> Shown:
    started = context.clock()
    request = request_of(SearchVectors, namespace)
    answer = vsearch.vsearch(context.settings.home, request, _models(context))
    if context.json:
        return Shown((as_json(answer),), (), failed=bool(answer.unsearched))
    return shown.hits(answer, context.clock() - started, context.platform, context.style, "semantic")


def _query(namespace: argparse.Namespace, context: Context) -> Shown:
    started = context.clock()
    answer = query.query(context.settings.home, request_of(SearchHybrid, namespace), _models(context))
    if context.json:
        return Shown((as_json(answer),), (), failed=bool(answer.unsearched))
    return shown.hits(answer, context.clock() - started, context.platform, context.style, "hybrid")


def _links(context: Context, request: ShowLinks) -> Links:
    return links.links(context.settings.home, request)


def _unresolved(context: Context, request: ListUnresolved) -> Unresolved:
    return links.unresolved(context.settings.home, request)


def _some_skipped(result: IndexResult | EmbedResult) -> bool:
    return any(isinstance(each.outcome, Skipped) for each in result.collections)


def _some_unread(found: Links | Unresolved) -> bool:
    return bool(found.unsearched)


def _some_unfound(found: Documents) -> bool:
    return bool(found.unfound)


COMMANDS = MappingProxyType(
    {
        "add": command("add a folder as a collection", AddCollection, _add, shown.added),
        "index": command(
            "read what is new or changed in collections", IndexCollections, _index, shown.indexed, _some_skipped
        ),
        "query": Command("find passages for a question: keyword and semantic, reranked", SearchHybrid, _query),
        "search": Command("find passages by their words, ranked by BM25", SearchKeywords, _search),
        "embed": command(
            "embed passages for semantic search, with the model SIFT_EMBED_MODEL names",
            EmbedCollections,
            _embed,
            shown.embedded,
            _some_skipped,
        ),
        "vsearch": Command("find passages near a query in meaning", SearchVectors, _vsearch),
        "get": command("print documents by their ids", GetDocuments, _get, shown.documents, _some_unfound),
        "links": command(
            "the links of a note: those it holds, and those into it", ShowLinks, _links, shown.links, _some_unread
        ),
        "unresolved": command(
            "link targets no note answers to yet, most linked first",
            ListUnresolved,
            _unresolved,
            shown.unresolved,
            _some_unread,
        ),
        "status": command(
            "show each collection: its folder and what its store holds", ShowStatus, _status, shown.status
        ),
    }
)


def _placed(settings: Settings, written: str) -> str:
    if not written:
        raise argparse.ArgumentTypeError("an empty path names nothing")
    return str(whole_path(written, settings.user, settings.cwd))


def _is_text(argument: str) -> bool:
    try:
        argument.encode()
    except UnicodeEncodeError:
        return False
    return True


def parser(settings: Settings) -> argparse.ArgumentParser:
    built = argparse.ArgumentParser(prog="sift", description=DESCRIPTION)
    built.add_argument("--version", action="version", version=f"sift {version('sift-rag')}")
    commands = built.add_subparsers(dest="command", metavar="COMMAND")
    for name, each in COMMANDS.items():
        chosen = commands.add_parser(name, help=each.help, description=each.help)
        options(chosen, each.request, partial(_placed, settings))
        chosen.add_argument("--json", action="store_true", help="answer in JSON")
        chosen.add_argument("--color", choices=list(Colour), default=Colour.AUTO, help="colour what sift says")
    return built


def _parsed(
    built: argparse.ArgumentParser, argv: Sequence[str], out: TextIO | None, err: TextIO | None
) -> argparse.Namespace | int:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            return built.parse_args(argv)
        except SystemExit as stop:
            return stop.code if isinstance(stop.code, int) else USAGE


def _style(err: TextIO | None, settings: Settings, choice: Colour) -> Style:
    terminal = is_terminal(err)
    escapes = terminal and err is not None and escapes_work(console(), err.fileno())
    encoding: object = err.encoding if err is not None else None
    stream = Terminal(terminal, escapes, encoding if isinstance(encoding, str) else None)
    return style_of(stream, Asked(choice, settings.no_color, settings.dumb))


def _live(err: TextIO | None, settings: Settings, style: Style, platform: str) -> Live:
    redraw = is_terminal(err) and not settings.dumb
    width = partial(columns, err) if err is not None else lambda: 0
    return Live(err, Looks(style, platform), redraw=redraw, columns=width, clock=time.perf_counter)


@dataclass(frozen=True, slots=True)
class Host:
    platform: str = sys.platform
    clock: Callable[[], float] = time.perf_counter


HOST = Host()


def _ran(
    argv: Sequence[str], settings: Settings, streams: tuple[TextIO | None, TextIO | None], host: Host
) -> Shown | int:
    out, err = streams
    platform = host.platform
    if (odd := next((argument for argument in argv if not _is_text(argument)), None)) is not None:
        return shown.failure(InvalidRequestError(f"an argument is no text: {odd!r}", NO_TEXT_REMEDY), PLAIN)
    built = parser(settings)
    namespace = _parsed(built, argv, out, err)
    if isinstance(namespace, int):
        return namespace
    given = cast("dict[str, object]", vars(namespace))
    chosen = given["command"]
    if not isinstance(chosen, str):
        return Shown((built.format_help().rstrip("\n"),), ())
    style = _style(err, settings, Colour(str(given["color"])))
    live = _live(err, settings, style, platform)
    try:
        context = Context(settings, platform, given["json"] is True, style, live, host.clock)
        return COMMANDS[chosen].run(namespace, context)
    except (SiftError, OSError) as error:
        return shown.failure(error, style)
    finally:
        live.close()


def run(argv: Sequence[str], settings: Settings, out: TextIO | None, err: TextIO | None, host: Host = HOST) -> int:
    for stream in (out, err):
        utf8(stream)
    try:
        found = _ran(argv, settings, (out, err), host)
        if isinstance(found, int):
            return found
        write_results(out, found.results)
    except ReaderLeftError as left:
        let_go(left.stream)
        return FAILED
    except OSError as error:
        say(err, shown.failure(error, PLAIN).messages)
        return FAILED
    except KeyboardInterrupt:
        return INTERRUPTED
    say(err, found.messages)
    return FAILED if found.failed else DONE


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(sys.argv[1:] if argv is None else argv, from_environment(), sys.stdout, sys.stderr))
