import argparse
import json
from collections.abc import Callable
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from sift.api.events import Reporter
from sift.api.search import SearchResult
from sift.api.wire import encoded
from sift.app.collections import chosen
from sift.interfaces.acts.acting import Act, Acting, fresh_models
from sift.interfaces.cli.arguments import request_of
from sift.interfaces.cli.context import Answer, Command, Context
from sift.interfaces.cli.shown.outcome import DONE, FAILED, Shown, empty
from sift.interfaces.cli.shown.searching import hits
from sift.interfaces.console.style import Style
from sift.interfaces.console.text import printable_json

if TYPE_CHECKING:
    from sift.interfaces.server.endpoint import Endpoint

CLIENT = "sift.interfaces.server.client"


def as_json(answer: Answer) -> str:
    return printable_json(json.dumps(encoded(answer), ensure_ascii=False, indent=2, allow_nan=False))


def never(_: object) -> bool:
    return False


def acting(context: Context) -> Acting:
    return Acting(context.settings, context.report, context.clock, partial(fresh_models, context.settings))


class Client(Protocol):
    def sent[R, A](self, endpoint: "Endpoint", act: Act[R, A], request: R, report: Reporter) -> A: ...


def _as_client(module: object) -> Client:
    return cast("Client", module)


def answered[R, A](act: Act[R, A], namespace: argparse.Namespace, context: Context) -> A:
    request = request_of(act.request, namespace)
    if context.server is None:
        return act.run(acting(context), request)
    return _as_client(import_module(CLIENT)).sent(context.server, act, request, context.report)


def command[R, A: Answer](
    act: Act[R, A],
    show: Callable[[A, str, Style], Shown],
    failed: Callable[[A], bool] = never,
) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        answer = answered(act, namespace, context)
        found = Shown((as_json(answer),), ()) if context.json else show(answer, context.platform, context.style)
        return found._replace(code=FAILED if failed(answer) else DONE)

    return Command(run)


def searched[R](act: Act[R, SearchResult], kind: str) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        started = context.clock()
        answer = answered(act, namespace, context)
        seconds = context.clock() - started
        found = (
            Shown((as_json(answer),), ())
            if context.json
            else hits(answer, seconds, context.platform, context.style, kind)
        )
        return found._replace(code=FAILED if answer.unsearched else DONE)

    return Command(run)


def needing_collections(made: Command) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        if not context.json and not chosen(context.settings.home, ()):
            return empty(context.platform, context.style)
        return made.run(namespace, context)

    return Command(run)
