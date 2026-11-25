import argparse
import json
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

from sift.api.collections import Audience
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
from sift.late import late

if TYPE_CHECKING:
    from sift.interfaces.server.endpoint import Endpoint

CLIENT = "sift.interfaces.server.client"


def as_json(answer: Answer) -> str:
    return printable_json(json.dumps(encoded(answer), ensure_ascii=False, indent=2, allow_nan=False))


def never(_: object) -> bool:
    return False


def acting(context: Context) -> Acting:
    models = partial(fresh_models, context.settings)
    return Acting(context.settings, context.report, context.clock, models, Audience.PEOPLE)


class Client(Protocol):
    def sent[R, A](self, endpoint: "Endpoint", act: Act[R, A], request: R, report: Reporter) -> A: ...


def answered[R, A](
    act: Act[R, A], namespace: argparse.Namespace, context: Context, made: Callable[[argparse.Namespace], R] | None
) -> A:
    request = request_of(act.request, namespace) if made is None else made(namespace)
    if context.server is None:
        return act.run(acting(context), request)
    return late(CLIENT, Client).sent(context.server, act, request, context.report)


def command[R, A: Answer](
    act: Act[R, A],
    show: Callable[[A, str, Style], Shown],
    failed: Callable[[A], bool] = never,
    made: Callable[[argparse.Namespace], R] | None = None,
) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        answer = answered(act, namespace, context, made)
        found = Shown((as_json(answer),), ()) if context.json else show(answer, context.platform, context.style)
        return found._replace(code=FAILED if failed(answer) else DONE)

    return Command(run, writes=act.writes)


def searched[R](act: Act[R, SearchResult]) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        started = context.clock()
        answer = answered(act, namespace, context, None)
        seconds = context.clock() - started
        first = cast("int", vars(namespace)["offset"]) + 1
        found = (
            Shown((as_json(answer),), ())
            if context.json
            else hits(answer, seconds, context.platform, context.style, first)
        )
        return found._replace(code=FAILED if answer.unsearched else DONE)

    return Command(run)


def needing_collections(made: Command) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        if not context.json and not chosen(context.settings.home, (), Audience.PEOPLE):
            return empty(context.platform, context.style)
        return made.run(namespace, context)

    return Command(run, writes=made.writes)


if TYPE_CHECKING:
    from sift.interfaces.server import client as _client

    _CLIENT: Client = _client
