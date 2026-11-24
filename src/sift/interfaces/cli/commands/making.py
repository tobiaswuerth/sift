import argparse
import json
from collections.abc import Callable
from functools import partial

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


def as_json(answer: Answer) -> str:
    return printable_json(json.dumps(encoded(answer), ensure_ascii=False, indent=2))


def never(_: object) -> bool:
    return False


def acting(context: Context) -> Acting:
    return Acting(context.settings, context.report, context.clock, partial(fresh_models, context.settings))


def command[R, A: Answer](
    act: Act[R, A],
    show: Callable[[A, str, Style], Shown],
    failed: Callable[[A], bool] = never,
) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        answer = act.run(acting(context), request_of(act.request, namespace))
        found = Shown((as_json(answer),), ()) if context.json else show(answer, context.platform, context.style)
        return found._replace(code=FAILED if failed(answer) else DONE)

    return Command(run)


def searched[R](act: Act[R, SearchResult], kind: str) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        started = context.clock()
        answer = act.run(acting(context), request_of(act.request, namespace))
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
