import argparse
import dataclasses
import json
from collections.abc import Callable

from sift.api.search import SearchResult
from sift.app.collections import chosen
from sift.interfaces.cli.arguments import request_of
from sift.interfaces.cli.context import Answer, Command, Context
from sift.interfaces.cli.shown.outcome import DONE, FAILED, Shown, empty
from sift.interfaces.cli.shown.searching import hits
from sift.interfaces.console.style import Style
from sift.interfaces.console.text import printable_json


def as_json(answer: Answer) -> str:
    return printable_json(json.dumps(dataclasses.asdict(answer), ensure_ascii=False, indent=2))


def never(_: object) -> bool:
    return False


def command[R, A: Answer](
    request: type[R],
    act: Callable[[Context, R], A],
    show: Callable[[A, str, Style], Shown],
    failed: Callable[[A], bool] = never,
) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        answer = act(context, request_of(request, namespace))
        found = Shown((as_json(answer),), ()) if context.json else show(answer, context.platform, context.style)
        return found._replace(code=FAILED if failed(answer) else DONE)

    return Command(run)


def searched[R](request: type[R], act: Callable[[Context, R], SearchResult], kind: str) -> Command:
    def run(namespace: argparse.Namespace, context: Context) -> Shown:
        started = context.clock()
        answer = act(context, request_of(request, namespace))
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
        if not chosen(context.settings.home, ()):
            return empty(context.platform, context.style)
        return made.run(namespace, context)

    return Command(run)
