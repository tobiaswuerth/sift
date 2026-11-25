import argparse
import sys
from collections.abc import Callable, Mapping
from importlib.metadata import version
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from sift.api.events import Event, Listening, Reporter
from sift.api.package import PACKAGE
from sift.api.serving import OpenUi, Served, ServeHome, UiOpened
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.cli.arguments import request_of
from sift.interfaces.cli.commands.making import acting, as_json, command
from sift.interfaces.cli.context import Command, Context
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, printable
from sift.interfaces.server.endpoint import found
from sift.interfaces.server.proxy import IDLE_SECONDS, LOCALHOST, Pipes, Starting, proxied
from sift.interfaces.web.view import PREFIX
from sift.late import late
from sift.settings import Settings

SERVING = "sift.interfaces.server.serving"


class HomeServes(Protocol):
    def serve(self, settings: Settings, request: ServeHome, version: str, report: Reporter) -> Served: ...


def _serve(acting: Acting, request: ServeHome) -> Served:
    if request.stdio:
        starting = Starting(idle=request.idle_exit or IDLE_SECONDS)
        proxied(acting.settings, acting.report, Pipes(sys.stdin.buffer, sys.stdout.buffer), starting)
        return Served("")
    return late(SERVING, HomeServes).serve(acting.settings, request, version(PACKAGE), acting.report)


def opened_once_listening(report: Reporter, opening: Callable[[str], None]) -> Reporter:
    def said(event: Event) -> None:
        report(event)
        if isinstance(event, Listening):
            opening(f"{event.url}{PREFIX}")

    return said


def _unopened(_url: str) -> None:
    return None


def _ui(acting: Acting, request: OpenUi, browse: Callable[[str], None]) -> UiOpened:
    opening = _unopened if request.no_browser else browse
    endpoint = found(acting.settings)
    if endpoint is not None:
        url = f"http://{LOCALHOST}:{endpoint.port}{PREFIX}"
        opening(url)
        return UiOpened(url, served=False)

    said = opened_once_listening(acting.report, opening)
    served = late(SERVING, HomeServes).serve(acting.settings, ServeHome(port=0), version(PACKAGE), said)
    return UiOpened(f"{served.url}{PREFIX}", served=True)


def _shown_opened(opened: UiOpened, style: Style) -> Shown:
    if opened.served:
        return Shown((), (labelled("done", f"stopped serving {printable(opened.url)}", Tone.DONE, style),))
    return Shown((), (labelled("ui", printable(opened.url)),))


def _ui_run(namespace: argparse.Namespace, context: Context) -> Shown:
    opened = _ui(acting(context), request_of(OpenUi, namespace), context.browse)
    return Shown((as_json(opened),), ()) if context.json else _shown_opened(opened, context.style)


def _stopped(served: Served, _platform: str, style: Style) -> Shown:
    if not served.url:
        return Shown((), ())
    return Shown((), (labelled("done", f"stopped serving {printable(served.url)}", Tone.DONE, style),))


COMMANDS: Mapping[str, Command] = MappingProxyType(
    {
        "serve": command(Act("serve", ServeHome, Served, _serve), _stopped),
        "ui": Command(_ui_run),
    }
)

if TYPE_CHECKING:
    from sift.interfaces.server import serving as _serving

    _SERVES: HomeServes = _serving
