import sys
from collections.abc import Mapping
from importlib.metadata import version
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from sift.api.events import Reporter
from sift.api.package import PACKAGE
from sift.api.serving import Served, ServeHome
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.context import Command
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, printable
from sift.interfaces.server.proxy import IDLE_SECONDS, Pipes, Starting, proxied
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


def _stopped(served: Served, _platform: str, style: Style) -> Shown:
    if not served.url:
        return Shown((), ())
    return Shown((), (labelled("done", f"stopped serving {printable(served.url)}", Tone.DONE, style),))


COMMANDS: Mapping[str, Command] = MappingProxyType(
    {"serve": command(Act("serve", ServeHome, Served, _serve), _stopped)}
)

if TYPE_CHECKING:
    from sift.interfaces.server import serving as _serving

    _SERVES: HomeServes = _serving
