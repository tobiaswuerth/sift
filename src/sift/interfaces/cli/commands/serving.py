from importlib.metadata import version
from types import MappingProxyType

from sift.api.serving import Served, ServeHome
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, printable
from sift.interfaces.server import serving

PACKAGE = "sift-rag"


def _serve(acting: Acting, request: ServeHome) -> Served:
    return serving.serve(acting.settings, request, version(PACKAGE), acting.report)


def _stopped(served: Served, _platform: str, style: Style) -> Shown:
    return Shown((), (labelled("done", f"stopped serving {printable(served.url)}", Tone.DONE, style),))


COMMANDS = MappingProxyType({"serve": command(Act("serve", ServeHome, _serve), _stopped)})
