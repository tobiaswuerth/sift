from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled

if TYPE_CHECKING:
    from sift.interfaces.cli.updates import Checking

UPDATES = "sift.interfaces.cli.updates"


class Updates(Protocol):
    def told(self, home: Path, installed: str, asked: "Checking | None" = None) -> str | None: ...


def _as_updates(module: object) -> Updates:
    return cast("Updates", module)


def notice(home: Path, installed: str, style: Style, checking: "Checking | None") -> tuple[str, ...]:
    line = _as_updates(import_module(UPDATES)).told(home, installed, checking)
    return () if line is None else (labelled("note", line, Tone.WARNING, style),)
