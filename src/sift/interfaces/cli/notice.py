from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sift.interfaces.cli.shown.outcome import next_step
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled
from sift.late import late
from sift.storage.store_files import CATALOG, FORMER_CATALOG

if TYPE_CHECKING:
    from sift.interfaces.cli.updates import Checking

UPDATES = "sift.interfaces.cli.updates"
IMPORT = "import"
FORMER_HOME = "this home holds the index of sift 1.x, which sift 2 reads once it is imported"


class Updates(Protocol):
    def told(self, home: Path, installed: str, asked: "Checking | None" = None) -> str | None: ...


def former_home(home: Path, command: str, platform: str, style: Style) -> tuple[str, ...]:
    if command == IMPORT or (home / CATALOG).exists() or not (home / FORMER_CATALOG).exists():
        return ()
    return labelled("note", FORMER_HOME, Tone.WARNING, style), next_step(("sift", IMPORT), platform, style)


def notice(home: Path, installed: str, style: Style, checking: "Checking | None") -> tuple[str, ...]:
    line = late(UPDATES, Updates).told(home, installed, checking)
    return () if line is None else (labelled("note", line, Tone.WARNING, style),)


if TYPE_CHECKING:
    from sift.interfaces.cli import updates as _updates

    _UPDATES: Updates = _updates
