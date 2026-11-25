from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled
from sift.late import late

if TYPE_CHECKING:
    from sift.interfaces.cli.updates import Checking

UPDATES = "sift.interfaces.cli.updates"


class Updates(Protocol):
    def told(self, home: Path, installed: str, asked: "Checking | None" = None) -> str | None: ...


def notice(home: Path, installed: str, style: Style, checking: "Checking | None") -> tuple[str, ...]:
    line = late(UPDATES, Updates).told(home, installed, checking)
    return () if line is None else (labelled("note", line, Tone.WARNING, style),)


if TYPE_CHECKING:
    from sift.interfaces.cli import updates as _updates

    _UPDATES: Updates = _updates
