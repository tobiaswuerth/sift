from collections.abc import Mapping
from types import MappingProxyType

from sift.api.importing import Brought, HomeImported, ImportHome
from sift.app import importing
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.acts.indexing import readers_of
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.context import Command
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style, Tone, paint
from sift.interfaces.console.text import labelled, printable

NAME = 24
HEADING = f"{'collection':<{NAME}}{'documents':>20}{'passages':>22}{'vectors':>22}"
KEPT = "sift 1.x's files are as they were: remove its stores and sift.db once these are as you want them"


def _imported(acting: Acting, request: ImportHome) -> HomeImported:
    return importing.imported(acting.settings.home, request, acting.report, readers_of(acting))


def _pair(before: int, now: int) -> str:
    return f"{before:,} -> {now:,}"


def _row(name: str, brought: Brought) -> str:
    counted = (
        _pair(brought.documents_before, brought.documents),
        _pair(brought.passages_before, brought.passages),
        _pair(brought.vectors_before, brought.vectors),
    )
    return f"{printable(name):<{NAME}}{counted[0]:>20}{counted[1]:>22}{counted[2]:>22}"


def _shown(done: HomeImported, _platform: str, style: Style) -> Shown:
    rows = tuple(_row(each.name, each.outcome) for each in done.collections if isinstance(each.outcome, Brought))
    table = (paint(HEADING, Tone.SECONDARY, style), *rows) if rows else ()
    return Shown(table, (labelled("kept", KEPT, Tone.SECONDARY, style),) if rows else ())


def _failed(done: HomeImported) -> bool:
    return any(not isinstance(each.outcome, Brought) for each in done.collections)


COMMANDS: Mapping[str, Command] = MappingProxyType(
    {"import": command(Act("import", ImportHome, HomeImported, _imported, writes=True), _shown, _failed)}
)
