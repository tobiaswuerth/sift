from collections.abc import Mapping
from types import MappingProxyType

from sift.api.importing import Brought, HomeImported, ImportHome
from sift.app import importing
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.acts.indexing import readers_of
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.context import Command
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, table

HEADING = ("collection", "documents", "passages", "vectors")
COUNTS = frozenset({1, 2, 3})
KEPT = "sift 1.x's files are as they were: remove its stores and sift.db once these are as you want them"


def _imported(acting: Acting, request: ImportHome) -> HomeImported:
    def embedding() -> str:
        return acting.models().name

    return importing.imported(acting.settings.home, request, acting.report, readers_of(acting), embedding)


def _pair(before: int, now: int) -> str:
    return f"{before:,} -> {now:,}"


def _row(name: str, brought: Brought) -> tuple[str, ...]:
    return (
        name,
        _pair(brought.documents_before, brought.documents),
        _pair(brought.passages_before, brought.passages),
        _pair(brought.vectors_before, brought.vectors),
    )


def imported_table(done: HomeImported) -> tuple[str, ...]:
    rows = [_row(each.name, each.outcome) for each in done.collections if isinstance(each.outcome, Brought)]
    return table([HEADING, *rows], COUNTS) if rows else ()


def _shown(done: HomeImported, _platform: str, style: Style) -> Shown:
    shown = imported_table(done)
    return Shown(shown, (labelled("kept", KEPT, Tone.SECONDARY, style),) if shown else ())


def _failed(done: HomeImported) -> bool:
    return any(not isinstance(each.outcome, Brought) for each in done.collections)


COMMANDS: Mapping[str, Command] = MappingProxyType(
    {"import": command(Act("import", ImportHome, HomeImported, _imported, writes=True), _shown, _failed)}
)
