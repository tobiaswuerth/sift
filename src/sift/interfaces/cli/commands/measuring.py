from collections.abc import Mapping
from types import MappingProxyType

from sift.api.bench import MeasureSearches, SearchesMeasured
from sift.app import bench
from sift.domain.text.wording import counted
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.context import Command
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled

HEADING = f"{'mode':<10}{'nDCG@10':>10}{'MRR@10':>10}{'Recall@10':>11}{'ms/query':>10}"


def _measured(acting: Acting, request: MeasureSearches) -> SearchesMeasured:
    return bench.measured(acting.settings.home, request, acting.models(), acting.report, acting.clock)


def _shown(measured: SearchesMeasured, _platform: str, style: Style) -> Shown:
    rows = tuple(
        f"{each.mode.value:<10}{each.ndcg:>10.3f}{each.mrr:>10.3f}{each.recall:>11.3f}{each.ms_per_query:>10,.0f}"
        for each in measured.modes
    )
    said = labelled("done", f"{counted(measured.cases, 'question')}, each asked in every mode", Tone.DONE, style)
    return Shown((HEADING, *rows), (said,))


COMMANDS: Mapping[str, Command] = MappingProxyType(
    {"bench": command(Act("bench", MeasureSearches, SearchesMeasured, _measured), _shown)}
)
