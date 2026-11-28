from collections.abc import Callable, Sequence

from sift.api.events import Event, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.inference.prepared import ModelsPrepared

type Load = Callable[[], ModelsPrepared]


def unreported(_: Event) -> None:
    return None


def compiled(report: Reporter, compile_models: Load) -> None:
    report(PhaseStarted(Phase.COMPILE, None))
    found = compile_models()
    report(PhaseDone(int(found.built)))
    for notice in found.notices:
        report(Noted(notice))


def prepared(report: Reporter, loads: Sequence[Load]) -> None:
    report(PhaseStarted(Phase.LOAD, None))
    found = [load() for load in loads]
    notices = [notice for each in found for notice in each.notices]
    report(PhaseDone(sum(each.built for each in found)))
    for notice in dict.fromkeys(notices):
        report(Noted(notice))
