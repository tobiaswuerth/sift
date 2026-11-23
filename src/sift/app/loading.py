from collections.abc import Callable, Sequence

from sift.api.events import Event, Noted, Phase, PhaseDone, PhaseStarted, Reporter

type Load = Callable[[], tuple[str, ...]]


def unreported(_: Event) -> None:
    return None


def prepared(report: Reporter, loads: Sequence[Load]) -> None:
    report(PhaseStarted(Phase.LOAD, None))
    notices = [notice for load in loads for notice in load()]
    report(PhaseDone(len(loads)))
    for notice in dict.fromkeys(notices):
        report(Noted(notice))
