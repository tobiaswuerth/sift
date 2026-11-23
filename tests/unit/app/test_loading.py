from sift.api.events import Event, Noted, Phase, PhaseDone, PhaseStarted
from sift.app.loading import prepared


def _on_the_cpu() -> tuple[str, ...]:
    return ("the models run on the CPU",)


def test_the_models_load_in_one_phase_and_each_notice_is_said_once_after() -> None:
    said: list[Event] = []
    prepared(said.append, (_on_the_cpu, _on_the_cpu))
    assert said == [PhaseStarted(Phase.LOAD, None), PhaseDone(2), Noted("the models run on the CPU")]
