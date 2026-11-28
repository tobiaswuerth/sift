from sift.api.events import Event, Noted, Phase, PhaseDone, PhaseStarted
from sift.app.loading import prepared
from sift.inference.prepared import ModelsPrepared


def _built_on_the_cpu() -> ModelsPrepared:
    return ModelsPrepared(("the models run on the CPU",), built=True)


def _held() -> ModelsPrepared:
    return ModelsPrepared((), built=False)


def test_the_models_load_in_one_phase_and_each_notice_is_said_once_after() -> None:
    said: list[Event] = []
    prepared(said.append, (_built_on_the_cpu, _built_on_the_cpu))
    assert said == [PhaseStarted(Phase.LOAD, None), PhaseDone(2), Noted("the models run on the CPU")]


def test_the_load_counts_the_models_it_built_and_not_those_held() -> None:
    said: list[Event] = []
    prepared(said.append, (_held, _built_on_the_cpu, _held))
    assert said[:2] == [PhaseStarted(Phase.LOAD, None), PhaseDone(1)]
