import math
from collections.abc import Callable
from functools import partial

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.plan.segments import POLICY, Held, Policy, merges

SMALL = Policy(factor=3, floor=10, most=500)


def _held(live: int, dead: int) -> Held:
    return Held(live + dead, dead)


held = st.builds(_held, st.integers(0, 2_000), st.integers(0, 300))


def merged(segments: list[Held], runs: tuple[range, ...]) -> list[Held]:
    starts = {run.start: run for run in runs}
    out: list[Held] = []
    at = 0
    while at < len(segments):
        run = starts.get(at, range(at, at + 1))
        if run in runs:
            out.append(Held(sum(each.chunks - each.dead for each in segments[run.start : run.stop]), 0))
        else:
            out.append(segments[at])
        at = run.stop
    return out


def settled(segments: list[Held], policy: Policy) -> tuple[list[Held], int, int]:
    written = rounds = 0
    while runs := merges(segments, policy):
        written += sum(segments[i].chunks - segments[i].dead for run in runs for i in run)
        segments = merged(segments, runs)
        rounds += 1
    return segments, written, rounds


@given(st.lists(held, max_size=60), st.sampled_from([SMALL, POLICY]))
def test_merges_are_disjoint_runs_of_neighbours_none_larger_than_the_most(segments: list[Held], policy: Policy) -> None:
    runs = merges(segments, policy)
    covered = [i for run in runs for i in run]
    assert len(covered) == len(set(covered))
    assert all(0 <= run.start < run.stop <= len(segments) for run in runs)
    assert [run.start for run in runs] == sorted(run.start for run in runs)
    for run in runs:
        live = sum(segments[i].chunks - segments[i].dead for i in run)
        assert len(run) == 1 or (len(run) == policy.factor and live <= policy.most)


@given(st.lists(held, max_size=60))
def test_a_segment_more_than_a_third_dead_is_rewritten(segments: list[Held]) -> None:
    covered = {i for run in merges(segments, SMALL) for i in run}
    assert all(i in covered for i, each in enumerate(segments) if 3 * each.dead > each.chunks)


@given(st.lists(st.integers(1, 400), min_size=1, max_size=300))
def test_appended_segments_settle_into_few_and_are_written_a_few_times_each(added: list[int]) -> None:
    segments: list[Held] = []
    written = 0
    for size in added:
        segments, more, _ = settled([*segments, Held(size, 0)], SMALL)
        written += more
    total = sum(added)
    levels = math.log(max(total, SMALL.floor) / SMALL.floor, SMALL.factor) + 2
    assert written <= total * levels
    large = sum(each.chunks > SMALL.most // SMALL.factor for each in segments)
    assert len(segments) <= (SMALL.factor - 1) * levels + large


def test_fewer_neighbours_of_a_level_than_the_factor_stay_and_as_many_merge() -> None:
    assert merges([Held(5, 0)] * 9) == ()
    assert merges([Held(5, 0)] * 10) == (range(10),)
    assert merges([Held(50_000, 0), *[Held(5, 0)] * 10]) == (range(1, 11),)
    assert merges([Held(500, 0)] * 10, Policy(floor=10, most=4_000)) == ()


@pytest.mark.parametrize(
    "made, refusal",
    [
        (partial(Policy, factor=1), "two segments at least"),
        (partial(Policy, floor=0), "one chunk at least"),
        (partial(Policy, most=10, floor=20), "the floor at least"),
        (partial(Policy, span=-0.1), "span"),
        (partial(Policy, dead_share=0.0), "share"),
        (partial(Policy, dead_share=1.0), "share"),
    ],
)
def test_a_policy_that_could_not_settle_is_refused(made: Callable[[], Policy], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        made()
