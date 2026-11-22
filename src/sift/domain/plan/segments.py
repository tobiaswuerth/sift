import math
from collections.abc import Sequence
from dataclasses import dataclass

FEWEST_JOINED = 2


@dataclass(frozen=True, slots=True)
class Held:
    chunks: int
    dead: int


@dataclass(frozen=True, slots=True)
class Policy:
    factor: int = 10
    floor: int = 1_000
    span: float = 0.75
    most: int = 2_000_000
    dead_share: float = 1 / 3

    def __post_init__(self) -> None:
        if self.factor < FEWEST_JOINED:
            raise ValueError("a merge joins two segments at least")
        if self.floor < 1:
            raise ValueError("the floor of a level is one chunk at least")
        if self.most < self.floor:
            raise ValueError("the most a merge writes is the floor at least")
        if self.span < 0:
            raise ValueError("a level's span is no less than zero")
        if not 0 < self.dead_share < 1:
            raise ValueError("the share of dead chunks that rewrites a segment lies between 0 and 1")


POLICY = Policy()


def _level(live: int, policy: Policy) -> float:
    return math.log(max(live, policy.floor), policy.factor)


def _by_level(live: Sequence[int], policy: Policy) -> list[range]:
    levels = [_level(n, policy) for n in live]
    found: list[range] = []
    start = 0
    while start < len(live):
        bottom = max(max(levels[start:]) - policy.span, _level(policy.floor, policy))
        upto = max(at for at in range(start, len(live)) if levels[at] >= bottom)
        while start + policy.factor <= upto + 1:
            run = range(start, start + policy.factor)
            if sum(live[at] for at in run) <= policy.most:
                found.append(run)
                start = run.stop
            else:
                start += 1
        start = upto + 1
    return found


def merges(held: Sequence[Held], policy: Policy = POLICY) -> tuple[range, ...]:
    found = _by_level([each.chunks - each.dead for each in held], policy)
    covered = {at for run in found for at in run}
    rewritten = [
        range(at, at + 1)
        for at, each in enumerate(held)
        if at not in covered and each.dead > policy.dead_share * each.chunks
    ]
    return tuple(sorted([*found, *rewritten], key=lambda run: run.start))
