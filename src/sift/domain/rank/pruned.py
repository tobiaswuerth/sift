from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from sift import _native
from sift.domain.arrays import Ids
from sift.domain.codec.postings import HeldRanges, HeldTable
from sift.domain.errors import DamagedError
from sift.domain.query.expressions import And, Expression, Leaf, Not, Or, Phrase, Prefix, Term

UNSEARCHED = "nothing is searched for: a query that matches nothing is answered before it gets here"
LEAF, NOT, OR, AND = 0, 1, 2, 3

type ScoringReads = _native.Reads
type NativePlan = _native.Plan


@dataclass(eq=False, frozen=True, slots=True)
class SegmentPlan:
    first: int
    last: int
    dead: Ids
    lengths: HeldTable | None


class PartPlan(NamedTuple):
    segment: int
    ranges: HeldRanges
    slack: float


@dataclass(eq=False, frozen=True, slots=True)
class ExactPlan:
    places: Ids
    counts: Ids
    segment: int | None


class LeafPlan(NamedTuple):
    exact: Sequence[ExactPlan]
    parts: Sequence[int]
    bounding: float
    weight: float


class NodePlan(NamedTuple):
    kind: int
    leaf: int
    parts: Sequence[int]


class ScoringPlan(NamedTuple):
    segments: Sequence[SegmentPlan]
    parts: Sequence[PartPlan]
    leaves: Sequence[LeafPlan]
    nodes: Sequence[NodePlan]
    numbers: tuple[float, tuple[int, int, int, int], int, int]
    reached: tuple[float, int]


def nodes_of(expression: Expression, leaves: Sequence[Leaf], before: int = 0) -> list[NodePlan]:
    match expression:
        case Term() | Phrase() | Prefix():
            return [NodePlan(LEAF, leaves.index(expression), ())]
        case Not(kept, dropped):
            first = nodes_of(kept, leaves, before)
            second = nodes_of(dropped, leaves, before + len(first))
            return [*first, *second, NodePlan(NOT, 0, (before + len(first) - 1, before + len(first) + len(second) - 1))]
        case And(parts) | Or(parts):
            found: list[NodePlan] = []
            roots: list[int] = []
            for part in parts:
                found += nodes_of(part, leaves, before + len(found))
                roots.append(before + len(found) - 1)
            return [*found, NodePlan(AND if isinstance(expression, And) else OR, 0, tuple(roots))]
        case _:
            raise ValueError(UNSEARCHED)


def _told(plan: ScoringPlan) -> NativePlan:
    segments = [(each.first, each.last, each.dead, each.lengths) for each in plan.segments]
    exact = [[(each.places, each.counts, each.segment) for each in leaf.exact] for leaf in plan.leaves]
    leaves = [(held, leaf.parts, leaf.bounding, leaf.weight) for held, leaf in zip(exact, plan.leaves, strict=True)]
    return segments, plan.parts, leaves, plan.nodes, plan.numbers, plan.reached


def scored(plan: ScoringPlan, reads: ScoringReads) -> tuple[bool, list[HeldTable]]:
    try:
        return _native.searched(_told(plan), reads)
    except DamagedError:
        raise
    except ValueError as error:
        raise DamagedError(str(error)) from None
