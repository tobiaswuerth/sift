import pytest

from sift.domain.codec.postings import HeldBlock
from sift.domain.codec.shape import SHAPE, shaped
from sift.domain.errors import DamagedError
from sift.domain.query.expressions import NOTHING, And, Not, Or, Phrase, Prefix, Term
from sift.domain.rank.pruned import AND, LEAF, NOT, OR, LeafPlan, NodePlan, ScoringPlan, nodes_of, scored

a, b, c = Term("a"), Phrase(("b", "c")), Prefix("d")


def test_a_query_is_told_to_the_compiled_search_as_its_parts_each_after_those_under_it() -> None:
    told = nodes_of(Or((And((a, b)), Not(c, a))), (a, b, c))
    assert told == [
        NodePlan(LEAF, 0, ()),
        NodePlan(LEAF, 1, ()),
        NodePlan(AND, 0, (0, 1)),
        NodePlan(LEAF, 2, ()),
        NodePlan(LEAF, 0, ()),
        NodePlan(NOT, 0, (3, 4)),
        NodePlan(OR, 0, (2, 5)),
    ]


def test_nothing_is_never_searched_for() -> None:
    with pytest.raises(ValueError, match="nothing is searched for"):
        nodes_of(NOTHING, ())


class Unread:
    def read_blocks(self, wanted: list[tuple[int, int]], /) -> list[HeldBlock]:
        raise AssertionError(wanted)

    def read_lengths(self, wanted: list[tuple[int, int]], /) -> list[bytes]:
        raise AssertionError(wanted)

    def offered(self, places: bytes, scores: bytes, /) -> tuple[float, int]:
        raise AssertionError(places, scores)

    def out_of_time(self) -> bool:
        raise AssertionError


@pytest.mark.parametrize("node", [NodePlan(9, 0, ()), NodePlan(NOT, 0, (0,))], ids=["no kind", "a not of one part"])
def test_a_query_told_with_a_part_of_no_kind_the_compiled_search_knows_is_refused(node: NodePlan) -> None:
    plan = ScoringPlan([], [], [LeafPlan([], [], 1.0, 1.0)], [node], (1.0, shaped(SHAPE), 16, 4096), (0.0, 0))
    with pytest.raises(DamagedError, match="a part of no kind a search knows"):
        scored(plan, Unread())
