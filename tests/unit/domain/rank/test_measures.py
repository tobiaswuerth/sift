import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.rank.measures import MOST_GRADE, Judged, found, ndcg, recall, reciprocal_rank

IDS = st.sampled_from([f"notes/{n}.md" for n in range(15)])
GRADES = st.dictionaries(IDS, st.integers(1, 3), min_size=1)
RANKED = st.lists(IDS, unique=True, max_size=15)


def test_ndcg_rewards_the_better_documents_first() -> None:
    assert ndcg(["a", "b"], {"b": 1}) == pytest.approx(1 / math.log2(3))
    assert (ndcg(["b", "a"], {"a": 2, "b": 1}) < 1, ndcg(["a", "b"], {"a": 2, "b": 1})) == (True, 1.0)
    assert ndcg([], {"b": 1}) == 0


def test_ndcg_takes_the_highest_grade_there_is_and_grades_between_the_whole_ones() -> None:
    assert ndcg(["a", "b"], {"a": MOST_GRADE, "b": MOST_GRADE}) == 1.0
    assert 0 < ndcg(["b", "a"], {"a": MOST_GRADE, "b": 0.5}) < 1


def test_the_reciprocal_rank_is_that_of_the_first_relevant_document_among_the_first_ten() -> None:
    assert (reciprocal_rank(["a", "b"], {"b": 1}), reciprocal_rank(["a"] * 10 + ["b"], {"b": 1})) == (0.5, 0.0)


def test_recall_is_the_share_of_the_relevant_documents_found_among_the_first_ten() -> None:
    assert (recall(["a", "b", "c"], {"b": 1, "x": 1}), recall(["a"] * 10 + ["b"], {"b": 1})) == (0.5, 0.0)


@given(RANKED, GRADES)
def test_each_measure_lies_between_none_found_and_the_best_order(ranked: list[str], relevant: dict[str, int]) -> None:
    ideal = sorted(relevant, key=lambda each: -relevant[each])
    for measure in (ndcg, reciprocal_rank, recall):
        assert 0.0 <= measure(ranked, relevant) <= measure(ideal, relevant) + 1e-12
    assert (ndcg(ideal, relevant), reciprocal_rank(ideal, relevant)) == (pytest.approx(1.0), 1.0)


def test_a_copy_of_the_paper_sought_counts_as_it_found() -> None:
    hits = [("pubmed/x.md", "arxiv/x.md"), ("arxiv/other.md",)]
    assert found(hits, {"arxiv/x.md": 1}) == Judged(("arxiv/x.md", "arxiv/other.md"), {"arxiv/x.md": 1})


def test_copies_a_case_grades_are_one_paper_found_with_its_best_grade() -> None:
    hits = [("pubmed/x.md", "arxiv/x.md"), ("arxiv/other.md",)]
    said = found(hits, {"arxiv/x.md": 1, "pubmed/x.md": 2})
    assert said == Judged(("pubmed/x.md", "arxiv/other.md"), {"pubmed/x.md": 2})
    assert (ndcg(said.ranked, said.relevant), recall(said.ranked, said.relevant)) == (1.0, 1.0)
