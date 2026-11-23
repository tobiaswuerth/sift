import unicodedata

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.rank.hybrid import blend, named, passage, standing_out, words, work_of


def test_words_are_lowercase_without_stopwords_or_single_latin_letters() -> None:
    assert words("How do Transformers handle it's long-input sequences? À 5 中 _") == [
        "transformers",
        "handle",
        "long",
        "input",
        "sequences",
        "5",
        "中",
    ]


def test_front_matter_values_sharing_a_word_with_the_query_are_named() -> None:
    asked = frozenset(words("papers by Ada Lovelace"))
    assert named(["Ada Lovelace", "Charles Babbage", "Analytical Engines"], asked) == ["Ada Lovelace"]


def test_the_reranker_reads_the_title_the_values_named_and_the_text() -> None:
    assert passage("Notes", ["Ada Lovelace"], "Text.") == "Notes\nAda Lovelace\n\nText."
    assert passage("Notes", [], "Text.") == "Notes\n\nText."


def test_the_cosine_counts_in_units_of_the_reranker_s_spread_and_names_lift_it_to_the_best() -> None:
    rerank = np.array([3.0, 1.0, 2.0])
    cosine = np.array([0.5, 0.9, 0.7])
    found = blend(rerank, cosine, np.array([False, False, False]))
    spread = rerank.std() / cosine.std()
    assert np.allclose(found, rerank + 0.5 * spread * (cosine - cosine.mean()))
    lifted = blend(rerank, cosine, np.array([True, False, False]))
    assert lifted[0] > found[0]


def test_cosines_all_alike_leave_the_reranker_s_order() -> None:
    rerank = np.array([3.0, 1.0])
    assert np.array_equal(blend(rerank, np.array([0.4, 0.4]), np.array([False, False])), rerank)
    assert len(blend(np.zeros(0), np.zeros(0), np.zeros(0, np.bool_))) == 0


@pytest.mark.parametrize(
    "scores, kept",
    [
        ([9.0, 8.9, 8.8, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2], 8),
        ([5.0, 5.0, 5.0, 5.0], 4),
        ([3.0, 1.0], 2),
        ([3.0, 2.0, 1.0], 3),
    ],
)
def test_the_results_standing_out_end_five_after_the_largest_drop(scores: list[float], kept: int) -> None:
    assert standing_out(scores) == kept


@given(st.lists(st.floats(-100, 100), max_size=40).map(lambda found: sorted(found, reverse=True)))
def test_the_cut_keeps_one_result_at_least_and_never_more_than_there_are(scores: list[float]) -> None:
    assert min(1, len(scores)) <= standing_out(scores) <= len(scores)


@pytest.mark.parametrize(
    "doi, work",
    [
        ("10.1000/ABC", "10.1000/abc"),
        ("https://doi.org/10.1000/abc", "10.1000/abc"),
        ("http://dx.doi.org/10.1000/abc", "10.1000/abc"),
        ("doi: 10.1000/abc ", "10.1000/abc"),
        ("arXiv:2101.00001", None),
        (12, None),
        (None, None),
    ],
)
def test_a_work_is_known_by_its_doi_however_it_is_written(doi: object, work: str | None) -> None:
    assert work_of(doi) == work


def test_a_cosine_not_known_counts_as_the_mean_of_the_others_neither_bonus_nor_penalty() -> None:
    rerank = np.array([3.0, 2.0, 1.0, 2.5])
    names = np.zeros(4, np.bool_)
    unknown = blend(rerank, np.array([0.9, np.nan, 0.3, np.nan]), names)
    assert np.allclose(unknown, blend(rerank, np.array([0.9, 0.6, 0.3, 0.6]), names))
    assert unknown[1] == pytest.approx(2.0)
    assert np.allclose(blend(rerank, np.full(4, np.nan), names), rerank)


def test_words_are_those_the_keyword_index_holds_however_the_text_is_composed() -> None:
    decomposed = unicodedata.normalize("NFD", "Résumé of Zürich, İstanbul and snake_case")
    assert words(decomposed) == ["resume", "zurich", "istanbul", "snake", "case"]
    assert named(["Zürich"], frozenset(words(decomposed))) == ["Zürich"]
