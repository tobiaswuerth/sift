import pytest

from sift.domain.rank.works import MOST_DOI, one_per_work, work_of, work_of_fields

type Ranked = tuple[str, str | None]


def work(each: Ranked) -> str | None:
    return each[1]


def copied(best: Ranked, copy: Ranked) -> Ranked:
    return f"{best[0]}+{copy[0]}", best[1]


def test_the_best_ranked_copy_of_a_work_stays_and_names_the_others() -> None:
    ranked: list[Ranked] = [("a", "w1"), ("b", None), ("c", "w1"), ("d", "w2"), ("e", None), ("f", "w1")]
    assert one_per_work(ranked, work, copied) == [("a+c+f", "w1"), ("b", None), ("d", "w2"), ("e", None)]
    assert one_per_work([], work, copied) == []


@pytest.mark.parametrize(
    "doi, work",
    [
        ("10.1000/ABC", "10.1000/abc"),
        ("https://doi.org/10.1000/abc", "10.1000/abc"),
        ("http://dx.doi.org/10.1000/abc", "10.1000/abc"),
        ("doi: 10.1000/abc ", "10.1000/abc"),
        ("arXiv:2101.00001", None),
        ("10.\N{ARABIC-INDIC DIGIT ONE}\N{ARABIC-INDIC DIGIT ZERO}/abc", None),
        (12, None),
        (None, None),
    ],
)
def test_a_work_is_known_by_its_doi_however_it_is_written(doi: object, work: str | None) -> None:
    assert work_of(doi) == work
    assert work_of_fields({"doi": doi} if isinstance(doi, (str, int)) else {}) == work


def test_a_doi_is_looked_for_in_the_first_characters_of_a_field_alone() -> None:
    assert work_of("10." * 100_000) is None
    assert work_of(" " * (MOST_DOI - 11) + "10.1000/abc") == "10.1000/abc"
    assert work_of(" " * MOST_DOI + "10.1000/abc") is None


def test_a_work_is_known_by_the_doi_field_alone() -> None:
    assert work_of_fields({"DOI": "10.1000/abc", "id": "10.1000/abc"}) is None


@pytest.mark.parametrize(
    "doi",
    [
        "doi: https://doi.org/10.1000/X",
        "doi.org/10.1000/x",
        "https://doi.org/10.1000%2Fx",
        "10.1000/x.",
        "(10.1000/X)",
        "https://dx.doi.org/10.1000/x;",
    ],
)
def test_a_doi_is_found_however_it_is_prefixed_quoted_or_ended(doi: str) -> None:
    assert work_of(doi) == "10.1000/x"
