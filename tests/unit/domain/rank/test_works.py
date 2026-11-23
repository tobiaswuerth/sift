import pytest

from sift.domain.rank.works import one_per_work, work_of, work_of_fields

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
        (12, None),
        (None, None),
    ],
)
def test_a_work_is_known_by_its_doi_however_it_is_written(doi: object, work: str | None) -> None:
    assert work_of(doi) == work
    assert work_of_fields({"doi": doi} if isinstance(doi, (str, int)) else {}) == work


def test_a_work_is_known_by_the_doi_field_alone() -> None:
    assert work_of_fields({"DOI": "10.1000/abc", "id": "10.1000/abc"}) is None
