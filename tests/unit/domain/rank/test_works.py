from sift.domain.rank.works import one_per_work

type Ranked = tuple[str, str | None]


def work(each: Ranked) -> str | None:
    return each[1]


def copied(best: Ranked, copy: Ranked) -> Ranked:
    return f"{best[0]}+{copy[0]}", best[1]


def test_the_best_ranked_copy_of_a_work_stays_and_names_the_others() -> None:
    ranked: list[Ranked] = [("a", "w1"), ("b", None), ("c", "w1"), ("d", "w2"), ("e", None), ("f", "w1")]
    assert one_per_work(ranked, work, copied) == [("a+c+f", "w1"), ("b", None), ("d", "w2"), ("e", None)]
    assert one_per_work([], work, copied) == []
