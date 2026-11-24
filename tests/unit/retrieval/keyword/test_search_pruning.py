from collections import Counter
from collections.abc import Sequence

from sift.domain.query.expressions import And, Or, Term
from sift.retrieval.keyword.lists import Opened
from sift.retrieval.keyword.search import Tuning
from tests.keyword_searches import brute, collections, same, search
from tests.keyword_stores import Round

FILLER = "filler"
RANGE = 4


def in_ranges(groups: Sequence[Sequence[str]]) -> list[Round]:
    texts = [FILLER] * (RANGE - 1)
    for group in groups:
        texts += [*group, *[FILLER] * (RANGE - len(group))]
    return [{f"n{n:04}.md": ("", "", text)} for n, text in enumerate(texts)]


def blocks_read(store: Opened) -> Counter[str]:
    return Counter(term for _, term, _ in store.blocks)


def test_a_word_in_every_chunk_is_read_only_where_the_others_bring_a_chunk_near() -> None:
    best = ["apple banana stop", "apple apple apple stop"]
    others = [["apple stop filler", "banana stop filler"]] * 8
    with collections([in_ranges([best, *others])]) as (stores, held):
        (store,) = stores
        query = Or((Term("stop"), Term("apple"), Term("banana")))
        same(search(stores, query, 1, tuning=Tuning(first_batch=1)), brute(held, query, 21, by_document=True), 1, query)
        read = blocks_read(store)
        assert read["stop"] < read["apple"] / 2


def test_terms_in_ranges_apart_are_each_read_where_they_occur_and_found_as_scoring_every_chunk_finds() -> None:
    alpha: list[Round] = [{f"n{n:03}.md": ("", f"n{n}", "alpha words " + "alpha " * (n % 3))} for n in range(40)]
    beta: list[Round] = [{f"n{n:03}.md": ("", f"n{n}", "beta words " + "beta " * (n % 3))} for n in range(40, 60)]
    with collections([[*alpha, *beta]]) as (stores, held):
        query = Or((Term("alpha"), Term("beta")))
        same(
            search(stores, query, 100, tuning=Tuning(first_batch=1)),
            brute(held, query, 120, by_document=True),
            100,
            query,
        )


def test_all_of_some_words_reads_each_only_where_chunks_with_all_rarer_ones_are_left() -> None:
    groups = [["alpha beta gamma", "alpha beta delta", "alpha beta"]] * 10
    with collections([in_ranges(groups)]) as (stores, held):
        (store,) = stores
        query = And((Term("alpha"), Term("beta"), Term("gamma"), Term("delta")))
        assert search(stores, query, 10) == ()
        assert brute(held, query, 10, by_document=True) == []
        read = blocks_read(store)
        assert read["gamma"] > 0
        assert read["alpha"] == read["beta"] == 0
