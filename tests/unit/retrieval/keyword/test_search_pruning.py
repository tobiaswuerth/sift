import cProfile
import pstats
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np

import sift.retrieval.keyword.lists as lists_module
import sift.retrieval.keyword.search as search_module
from sift.domain.query.expressions import And, Or, Term
from sift.retrieval.keyword.lists import Opened
from sift.retrieval.keyword.search import FoundChunk, Tuning
from tests.keyword_searches import Draws, blocks_read, brute, collections, prose, same, search
from tests.keyword_stores import Round

FILLER = "filler"
RANGE = 4


def in_ranges(groups: Sequence[Sequence[str]]) -> list[Round]:
    texts = [FILLER] * (RANGE - 1)
    for group in groups:
        texts += [*group, *[FILLER] * (RANGE - len(group))]
    return [{f"n{n:04}.md": ("", "", text)} for n, text in enumerate(texts)]


def terms_read(store: Opened) -> Counter[str]:
    return Counter(term for _, term, _ in blocks_read(store))


def calls(run: Callable[[], object], named: Sequence[tuple[ModuleType, str]]) -> list[int]:
    profile = cProfile.Profile()
    _ = profile.runcall(run)
    counted = pstats.Stats(profile).get_stats_profile().func_profiles
    found = [counted[name] for _, name in named]
    assert [Path(each.file_name) for each in found] == [Path(str(module.__file__)) for module, _ in named]
    return [int(each.ncalls.split("/")[0]) for each in found]


def test_a_word_s_postings_in_many_segments_get_their_lengths_once_a_batch_of_ranges() -> None:
    rng = Draws(np.random.default_rng(3))
    rounds = [prose(rng, 30, f"r{n}_") for n in range(8)]
    with collections([rounds]) as (stores, held):
        (store,) = stores
        assert len(store.segments) == len(rounds)
        query = Or((Term("w0"), Term("w1"), Term("w2")))
        found: list[tuple[FoundChunk, ...]] = []
        scored = [(search_module, "_scored_group"), (lists_module, "lengths_at")]
        batches, looked_up = calls(lambda: found.append(search(stores, query, 5)), scored)
        same(found[0], brute(held, query, 20, by_document=True), 5, query)
        assert 0 < looked_up <= (batches + 1) * len(query.parts)


def test_a_word_in_every_chunk_is_read_only_where_the_others_bring_a_chunk_near() -> None:
    best = ["apple banana stop", "apple apple apple stop"]
    others = [["apple stop filler", "banana stop filler"]] * 8
    with collections([in_ranges([best, *others])]) as (stores, held):
        (store,) = stores
        query = Or((Term("stop"), Term("apple"), Term("banana")))
        same(search(stores, query, 1, tuning=Tuning(first_batch=1)), brute(held, query, 21, by_document=True), 1, query)
        read = terms_read(store)
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
        read = terms_read(store)
        assert read["gamma"] > 0
        assert read["alpha"] == read["beta"] == 0
