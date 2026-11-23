import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sift.domain.arrays import ints, last
from sift.domain.codec.segments import Shape
from sift.domain.plan.segments import Policy
from sift.domain.query.expressions import NOTHING, And, Or, Term
from sift.domain.query.grammar import parsed
from sift.domain.rank.best import Best
from sift.retrieval.keyword.lists import Ranged, stored
from sift.retrieval.keyword.search import Tuning, found_in, planned, prepared, query_of, search, searched
from sift.storage.database import BATCHES, Batches
from tests.allowed import Chosen
from tests.keyword_searches import (
    MERGING,
    SEEDS,
    Draws,
    brute,
    collections,
    fts5_query,
    fts5_ranked,
    phrases_in,
    prose,
    random_expression,
    same,
)
from tests.keyword_stores import SMALL, Content, Round


def test_passages_of_equal_score_keep_the_order_of_their_chunks_however_the_search_is_batched() -> None:
    notes: Round = {f"s{n:03}.md": ("", "T", "alpha beta gamma") for n in range(30)}
    notes |= {f"f{n:03}.md": ("", "U", "alpha zeta eta") for n in range(12)}
    notes |= {f"g{n:03}.md": ("", "V", "beta theta") for n in range(2)}
    notes |= {f"h{n:03}.md": ("", "W", "omega theta") for n in range(60)}
    words = ["alpha", "beta", "gamma"]
    with collections([[notes]], SMALL) as (stores, _):
        for first_batch in (1, 2, 4096):
            for order in itertools.permutations(words):
                found = search(stores, Or(tuple(map(Term, order))), 10, tuning=Tuning(first_batch=first_batch))
                assert len({each.score for each in found}) == 1
                assert [each.chunk & 0xFF_FFFF_FFFF for each in found] == list(range(1, 11))


@settings(max_examples=6)
@given(seed=SEEDS, shape=st.sampled_from([SMALL, Shape()]), batches=st.sampled_from([BATCHES, MERGING]))
def test_finds_what_scoring_every_chunk_finds(seed: int, shape: Shape, batches: Batches) -> None:
    rng = Draws(np.random.default_rng(seed))
    gone: Round = {f"n{n:03}.md": None for n in range(0, 60, 7)}
    rounds = [prose(rng, 60), prose(rng, 50, "m"), gone, prose(rng, 8)]
    with collections([rounds], shape, batches) as (stores, held):
        phrases = phrases_in(held)
        for _ in range(30):
            expression, k = random_expression(rng, phrases), rng.choice([1, 3, 10, 50])
            by_document = rng.random() < 0.7
            tuning = Tuning(first_batch=rng.choice([1, 2, 16]))
            got = search(stores, expression, k, by_document=by_document, tuning=tuning)
            same(got, brute(held, expression, k + 20, by_document=by_document), k, expression)


@settings(max_examples=1)
@given(seed=SEEDS)
def test_across_collections_with_their_statistics_as_one(seed: int) -> None:
    rng = Draws(np.random.default_rng(seed))
    with collections([[prose(rng, 50)], [prose(rng, 30), prose(rng, 5, "x")]]) as (stores, held):
        phrases = phrases_in(held)
        for _ in range(25):
            expression, k = random_expression(rng, phrases), rng.choice([1, 10, 30])
            same(search(stores, expression, k), brute(held, expression, k + 20, by_document=True), k, expression)


@settings(max_examples=3)
@given(seed=SEEDS)
def test_nested_queries_score_as_fts5_scores_them(seed: int) -> None:
    rng = Draws(np.random.default_rng(seed))
    with collections([[prose(rng, 60)]]) as (stores, held):
        phrases = phrases_in(held)
        for _ in range(40):
            expression = random_expression(rng, phrases)
            want = fts5_ranked(held[0], fts5_query(expression), 30)
            same(search(stores, expression, 10), want, min(10, len(want)), fts5_query(expression))


def test_a_list_long_enough_for_ranges_in_the_default_shape_finds_what_scoring_every_chunk_finds() -> None:
    notes: Round = {f"n{n:05}.md": ("", "N", f"w1 w{2 + n % 7}") for n in range(5000)}
    with collections([[notes]], Shape()) as (stores, held):
        (store,) = stores
        assert any(isinstance(each, Ranged) for each in stored(store, ["w1"])["w1"])
        for expression in (Term("w1"), Or((Term("w1"), Term("w3"))), And((Term("w1"), Term("w4")))):
            same(search(stores, expression, 10), brute(held, expression, 40, by_document=True), 10, expression)


@pytest.mark.parametrize(
    "query",
    ["w1", "w0", "w3 OR w5 OR muller", "w1 w2", '"w0 w1"', "w1*", "w2* w0", "w0 NOT w1", '"w0 w0" OR w7', "mul*"],
)
@settings(max_examples=1)
@given(seed=SEEDS)
def test_finds_what_fts5_finds(query: str, seed: int) -> None:
    rng = Draws(np.random.default_rng(seed))
    with collections([[prose(rng, 80), {f"n{n:03}.md": None for n in range(0, 80, 5)}]]) as (stores, (held,)):
        expression = parsed(query)
        assert expression != NOTHING
        for k in (1, 5, 40):
            same(search(stores, expression, k), fts5_ranked(held, query, k + 20), k, query)


def test_nothing_is_found_in_collections_without_chunks() -> None:
    with collections([[{}]]) as (stores, _):
        assert search(stores, Term("w1"), 10) == ()


@settings(max_examples=1)
@given(seed=SEEDS)
def test_a_term_in_no_chunk_finds_nothing_and_leaves_an_or_as_it_was(seed: int) -> None:
    rng = Draws(np.random.default_rng(seed))
    with collections([[prose(rng, 40)]]) as (stores, _):
        assert search(stores, Term("absent"), 10) == ()
        assert search(stores, And((Term("w1"), Term("absent"))), 10) == ()
        assert search(stores, Or((Term("w1"), Term("absent"))), 3) == search(stores, Term("w1"), 3)


@settings(max_examples=1)
@given(seed=SEEDS)
def test_most_blocks_of_a_common_term_are_never_read(seed: int) -> None:
    rng = Draws(np.random.default_rng(seed))
    with collections([[prose(rng, 300)]]) as (stores, _):
        (store,) = stores
        search(stores, Term("w0"), 1, tuning=Tuning(first_batch=2))
        read = len(store.blocks)
        lists = [held for held in stored(store, ["w0"])["w0"] if isinstance(held, Ranged)]
        total = sum(-(-last(held.ranges.ends) // store.shape.block) for held in lists)
        assert 0 < read < total / 3


@pytest.mark.parametrize("by_document", [True, False])
def test_passages_of_the_same_score_are_ranked_by_their_order(*, by_document: bool) -> None:
    same_note: Content = ("", "Same", "w1 w2 w3")
    rounds: list[list[Round]] = [
        [{f"n{n}.md": same_note for n in range(4)}],
        [{f"m{n}.md": same_note for n in range(3)}],
    ]
    with collections(rounds) as (stores, held):
        got = search(stores, Term("w1"), 5, by_document=by_document)
        every = sorted(chunk for store in held for chunk in store)
        assert [each.chunk for each in got] == every[:5]
        assert len({each.score for each in got}) == 1


def test_a_segment_rewritten_after_a_later_one_is_still_found_in_chunk_order() -> None:
    first: Round = {f"a{n}.md": ("", "A", "w1 w2 " * (n + 1)) for n in range(3)}
    later: Round = {f"b{n}.md": ("", "B", "w1 w3 " * (n + 1)) for n in range(3)}
    gone: Round = {"a0.md": None, "a1.md": None}
    with collections([[first, later, gone]], SMALL, Batches(merging=Policy(dead_share=0.3))) as (stores, held):
        (store,) = stores
        assert [segment.first for segment in store.segments] == sorted(segment.first for segment in store.segments)
        assert [segment.id for segment in store.segments] != sorted(segment.id for segment in store.segments)
        for expression in (Term("w1"), Term("w2"), Term("w3")):
            same(search(stores, expression, 10), brute(held, expression, 30, by_document=True), 10, expression)


@settings(max_examples=3)
@given(seed=SEEDS)
def test_only_the_chunks_allowed_are_found_as_many_as_asked_for(seed: int) -> None:
    rng = Draws(np.random.default_rng(seed))
    with collections([[prose(rng, 50)], [prose(rng, 30)]]) as (stores, held):
        phrases = phrases_in(held)
        for _ in range(20):
            kept = {document for store in held for _, (document, _) in store.items() if rng.random() < 0.3}
            allowed = [np.array(sorted(c for c, (d, _) in store.items() if d in kept), np.int64) for store in held]
            expression, k = random_expression(rng, phrases), rng.choice([1, 3, 10])
            by_document = rng.random() < 0.7
            owners = np.array(sorted(kept), np.int64)
            each = [
                prepared(planned(store, expression), allowed=Chosen(chunks, owners))
                for store, chunks in zip(stores, allowed, strict=True)
            ]
            query = query_of(expression, each)
            assert query is not None
            best = Best(k, by_document)
            for held_store in each:
                best = searched(held_store, query, best).best
            got = found_in(best)
            every = brute(held, expression, 10**6, by_document=by_document)
            chosen = {chunk for each in allowed for chunk in ints(each)}
            same(got, [(chunk, score) for chunk, score in every if chunk in chosen], k, expression)
