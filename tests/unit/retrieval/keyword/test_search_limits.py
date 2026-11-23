from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
from hypothesis import given, settings

from sift.api.errors import InvalidRequestError
from sift.domain.arrays import floats, ints
from sift.domain.codec.postings import encoded, ranged
from sift.domain.codec.segments import Shape
from sift.domain.errors import DamagedError
from sift.domain.query.expressions import And, Or, Phrase, Prefix, Term
from sift.domain.rank.best import Best
from sift.retrieval.keyword.lists import Ranged, opened, stored
from sift.retrieval.keyword.search import Tuning, planned, prepared, query_of, search, searched
from sift.storage.database import rows
from tests.keyword_searches import (
    SEEDS,
    Draws,
    collections,
    prose,
)


class Block(NamedTuple):
    segment: int
    term: str
    number: int
    postings: bytes


if TYPE_CHECKING:
    from tests.keyword_stores import Round


@pytest.mark.parametrize(
    "sql, damage",
    [
        ("DELETE FROM keyword_blocks WHERE ? IS NOT NULL", "a block of postings is missing"),
        ("UPDATE keyword_blocks SET postings = ?", "where its ranges place"),
    ],
)
def test_a_block_missing_or_of_another_size_than_its_ranges_say_is_damage(sql: str, damage: str) -> None:
    three = encoded(np.arange(1, 4, dtype=np.int64), np.ones(3, np.int64), np.zeros(3, np.int64), [0, 3])
    with collections([[prose(Draws(np.random.default_rng(7)), 60)]]) as (stores, _):
        (store,) = stores
        assert any(isinstance(held, Ranged) for held in stored(store, ["w0"])["w0"])
        store.transaction.connection.execute(sql, (three.postings[0],))
        with pytest.raises(DamagedError, match=damage):
            search(stores, Term("w0"), 10)


@pytest.mark.parametrize("place", [0, 1 << 40])
def test_ranges_outside_their_segment_are_damage(place: int) -> None:
    far = ranged(np.array([place], np.int64), np.ones(1, np.int64), np.zeros(1, np.uint8))
    with collections([[prose(Draws(np.random.default_rng(7)), 60)]]) as (stores, _):
        (store,) = stores
        store.transaction.connection.execute("UPDATE keyword_postings SET ranges = ? WHERE ranges IS NOT NULL", (far,))
        with pytest.raises(DamagedError, match="outside its segment"):
            search(stores, Term("w0"), 10)


@settings(max_examples=1)
@given(seed=SEEDS)
def test_a_search_out_of_time_returns_the_best_it_found_scored_exactly(seed: int) -> None:
    rng = Draws(np.random.default_rng(seed))
    with collections([[prose(rng, 300)]]) as (stores, _):
        (store,) = stores
        held = prepared(planned(store, Term("w0")))
        query = query_of(Term("w0"), [held])
        assert query is not None
        exact = searched(held, query, Best(50), Tuning(first_batch=1))
        cut = searched(held, query, Best(50), Tuning(first_batch=1), expired=lambda: True)
        assert (exact.cut, cut.cut) == (False, True)
        assert 0 < len(cut.best.chunks) <= len(exact.best.chunks)
        scores = dict(zip(ints(exact.best.chunks), floats(exact.best.scores), strict=True))
        everything = {each.chunk: each.score for each in search(stores, Term("w0"), 10_000)}
        assert all(
            everything[chunk] == score
            for chunk, score in zip(ints(cut.best.chunks), floats(cut.best.scores), strict=True)
        )
        assert scores == {chunk: everything[chunk] for chunk in scores}


def test_a_prefix_of_words_in_more_passages_than_a_search_reads_is_refused() -> None:
    with collections([[prose(Draws(np.random.default_rng(7)), 40)]]) as (stores, _):
        assert search(stores, Prefix("w1"), 10, tuning=Tuning(most_read=10**6))
        refusal = r"w1\* reads [\d,]+ passages, more than the 5 a search reads"
        with pytest.raises(InvalidRequestError, match=refusal) as refused:
            search(stores, Prefix("w1"), 10, tuning=Tuning(most_read=5))
        assert refused.value.remedy == "write more of the word before the *"


def test_segments_that_overlap_are_damage() -> None:
    with collections([[{"a.md": ("", "A", "w1")}, {"b.md": ("", "B", "w1")}]]) as (stores, _):
        connection = stores[0].transaction.connection
        connection.execute(
            "UPDATE keyword_segments SET last = last + 1 WHERE first = (SELECT min(first) FROM keyword_segments)"
        )
        with pytest.raises(DamagedError, match="overlap"):
            opened(stores[0].transaction)


def test_postings_outside_their_segment_are_damage() -> None:
    with collections([[{"a.md": ("", "A", "w1 w2 " * 3000)}]], Shape()) as (stores, _):
        connection = stores[0].transaction.connection
        connection.execute("UPDATE keyword_segments SET last = first")
        with pytest.raises(DamagedError, match="outside its segments"):
            search([opened(stores[0].transaction)], Term("w1"), 10)


def test_a_search_out_of_time_goes_on_until_it_found_any() -> None:
    apart: Round = {"a0.md": ("", "A", "w1 w1 w1 w1"), "a1.md": ("", "A", "w2 w2 w2 w2")}
    fillers: Round = {f"f{n}.md": ("", "F", "zz") for n in range(6)}
    together: Round = {"b.md": ("", "B", "w1 w2")}
    with collections([[apart, fillers, together]]) as (stores, _):
        (store,) = stores
        both = And((Term("w1"), Term("w2")))
        held = prepared(planned(store, both))
        query = query_of(both, [held])
        assert query is not None
        found = searched(held, query, Best(5), Tuning(first_batch=1), expired=lambda: True)
        assert len(found.best.chunks) == 1


def test_every_store_scores_its_best_ranges_though_time_is_out() -> None:
    weak: Round = {"a.md": ("", "A", "w1 " + "zz " * 50)}
    strong: Round = {"b.md": ("", "B", "w1 w1 w1")}
    with collections([[weak], [strong]]) as (stores, _):
        each = [prepared(planned(store, Term("w1"))) for store in stores]
        query = query_of(Term("w1"), each)
        assert query is not None
        first = searched(each[0], query, Best(1), Tuning(first_batch=1))
        second = searched(each[1], query, first.best, Tuning(first_batch=1), expired=lambda: True)
        assert ints(second.best.chunks) != ints(first.best.chunks)


def test_a_phrase_whose_rarest_word_is_in_more_passages_than_a_search_reads_is_refused() -> None:
    notes: Round = {f"n{n}.md": ("", "N", "of the w1") for n in range(4)}
    with collections([[notes]]) as (stores, _):
        assert len(search(stores, Phrase(("of", "the")), 10, tuning=Tuning(most_read=4))) == 4
        with pytest.raises(InvalidRequestError, match='"of the" reads 4 passages, more than the 3') as refused:
            search(stores, Phrase(("of", "the")), 10, tuning=Tuning(most_read=3))
        assert refused.value.remedy == "add a rarer word to the phrase, or search its words without quotes"


def test_the_prefixes_and_phrases_of_a_query_are_held_to_what_a_search_reads_together() -> None:
    notes: Round = {"a.md": ("", "A", "w1 w20"), "b.md": ("", "B", "w11 w2"), "c.md": ("", "C", "w12")}
    with collections([[notes], [notes]]) as (stores, _):
        either = Or((Prefix("w1"), Prefix("w2")))
        assert search(stores, either, 10, tuning=Tuning(most_read=10))
        refusal = r"w1\* reads 6 passages; with the rest of the query, 10 passages, more than the 9"
        with pytest.raises(InvalidRequestError, match=refusal) as refused:
            search(stores, either, 10, tuning=Tuning(most_read=9))
        assert refused.value.remedy == "write more of the word before the *"


def test_a_prefix_of_words_in_as_many_passages_as_a_search_reads_is_read() -> None:
    with collections([[{"a.md": ("", "A", "w1 w10"), "b.md": ("", "B", "w11")}]]) as (stores, _):
        assert len(search(stores, Prefix("w1"), 10, tuning=Tuning(most_read=3))) == 2
        with pytest.raises(InvalidRequestError, match=r"w1\* reads 3 passages, more than the 2"):
            search(stores, Prefix("w1"), 10, tuning=Tuning(most_read=2))


def test_a_prefix_is_held_to_the_passages_its_words_occur_in_not_to_its_words() -> None:
    with collections([[{"a.md": ("", "A", "w1"), "b.md": ("", "B", "w1"), "c.md": ("", "C", "w1")}]]) as (stores, _):
        assert len(search(stores, Prefix("w1"), 10, tuning=Tuning(most_read=3))) == 3
        with pytest.raises(InvalidRequestError, match=r"w1\* reads 3 passages, more than the 2"):
            search(stores, Prefix("w1"), 10, tuning=Tuning(most_read=2))


def test_a_block_whose_postings_lie_outside_its_ranges_is_damage() -> None:
    with collections([[prose(Draws(np.random.default_rng(7)), 60)]]) as (stores, _):
        (store,) = stores
        sql = "SELECT segment, term, number, postings FROM keyword_blocks WHERE term = 'w0' ORDER BY segment, number"
        first, second, *_ = rows(store.transaction, Block, sql)
        store.transaction.connection.execute(
            "UPDATE keyword_blocks SET postings = ? WHERE segment = ? AND term = ? AND number = ?",
            (second.postings, first.segment, first.term, first.number),
        )
        with pytest.raises(DamagedError, match="outside its ranges"):
            search(stores, Term("w0"), 10_000)
