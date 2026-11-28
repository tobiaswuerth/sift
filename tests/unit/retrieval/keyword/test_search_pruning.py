import cProfile
import pstats
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np

import sift.domain.codec.postings as postings_module
import sift.domain.rank.pruned as pruned_module
import sift.retrieval.keyword.lists as lists_module
import sift.retrieval.keyword.scoring as scoring_module
import sift.storage.keyword_lists as keyword_lists_module
from sift.domain.codec.postings import kept_bytes
from sift.domain.codec.shape import Shape
from sift.domain.query.expressions import And, Leaf, Or, Phrase, Term
from sift.retrieval.keyword.lists import BlockKey, DecodedLists, Opened, opened
from sift.retrieval.keyword.search import FoundChunk, Tuning
from tests.keyword_searches import Chunks, Draws, blocks_read, brute, collections, prose, same, search
from tests.keyword_stores import Round

if TYPE_CHECKING:
    from sift.domain.codec.postings import HeldBlock, HeldTable

FILLER = "filler"
W1 = Term("w1")
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


def finding(found: list[tuple[FoundChunk, ...]], stores: list[Opened], query: Leaf | Or, k: int) -> Callable[[], None]:
    return lambda: found.append(search(stores, query, k))


def test_a_word_s_postings_in_many_segments_get_their_lengths_in_one_ask_a_batch_of_ranges() -> None:
    rng = Draws(np.random.default_rng(3))
    rounds = [prose(rng, 30, f"r{n}_") for n in range(8)]
    with collections([rounds]) as (stores, held):
        (store,) = stores
        assert len(store.segments) == len(rounds)
        query = Or((Term("w0"), Term("w1"), Term("w2")))
        found: list[tuple[FoundChunk, ...]] = []
        asked = [(scoring_module, "read_blocks"), (scoring_module, "read_lengths"), (keyword_lists_module, "lengths")]
        stages, looked_up, statements = calls(finding(found, stores, query, 5), asked)
        same(found[0], brute(held, query, 20, by_document=True), 5, query)
        assert 0 < looked_up <= stages + 1 < statements


@dataclass(eq=False, slots=True)
class Recording[K, V]:
    held: dict[K, V] = field(default_factory=dict[K, V])
    asked: list[K] = field(default_factory=list[K])

    def get(self, key: K, /) -> V | None:
        self.asked.append(key)
        return self.held.get(key)

    def __setitem__(self, key: K, value: V, /) -> None:
        self.held[key] = value


def test_a_store_is_scored_by_one_call_that_asks_python_for_each_block_once_and_each_segment_s_lengths_once() -> None:
    rng = Draws(np.random.default_rng(3))
    with collections([[prose(rng, 30, f"r{n}_") for n in range(8)], [prose(rng, 40, "o")]]) as (stores, held):
        kept: list[tuple[Recording[int, HeldTable], Recording[BlockKey, HeldBlock]]]
        kept = [(Recording(), Recording()) for _ in stores]
        recorded = [opened(store.transaction, DecodedLists(*each)) for store, each in zip(stores, kept, strict=True)]
        query = Or((Term("w0"), Term("w1"), Term("w2")))
        found: list[tuple[FoundChunk, ...]] = []
        (scorings,) = calls(finding(found, recorded, query, 5), [(pruned_module, "scored")])
        same(found[0], brute(held, query, 20, by_document=True), 5, query)
    blocks = [asked.asked for _, asked in kept]
    lengths = [sorted(asked.asked) for asked, _ in kept]
    assert scorings == len(stores)
    assert all(len(keys) == len(set(keys)) > 0 for keys in blocks), blocks
    assert lengths == [sorted(segment.id for segment in store.segments) for store in recorded]


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


def calls_finding_half(
    stores: list[Opened], held: list[Chunks], named: Sequence[tuple[ModuleType, str]], query: Leaf = W1
) -> list[int]:
    found: list[tuple[FoundChunk, ...]] = []
    counted = calls(lambda: found.append(search(stores, query, 300, by_document=False)), named)
    same(found[0], brute(held, query, 600, by_document=False), 300, query)
    return counted


def test_the_blocks_of_a_list_that_a_batch_of_ranges_needs_are_read_by_one_statement_not_one_each() -> None:
    notes: Round = {f"n{n:04}.md": ("", "N", f"w1 x{n}") for n in range(600)}
    with collections([[notes]]) as (stores, held):
        (store,) = stores
        asked = [(scoring_module, "read_blocks"), (keyword_lists_module, "blocks")]
        stages, statements = calls_finding_half(stores, held, asked)
        assert len(blocks_read(store)) > 10 * stages
        assert statements <= stages


def test_the_blocks_a_statement_reads_are_unpacked_by_one_call_not_one_each() -> None:
    notes: Round = {f"n{n:04}.md": ("", "N", f"w1 x{n}") for n in range(600)}
    with collections([[notes]]) as (stores, held):
        (store,) = stores
        asked = [(keyword_lists_module, "blocks"), (postings_module, "blocked_within")]
        statements, unpacked = calls_finding_half(stores, held, asked)
        assert len(blocks_read(store)) > 10 * statements
        assert unpacked == statements


def test_the_lengths_a_search_reads_are_read_by_one_statement_a_stage_not_one_for_each_block() -> None:
    notes: Round = {f"n{n:04}.md": ("", "N", f"w1 x{n}") for n in range(600)}
    with collections([[notes]]) as (stores, held):
        (store,) = stores
        asked = [(scoring_module, "read_lengths"), (keyword_lists_module, "lengths")]
        stages, statements = calls_finding_half(stores, held, asked)
        tables = cast("dict[int, HeldTable]", store.decoded.lengths).values()
        blocks = sum(kept_bytes(table) for table in tables) // (4 << store.shape.lengths_shift)
        assert blocks > 5 * statements
        assert statements == stages


def test_the_positions_a_phrase_reads_are_unpacked_by_one_call_a_list_not_one_for_each_block() -> None:
    notes: Round = {f"n{n:04}.md": ("", "N", f"w1 w2 x{n}") for n in range(600)}
    with collections([[notes]]) as (stores, held):
        (store,) = stores
        asked = [(lists_module, "positions_at"), (postings_module, "occurrences")]
        lists_read, unpacked = calls_finding_half(stores, held, asked, Phrase(("w1", "w2")))
        assert len(blocks_read(store)) > 100 * lists_read
        assert unpacked == lists_read


def rows_made(run: Callable[[], object]) -> int:
    profile = cProfile.Profile()
    _ = profile.runcall(run)
    counted = pstats.Stats(profile).get_stats_profile().func_profiles
    return sum(int(each.ncalls.split("/")[0]) for name, each in counted.items() if name == "_make")


def test_the_places_offered_to_the_best_are_read_as_columns_not_a_row_object_each() -> None:
    notes: Round = {f"n{n}.md": ("", "N", f"w1 x{n}") for n in range(1000)}
    with collections([[notes]], Shape()) as (stores, _):
        found: list[tuple[FoundChunk, ...]] = []
        made = rows_made(lambda: found.append(search(stores, Term("w1"), 500, by_document=False)))
        assert (len(found[0]), made < 50) == (500, True)
