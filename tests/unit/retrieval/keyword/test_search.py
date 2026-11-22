import contextlib
import math
import sqlite3
import tempfile
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sift.domain.arrays import last
from sift.domain.codec.postings import encoded, ranged
from sift.domain.codec.segments import Shape
from sift.domain.errors import DamagedError
from sift.domain.query.expressions import NOTHING, And, Expression, Leaf, Not, Or, Phrase, Prefix, Term
from sift.domain.query.grammar import parsed
from sift.retrieval.keyword.lists import Opened, Ranged, opened, stored
from sift.retrieval.keyword.search import Found, Tuning, search
from sift.storage.database import Connection, reading
from tests.keyword_stores import COLUMNS, SMALL, Content, Round, applied, instances, opened_store, texts

VOCABULARY = [f"w{n}" for n in range(30)] + ["Müller"]
SEEDS = st.integers(0, 2**32 - 1)
TERMS = [f"w{n}" for n in range(30)] + ["muller", "absent"]
FTS5_K1, FTS5_B = 1.2, 0.75
type Chunks = dict[int, tuple[int, list[list[str]]]]


@dataclass(frozen=True, slots=True)
class Draws:
    rng: np.random.Generator

    def random(self) -> float:
        return float(self.rng.random())

    def choice[T](self, items: Sequence[T]) -> T:
        return items[int(self.rng.integers(len(items)))]

    def randint(self, low: int, high: int) -> int:
        return int(self.rng.integers(low, high + 1))

    def paretovariate(self, alpha: float) -> float:
        return float(self.rng.pareto(alpha)) + 1


def prose(rng: Draws, notes: int, name: str = "n") -> Round:
    def words(n: int) -> str:
        return " ".join(VOCABULARY[min(int(rng.paretovariate(0.8)) - 1, len(VOCABULARY) - 1)] for _ in range(n))

    def content() -> Content:
        fields = rng.choice(["", f"tags: [{words(1)}, {words(1)}]\n", f"author: {words(2)}\n"])
        return fields, words(rng.randint(0, 3)), words(rng.randint(1, 60))

    return {f"{name}{n:03}.md": content() for n in range(notes)}


def chunks_of(connection: Connection) -> Chunks:
    with reading(connection) as transaction:
        found = instances(transaction)
        documents = {row.chunk: row.document for row in texts(transaction)}
    held: Chunks = {chunk: (document, [[], [], []]) for chunk, document in documents.items()}
    for each in found:
        held[each.chunk][1][COLUMNS.index(each.column)].append(each.term)
    return held


def tf(leaf: Leaf, columns: list[list[str]]) -> int:
    match leaf:
        case Term(text):
            return sum(column.count(text) for column in columns)
        case Prefix(text):
            return sum(token.startswith(text) for column in columns for token in column)
        case Phrase(tokens):
            n = len(tokens)
            return sum(tuple(column[at : at + n]) == tokens for column in columns for at in range(len(column)))


def held_by(expression: Expression, present: set[Leaf]) -> tuple[bool, list[Leaf]]:
    match expression:
        case Term() | Phrase() | Prefix():
            return expression in present, [expression] if expression in present else []
        case Not(kept, dropped):
            ok, leaves = held_by(kept, present)
            ok = ok and not held_by(dropped, present)[0]
            return ok, leaves if ok else []
        case And(parts) | Or(parts):
            found = [held_by(part, present) for part in parts]
            ok = (all if isinstance(expression, And) else any)(each for each, _ in found)
            leaves = [leaf for _, each in found for leaf in each]
            return ok, leaves if ok or isinstance(expression, Or) else []
        case _:
            raise AssertionError(expression)


def brute(stores: Sequence[Chunks], expression: Expression, k: int, *, by_document: bool) -> list[tuple[int, float]]:
    every = {chunk: held for store in stores for chunk, held in store.items()}
    chunks = len(every)
    average = sum(len(column) for _, columns in every.values() for column in columns) / chunks
    named = set(leaves_of(expression))
    counts = {leaf: {chunk: tf(leaf, columns) for chunk, (_, columns) in every.items()} for leaf in named}
    df = {leaf: sum(n > 0 for n in counts[leaf].values()) for leaf in named}
    weight = {leaf: max(math.log((chunks - df[leaf] + 0.5) / (df[leaf] + 0.5)), 1e-6) for leaf in named}
    found: list[tuple[float, int, int]] = []
    for chunk, (document, columns) in every.items():
        length = sum(map(len, columns))
        ok, counted = held_by(expression, {leaf for leaf in named if counts[leaf][chunk]})
        if ok:
            score = sum(
                weight[leaf]
                * counts[leaf][chunk]
                * (FTS5_K1 + 1)
                / (counts[leaf][chunk] + FTS5_K1 * (1 - FTS5_B + FTS5_B * length / average))
                for leaf in counted
            )
            found.append((score, chunk, document))
    found.sort(key=lambda each: (-each[0], each[1]))
    if by_document:
        seen: set[int] = set()
        found = [each for each in found if each[2] not in seen and not seen.add(each[2])]
    return [(chunk, score) for score, chunk, _ in found[:k]]


def leaves_of(expression: Expression) -> list[Leaf]:
    match expression:
        case Term() | Phrase() | Prefix():
            return [expression]
        case Not(kept, dropped):
            return leaves_of(kept) + leaves_of(dropped)
        case And(parts) | Or(parts):
            return [leaf for part in parts for leaf in leaves_of(part)]
        case _:
            return []


TIE = 1e-12


def same(got: tuple[Found, ...], more: list[tuple[int, float]], k: int, query: object) -> None:
    want = more[:k]
    assert len(got) == len(want), query
    assert np.allclose([each.score for each in got], [score for _, score in want], rtol=1e-9), (query, got, want)
    ranked = dict(more)
    for each, (chunk, score) in zip(got, want, strict=True):
        if each.chunk != chunk:
            assert each.chunk in ranked, (query, got, want)
            assert abs(ranked[each.chunk] - score) <= TIE * abs(score), (query, got, want)


def random_expression(rng: Draws, phrases: list[tuple[str, ...]], depth: int = 0) -> Expression:
    if depth >= 2 or rng.random() < 0.35:
        pick = rng.random()
        if pick < 0.2 and phrases:
            return Phrase(rng.choice(phrases))
        if pick < 0.3:
            return Prefix(rng.choice(["w1", "w2", "mul", "w"]))
        return Term(rng.choice(TERMS))
    kind = rng.choice(["and", "or", "or", "not"])
    if kind == "not":
        return Not(random_expression(rng, phrases, depth + 1), random_expression(rng, phrases, depth + 1))
    parts = tuple(random_expression(rng, phrases, depth + 1) for _ in range(rng.randint(2, 3)))
    return And(parts) if kind == "and" else Or(parts)


def phrases_in(stores: Sequence[Chunks]) -> list[tuple[str, ...]]:
    found = {
        tuple(column[at : at + 2])
        for store in stores
        for _, columns in store.values()
        for column in columns
        for at in range(len(column) - 1)
    }
    overlapping = {each for each in found if each[0] == each[1]}
    return sorted(found)[:20] + sorted(overlapping)[:5] + [("w0", "w1", "w0")]


@contextlib.contextmanager
def collections(
    rounds: Sequence[Sequence[Round]], shape: Shape = SMALL
) -> Generator[tuple[list[Opened], list[Chunks]]]:
    with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
        stores: list[Opened] = []
        held: list[Chunks] = []
        for number, each in enumerate(rounds, 2):
            connection = stack.enter_context(opened_store(Path(folder), number, shape))
            applied(connection, list(each))
            held.append(chunks_of(connection))
            stores.append(opened(stack.enter_context(reading(connection))))
        yield stores, held


@settings(max_examples=6)
@given(seed=SEEDS, shape=st.sampled_from([SMALL, Shape()]))
def test_finds_what_scoring_every_chunk_finds(seed: int, shape: Shape) -> None:
    rng = Draws(np.random.default_rng(seed))
    gone: Round = {f"n{n:03}.md": None for n in range(0, 60, 7)}
    rounds = [prose(rng, 60), prose(rng, 50, "m"), gone, prose(rng, 8)]
    with collections([rounds], shape) as (stores, held):
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


def fts5_ranked(held: Chunks, query: str, k: int) -> list[tuple[int, float]]:
    with contextlib.closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE VIRTUAL TABLE t USING fts5(title, text, meta, tokenize='unicode61 remove_diacritics 2')")
        db.execute("CREATE TABLE owners(chunk INTEGER PRIMARY KEY, document INTEGER)")
        for chunk, (document, columns) in held.items():
            db.execute(
                "INSERT INTO t(rowid, title, text, meta) VALUES (?, ?, ?, ?)", (chunk, *(" ".join(c) for c in columns))
            )
            db.execute("INSERT INTO owners VALUES (?, ?)", (chunk, document))
        sql = """WITH p AS MATERIALIZED (SELECT t.rowid chunk, o.document document, -bm25(t) score FROM t
                 JOIN owners o ON o.chunk = t.rowid WHERE t MATCH ?)
                 SELECT chunk, max(score) FROM p GROUP BY document ORDER BY 2 DESC, 1 LIMIT ?"""
        return cast("list[tuple[int, float]]", db.execute(sql, (query, k)).fetchall())


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
