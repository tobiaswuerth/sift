import contextlib
import itertools
import math
import sqlite3
import tempfile
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from hypothesis import strategies as st

from sift.domain.codec.shape import Shape
from sift.domain.plan.segments import Policy
from sift.domain.query.expressions import And, Expression, Leaf, Not, Or, Phrase, Prefix, Term
from sift.domain.rank.best import Best
from sift.retrieval.keyword.lists import BlockKey, Opened, PostingArrays, opened
from sift.retrieval.keyword.search import (
    TUNING,
    FoundChunk,
    Tuning,
    found_in,
    planned,
    prepared,
    query_of,
    refuse_reading_too_much,
    searched,
)
from sift.storage.database import BATCHES, Batches, Connection, reading
from tests.keyword_stores import COLUMNS, SMALL, Content, Round, applied, instances, opened_store, texts

VOCABULARY = [f"w{n}" for n in range(30)] + ["Müller"]


SEEDS = st.integers(0, 2**32 - 1)


TERMS = [f"w{n}" for n in range(30)] + ["muller", "absent"]


FTS5_K1, FTS5_B = 1.2, 0.75


MERGING = Batches(chunks_a_segment=9, postings_a_merge=40, merging=Policy(factor=2, floor=1))


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


def tied(ranked: list[tuple[int, float]]) -> list[list[tuple[int, float]]]:
    groups: list[list[tuple[int, float]]] = []
    for each in sorted(ranked, key=lambda each: (-each[1], each[0])):
        if groups and abs(groups[-1][0][1] - each[1]) <= TIE * abs(each[1]):
            groups[-1].append(each)
        else:
            groups.append([each])
    return groups


def same(got: tuple[FoundChunk, ...], more: list[tuple[int, float]], k: int, query: object) -> None:
    groups = tied(more)
    want = [each for group in groups for each in group][:k]
    assert len(got) == len(want), query
    assert np.allclose([each.score for each in got], [score for _, score in want], rtol=1e-9), (query, got, want)
    assert list(got) == sorted(got, key=lambda each: (-each.score, each.chunk)), (query, got)
    assert len({each.chunk for each in got}) == len(got), (query, got)
    for group, start in zip(groups, itertools.accumulate(map(len, groups), initial=0), strict=False):
        held = {each.chunk for each in got[start : start + len(group)]}
        assert held <= {chunk for chunk, _ in group}, (query, got, want)


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


def fts5_query(expression: Expression) -> str:
    match expression:
        case Term(text):
            return f'"{text}"'
        case Phrase(tokens):
            return f'"{" ".join(tokens)}"'
        case Prefix(text):
            return f"{text}*"
        case Not(kept, dropped):
            return f"({fts5_query(kept)} NOT {fts5_query(dropped)})"
        case And(parts) | Or(parts):
            joined = " AND " if isinstance(expression, And) else " OR "
            return "(" + joined.join(map(fts5_query, parts)) + ")"
        case _:
            raise AssertionError(expression)


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


def blocks_read(store: Opened) -> list[BlockKey]:
    return list(cast("dict[BlockKey, PostingArrays]", store.decoded.blocks))


@contextlib.contextmanager
def collections(
    rounds: Sequence[Sequence[Round]], shape: Shape = SMALL, batches: Batches = BATCHES
) -> Generator[tuple[list[Opened], list[Chunks]]]:
    with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
        stores: list[Opened] = []
        held: list[Chunks] = []
        for number, each in enumerate(rounds, 2):
            connection = stack.enter_context(opened_store(Path(folder), number, shape))
            applied(connection, list(each), batches)
            held.append(chunks_of(connection))
            stores.append(opened(stack.enter_context(reading(connection))))
        yield stores, held


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


def search(
    stores: Sequence[Opened], expression: Expression, k: int, *, by_document: bool = True, tuning: Tuning = TUNING
) -> tuple[FoundChunk, ...]:
    plans = [planned(store, expression) for store in stores]
    refuse_reading_too_much(plans, tuning)
    each = [prepared(plan) for plan in plans]
    query = query_of(expression, each)
    if query is None:
        return ()
    best = Best(k, by_document)
    for held in each:
        best = searched(held, query, best, tuning).best
    return found_in(best)
