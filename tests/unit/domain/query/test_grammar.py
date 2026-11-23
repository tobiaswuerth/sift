import sqlite3
from collections.abc import Sequence
from contextlib import closing
from itertools import combinations
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sift.api.errors import InvalidRequestError
from sift.domain.arrays import Mask, ints
from sift.domain.query.expressions import NOTHING, And, Expression, Leaf, Not, Or, Phrase, Prefix, Term, leaves
from sift.domain.query.grammar import MOST_DEPTH, MOST_QUERY_CHARS, MOST_WORDS, parsed, relaxed, words_in
from sift.domain.rank.boolean import counted, may_hold

DASH = "\N{EN DASH}"
a, b, c = Term("a"), Term("b"), Term("c")


@pytest.mark.parametrize(
    "query, expression",
    [
        ("cancer", Term("cancer")),
        ('"cancer"', Term("cancer")),
        ("Müller naïve", And((Term("muller"), Term("naive")))),
        ("a OR b c", Or((a, And((b, c))))),
        ("a b OR c", Or((And((a, b)), c))),
        ("a NOT b c", Not(a, And((b, c)))),
        ("a NOT b NOT c", Not(Not(a, b), c)),
        ("a OR b NOT c", Or((a, Not(b, c)))),
        ("a AND b OR c", Or((And((a, b)), c))),
        ("a AND (b OR c)", And((a, Or((b, c))))),
        ("a AND b AND c", And((a, b, c))),
        ("a b AND c", And((a, b, c))),
        ("(a OR b) c", And((Or((a, b)), c))),
        ('"w1" OR "w2" OR "w3"', Or((Term("w1"), Term("w2"), Term("w3")))),
        ("a and b", And((a, Term("and"), b))),
        ("日本語", Term("日本語")),
        ('  "a"\t\nb\r', And((a, b))),
        ('"a"""', a),
        ('"retrieval augmented"', Phrase(("retrieval", "augmented"))),
        ("a_b", Phrase(("a", "b"))),
        ("first-stage", Phrase(("first", "stage"))),
        ("COVID-19 vaccine", And((Phrase(("covid", "19")), Term("vaccine")))),
        ("2301.09728", Phrase(("2301", "09728"))),
        ('w1 "First Stage"', And((Term("w1"), Phrase(("first", "stage"))))),
        ("superconduct*", Prefix("superconduct")),
        ("Straßenverkehr*", Prefix("straßenverkehr")),
        ("reinforc* learning", And((Prefix("reinforc"), Term("learning")))),
        ('"pigm*"', Term("pigm")),
        ("title: a", And((Term("title"), a))),
        ("^a", a),
        ("NEAR a", And((Term("near"), a))),
        ("a\N{NO-BREAK SPACE}b", Phrase(("a", "b"))),
        ("a\fb", And((a, b))),
        ("a\N{NULL}b", And((a, b))),
        (f"Deep learning {DASH} a review", And((Term("deep"), Term("learning"), a, Term("review")))),
        (f'"deep" "{DASH}" "learning"', And((Term("deep"), Term("learning")))),
        ('a ""', a),
        (f"a OR {DASH}", a),
        (f"{DASH} OR a", a),
        (f"a NOT {DASH}", a),
        (f"a NOT ({DASH} {DASH})", a),
        (f"a AND ({DASH} OR b)", And((a, b))),
        (f"({DASH} AND a) OR b", b),
        (f"a {DASH} AND b", And((a, b))),
        (f"a ({DASH})", a),
        ("", NOTHING),
        ("   ", NOTHING),
        (DASH, NOTHING),
        (f"a AND {DASH}", NOTHING),
        (f"{DASH} NOT a", NOTHING),
        (f"{DASH}*", NOTHING),
        ('"dark"*', Prefix("dark")),
        ('a "b"* c', And((a, Prefix("b"), c))),
    ],
)
def test_a_query_is_read_as_fts5_reads_it_where_both_read_it(query: str, expression: Expression) -> None:
    assert parsed(query) == expression


@pytest.mark.parametrize(
    "query, problem",
    [
        ('a "b', "a quote is not closed"),
        ("(a", 'a "(" is not closed'),
        ("a)", 'a ")" closes nothing'),
        ("()", "parentheses hold nothing"),
        ("a ()", "parentheses hold nothing"),
        ("NOT a", "NOT needs words before and after it"),
        ("a AND", "it ends where words are missing"),
        ("a OR OR b", "OR needs words before and after it"),
        ("(a OR)", '")" where words are missing'),
        ("NEAR(a b)", "NEAR is not supported"),
        ("NEAR (a b)", "NEAR is not supported"),
        ("ab" + chr(0xD800) + "*", "half a surrogate pair"),
        ("a " * (MOST_WORDS + 1), f"more than {MOST_WORDS:,} words"),
        ('"' + "a " * (MOST_WORDS + 1) + '"', f"more than {MOST_WORDS:,} words"),
        ("su*per", "* ends a prefix"),
        ("*", "* ends a prefix"),
        ("a**", "* ends a prefix"),
        ("first-sta*", "a prefix is one word"),
        ('"dark mat"*', "a prefix is one word"),
        ("a + b", "+ joins nothing here"),
        ("x" * (MOST_QUERY_CHARS + 1), f"more than {MOST_QUERY_CHARS:,} characters"),
        ("(" * (MOST_DEPTH + 1) + "a" + ")" * (MOST_DEPTH + 1), "parentheses nested more than"),
        ("a" + " NOT a" * (MOST_DEPTH + 1), "it nests more than"),
    ],
)
def test_a_query_outside_the_grammar_is_refused_with_what_to_write(query: str, problem: str) -> None:
    with pytest.raises(InvalidRequestError, match="the query is not understood") as refused:
        parsed(query)
    assert problem in refused.value.message
    assert refused.value.remedy


def test_parentheses_nested_as_deep_as_allowed_are_read() -> None:
    assert parsed("(" * MOST_DEPTH + "a" + ")" * MOST_DEPTH) == a


def test_a_leaf_written_twice_is_counted_twice() -> None:
    assert leaves(parsed('a OR (a AND "b c")')) == (a, a, Phrase(("b", "c")))
    assert leaves(NOTHING) == ()


@pytest.mark.parametrize(
    "query, expression",
    [
        ("dark matter halo", Or((Term("dark"), Term("matter"), Term("halo")))),
        ('"dark matter" halo*', Or((Phrase(("dark", "matter")), Prefix("halo")))),
        (f"a {DASH} b", Or((a, b))),
        ("(a b)", Or((a, b))),
        ("a _", a),
        ("a", None),
        ("first-stage", None),
        ("a OR b", None),
        ("a NOT b", None),
        ("a AND b", None),
        (f"{DASH} {DASH}", None),
    ],
)
def test_a_query_of_words_alone_is_relaxed_to_any_of_them(query: str, expression: Expression | None) -> None:
    assert relaxed(query) == expression


ROWS = tuple(" ".join(chosen) for n in range(1, 4) for chosen in combinations("abcde", n))


def pick[T](rng: np.random.Generator, items: Sequence[T]) -> T:
    return items[int(rng.integers(len(items)))]


def random_query(rng: np.random.Generator, depth: int = 0) -> str:
    if depth > 2 or rng.random() < 0.3:
        words = [pick(rng, "abcdefx" + DASH) for _ in range(pick(rng, [1, 1, 2]))]
        written = [f'"{word}"' if rng.random() < 0.3 else word for word in words]
        return pick(rng, [" ", " ", " + "]).join(each + ("*" if rng.random() < 0.2 else "") for each in written)
    left, right = random_query(rng, depth + 1), random_query(rng, depth + 1)
    left = f"({left})" if rng.random() < 0.3 else left
    right = f"({right})" if rng.random() < 0.3 else right
    return f"{left} {pick(rng, ['AND', 'OR', 'NOT'])} {right}"


@settings(max_examples=3)
@given(st.integers(0, 2**32 - 1))
def test_random_queries_match_what_fts5_matches(seed: int) -> None:
    rng = np.random.default_rng(seed)
    held = {word: mask(*(word in row.split() for row in ROWS)) for word in "abcdefx"}
    present: dict[Leaf, Mask] = {Term(word): found for word, found in held.items()}
    present |= {Prefix(word): found for word, found in held.items()}
    compared = 0
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='unicode61 remove_diacritics 2')")
        db.executemany("INSERT INTO t(rowid, x) VALUES (?, ?)", list(enumerate(ROWS, 1)))
        for _ in range(500):
            query = random_query(rng)
            try:
                matched = cast(
                    "list[tuple[int]]", db.execute("SELECT rowid FROM t WHERE t MATCH ?", (query,)).fetchall()
                )
                want = {row for (row,) in matched}
            except sqlite3.OperationalError:
                continue
            try:
                expression = parsed(query)
            except InvalidRequestError:
                continue
            found = (
                set[int]()
                if expression == NOTHING
                else {n + 1 for n in ints(np.flatnonzero(counted(expression, present)[0]))}
            )
            assert found == want, query
            compared += 1
    assert compared > 300


def mask(*values: int) -> Mask:
    return np.array(values, np.bool_)


def test_a_leaf_counts_only_where_no_and_or_not_above_it_fails() -> None:
    present: dict[Leaf, Mask] = {a: mask(1, 1, 0), b: mask(1, 0, 1), c: mask(0, 1, 1)}
    held, counting = counted(Or((And((a, b)), Not(c, a))), present)
    assert held.tolist() == [True, False, True]
    assert [(leaf, where.tolist()) for leaf, where in counting] == [
        (a, [True, False, False]),
        (b, [True, False, False]),
        (c, [False, False, True]),
    ]


def test_a_range_may_hold_a_not_wherever_it_holds_what_the_not_keeps() -> None:
    present: dict[Leaf, Mask] = {a: mask(1, 0), b: mask(1, 1)}
    assert may_hold(Not(a, b), present).tolist() == [True, False]
    assert may_hold(Or((a, And((a, b)))), present).tolist() == [True, False]


def test_nothing_is_never_searched_for() -> None:
    with pytest.raises(ValueError, match="nothing is searched for"):
        counted(NOTHING, {})
    with pytest.raises(ValueError, match="nothing is searched for"):
        may_hold(NOTHING, {})


def test_a_query_of_as_many_words_as_allowed_is_read() -> None:
    assert words_in(parsed("a " * MOST_WORDS)) == MOST_WORDS
