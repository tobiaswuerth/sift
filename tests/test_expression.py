"""Keyword queries as FTS5 reads them (sift.expression): terms with AND, OR, NOT, parentheses and words side by side,
tokenized as FTS5 tokenizes; what it leaves to FTS5 itself; and, for random queries, the same matches as FTS5's."""

import itertools
import random
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from sift import expression, store, topk


def t(term):
    return ("term", term)


@pytest.mark.parametrize(
    "query, parsed",
    [
        ("cancer", t("cancer")),
        ('"cancer"', t("cancer")),
        ("Müller naïve", ("and", [t("muller"), t("naive")])),  # tokenized as the index was: folded, no diacritics
        ("a OR b c", ("or", [t("a"), ("and", [t("b"), t("c")])])),  # words side by side join first
        ("a b OR c", ("or", [("and", [t("a"), t("b")]), t("c")])),
        ("a NOT b c", ("not", t("a"), ("and", [t("b"), t("c")]))),
        ("a NOT b NOT c", ("not", ("not", t("a"), t("b")), t("c"))),
        ("a OR b NOT c", ("or", [t("a"), ("not", t("b"), t("c"))])),
        ("a AND b OR c", ("or", [("and", [t("a"), t("b")]), t("c")])),
        ("a AND (b OR c)", ("and", [t("a"), ("or", [t("b"), t("c")])])),
        ('"w1" OR "w2" OR "w3"', ("or", [t("w1"), t("w2"), t("w3")])),
        ("a and b", ("and", [t("a"), t("and"), t("b")])),  # operators are capitals
        ("日本語", t("日本語")),
        ('  "a"\t\nb\r', ("and", [t("a"), t("b")])),
        ('"a"""', t("a")),  # "" in quotes is a quote: a", one token
    ],
)
def test_parses_as_fts5_does(query, parsed):
    assert expression.parse(query) == parsed


@pytest.mark.parametrize(
    "query",
    [
        '"retrieval augmented"',  # a phrase
        "a_b",  # a phrase too: two tokens
        "e-mail",
        "pigm*",
        "NEAR(a b)",
        "title: a",
        "{title meta}: a",
        "^a",
        "a + b",
        '""',
        "2301.09728",
        "(a OR b) c",  # syntax errors, as FTS5 has them: FTS5 says what's wrong
        "NOT a",
        "a AND",
        "a OR OR b",
        "(a",
        "(a (b))",
        'a "b',
        "",
    ],
)
def test_leaves_to_fts5_what_it_doesnt_read(query):
    assert expression.parse(query) is None


ROWS = [" ".join(c) for n in range(1, 4) for c in itertools.combinations(["a", "b", "c", "d", "e"], n)]


def random_query(rng, depth=0):
    """A query string FTS5 reads, mostly: terms, AND, OR, NOT, parentheses, words side by side, quotes."""
    if depth > 2 or rng.random() < 0.3:
        words = [rng.choice("abcdefx") for _ in range(rng.choice([1, 1, 2]))]
        return " ".join(f'"{w}"' if rng.random() < 0.3 else w for w in words)
    left, right = random_query(rng, depth + 1), random_query(rng, depth + 1)
    if rng.random() < 0.3:
        left = f"({left})"
    if rng.random() < 0.3:
        right = f"({right})"
    return f"{left} {rng.choice(['AND', 'OR', 'NOT'])} {right}"


@pytest.fixture
def rows():
    """ROWS in an FTS5 table, tokenized as sift's are."""
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute(f"CREATE VIRTUAL TABLE t USING fts5(x, tokenize='{store.TOKENIZER}')")
        db.executemany("INSERT INTO t(rowid, x) VALUES (?, ?)", list(enumerate(ROWS, 1)))
        yield db


def test_random_queries_match_what_fts5_matches(rows):
    db = rows
    holding = {w: np.array([w in row.split() for row in ROWS]) for w in "abcdefx"}
    rng, read = random.Random(0), 0
    for _ in range(400):
        query = random_query(rng)
        try:
            want = {r for (r,) in db.execute("SELECT rowid FROM t WHERE t MATCH ?", (query,))}
        except sqlite3.OperationalError:
            assert expression.parse(query) is None, query  # FTS5's error, FTS5's to give
            continue
        parsed = expression.parse(query)
        if parsed is None:
            continue
        read += 1
        found, _ = topk.counted(parsed, holding)
        assert {i + 1 for i in np.flatnonzero(found)} == want, query
    assert read > 200  # most are read here, not left to FTS5
