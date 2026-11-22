import math
import sqlite3
from contextlib import closing

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sift.domain.arrays import floats
from sift.domain.rank.bm25 import TOP, idf, quantized, slack, term_scores, unquantized
from sift.domain.text.tokens import tokens

WORDS = ("alpha", "beta", "gamma", "delta", "Müller", "x-ray")
column = st.lists(st.sampled_from(WORDS), max_size=8).map(" ".join)
rows = st.lists(st.tuples(column, column, column), min_size=1, max_size=12)


def fts5_scores(written: list[tuple[str, str, str]], term: str) -> dict[int, float]:
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE VIRTUAL TABLE t USING fts5(a, b, c, tokenize='unicode61 remove_diacritics 2')")
        db.executemany("INSERT INTO t(rowid, a, b, c) VALUES (?, ?, ?, ?)", [(n, *r) for n, r in enumerate(written)])
        found: list[tuple[int, float]] = db.execute(
            "SELECT rowid, -bm25(t) FROM t WHERE t MATCH ?", (f'"{term}"',)
        ).fetchall()
        return dict(found)


@settings(max_examples=200)
@given(rows, st.sampled_from(["alpha", "beta", "muller", "ray"]))
def test_a_term_scores_as_fts5_s_bm25_scores_it(written: list[tuple[str, str, str]], term: str) -> None:
    held = [[token for text in row for token in tokens(text)] for row in written]
    average = sum(map(len, held)) / len(held)
    df = sum(term in row for row in held)
    want = fts5_scores(written, term)
    got = {
        n: idf(len(held), df) * floats(term_scores(np.array([row.count(term)]), np.array([len(row)]), average))[0]
        for n, row in enumerate(held)
        if term in row
    }
    assert got.keys() == want.keys()
    assert all(math.isclose(got[n], want[n], rel_tol=1e-12) for n in want)


def test_a_term_in_more_than_half_the_chunks_still_counts_a_little() -> None:
    assert idf(10, 9) == 1e-6
    assert idf(10, 1) == math.log(9.5 / 1.5)


@given(st.lists(st.floats(0, TOP), min_size=1, max_size=50))
def test_scores_kept_as_bytes_are_rounded_up_so_they_still_bound(scores: list[float]) -> None:
    held = np.array(scores)
    back = unquantized(quantized(held))
    assert np.all(back >= held)
    assert np.all(back - held <= TOP / 255 + 1e-12)


@given(st.integers(1, 50), st.integers(1, 2000), st.floats(1, 500), st.floats(1, 500))
def test_a_score_bounded_at_one_average_length_is_bounded_at_another_times_the_slack(
    tf: int, length: int, built: float, now: float
) -> None:
    before = floats(term_scores(np.array([tf]), np.array([length]), built))[0]
    after = floats(term_scores(np.array([tf]), np.array([length]), now))[0]
    assert after <= before * slack(now, built) * (1 + 1e-12)
