import math

import numpy as np
from numpy.typing import NDArray

from sift.domain.arrays import Scores
from sift.domain.errors import DamagedError

type TermCounts = NDArray[np.int64]

K1, B = 1.2, 0.75
TOP = K1 + 1
LEAST_IDF = 1e-6
STEPS = 255
OVERCOUNTED = "the keyword index counts a word in more passages than it holds"
LENGTHLESS = "the keyword index holds words in passages of no length"


def idf(chunks: int, df: int) -> float:
    if df > chunks:
        raise DamagedError(OVERCOUNTED)
    value = math.log((chunks - df + 0.5) / (df + 0.5))
    return value if value > 0 else LEAST_IDF


def term_scores(tf: TermCounts, lengths: TermCounts, average: float) -> Scores:
    if not len(tf):
        return tf.astype(np.float64)
    if average <= 0:
        raise DamagedError(LENGTHLESS)
    counts = tf.astype(np.float64)
    return counts * TOP / (counts + K1 * (1 - B + B * lengths / average))


def unquantized(steps: NDArray[np.uint8]) -> Scores:
    return steps.astype(np.float64) * (TOP / STEPS)


def slack(average: float, built: float) -> float:
    if built <= 0:
        raise DamagedError(LENGTHLESS)
    return max(1.0, average / built)
