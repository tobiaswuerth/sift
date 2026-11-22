import math

import numpy as np
from numpy.typing import NDArray

type Scores = NDArray[np.float64]
type Counts = NDArray[np.int64]

K1, B = 1.2, 0.75
TOP = K1 + 1
LEAST_IDF = 1e-6
STEPS = 255


def idf(chunks: int, df: int) -> float:
    value = math.log((chunks - df + 0.5) / (df + 0.5))
    return value if value > 0 else LEAST_IDF


def term_scores(tf: Counts, lengths: Counts, average: float) -> Scores:
    counts = tf.astype(np.float64)
    return counts * TOP / (counts + K1 * (1 - B + B * lengths / average))


def quantized(scores: Scores) -> NDArray[np.uint8]:
    return np.minimum(np.ceil(scores * (STEPS / TOP)), STEPS).astype(np.uint8)


def unquantized(steps: NDArray[np.uint8]) -> Scores:
    return steps.astype(np.float64) * (TOP / STEPS)


def slack(average: float, built: float) -> float:
    return max(1.0, average / built)
