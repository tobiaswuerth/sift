import numpy as np
from numpy.typing import NDArray

from sift.domain.arrays import Ids, Scores

K1, B = 1.2, 0.75
TOP = K1 + 1
STEPS = 255


def term_scores(tf: Ids, lengths: Ids, average: float) -> Scores:
    counts = tf.astype(np.float64)
    return counts * TOP / (counts + K1 * (1 - B + B * lengths / average))


def unquantized(steps: NDArray[np.uint8]) -> Scores:
    return steps.astype(np.float64) * (TOP / STEPS)
