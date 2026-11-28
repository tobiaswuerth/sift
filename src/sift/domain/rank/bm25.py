import math

from sift.domain.errors import DamagedError

LEAST_IDF = 1e-6
OVERCOUNTED = "the keyword index counts a word in more passages than it holds"
LENGTHLESS = "the keyword index holds words in passages of no length"


def idf(chunks: int, df: int) -> float:
    if df > chunks:
        raise DamagedError(OVERCOUNTED)
    value = math.log((chunks - df + 0.5) / (df + 0.5))
    return value if value > 0 else LEAST_IDF


def slack(average: float, built: float) -> float:
    if built <= 0:
        raise DamagedError(LENGTHLESS)
    return max(1.0, average / built)
