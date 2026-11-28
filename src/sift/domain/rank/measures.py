import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

CUT = 10
MOST_GRADE = 10

type Grades = Mapping[str, float]


def ndcg(ranked: Sequence[str], relevant: Grades, k: int = CUT) -> float:
    gained = sum((2 ** relevant.get(each, 0) - 1) / math.log2(at + 2) for at, each in enumerate(ranked[:k]))
    best = sorted(relevant.values(), reverse=True)[:k]
    ideal = sum((2**grade - 1) / math.log2(at + 2) for at, grade in enumerate(best))
    return gained / ideal


def reciprocal_rank(ranked: Sequence[str], relevant: Grades, k: int = CUT) -> float:
    return next((1 / (at + 1) for at, each in enumerate(ranked[:k]) if each in relevant), 0.0)


def recall(ranked: Sequence[str], relevant: Grades, k: int = CUT) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


@dataclass(frozen=True, slots=True)
class Judged:
    ranked: tuple[str, ...]
    relevant: Grades


def found(hits: Sequence[Sequence[str]], relevant: Grades) -> Judged:
    left, ranked = dict(relevant), list[str]()
    for copies in hits:
        best = max(copies, key=lambda each: left.get(each, 0))
        graded = [left.pop(each) for each in copies if each in left]
        if graded:
            left[best] = max(graded)
        ranked.append(best)
    return Judged(tuple(ranked), MappingProxyType(left))
