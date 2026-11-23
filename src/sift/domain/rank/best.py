from dataclasses import dataclass, field

import numpy as np

from sift.domain.arrays import Ids, Mask, Scores, firsts, floats, ints

ROUNDING = 1e-9


def _none() -> Ids:
    return np.zeros(0, np.int64)


def _no_scores() -> Scores:
    return np.zeros(0)


@dataclass(eq=False, frozen=True, slots=True)
class Best:
    k: int
    by_document: bool = True
    chunks: Ids = field(default_factory=_none)
    documents: Ids = field(default_factory=_none)
    scores: Scores = field(default_factory=_no_scores)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"the best {self.k} are none: k is one at least")

    @property
    def threshold(self) -> float:
        return floats(self.scores[-1:])[0] if len(self.scores) >= self.k else 0.0


def may_reach(bound: Scores, threshold: float) -> Mask:
    return bound * (1 + ROUNDING) >= threshold


def offered(best: Best, chunks: Ids, documents: Ids, scores: Scores) -> Best:
    if not len(chunks):
        return best
    held = np.concatenate([best.chunks, chunks])
    owners = np.concatenate([best.documents, documents])
    values = np.concatenate([best.scores, scores])
    if best.by_document:
        order = np.lexsort((held, -values, owners))
        order = order[firsts(owners[order])]
        held, owners, values = held[order], owners[order], values[order]
    kept = np.lexsort((held, -values))[: best.k]
    return Best(best.k, best.by_document, held[kept], owners[kept], values[kept])


def ranked(best: Best) -> tuple[tuple[int, float], ...]:
    return tuple(zip(ints(best.chunks), floats(best.scores), strict=True))
