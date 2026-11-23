from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import ints
from sift.domain.codec.codes import Floats

type Run = Callable[[list[str]], Floats]


@dataclass(frozen=True, slots=True)
class Shape:
    width: int
    batch: int


@dataclass(frozen=True, slots=True)
class Attempt:
    run: Run
    too_large: Callable[[BaseException], bool]
    freed: Callable[[], None]


@dataclass(frozen=True, slots=True)
class Batched:
    rows: Floats
    batch: int


def in_batches(texts: Sequence[str], lengths: Sequence[int], shape: Shape, attempt: Attempt) -> Batched:
    order = np.argsort(-np.array(lengths, np.int64), kind="stable")
    rows, batch = np.empty((len(texts), shape.width), np.float32), shape.batch
    start = 0
    while start < len(texts):
        part = order[start : start + batch]
        try:
            rows[part] = attempt.run([texts[at] for at in ints(part)])
        except Exception as error:
            if batch == 1 or not attempt.too_large(error):
                raise
            attempt.freed()
            batch //= 2
            continue
        start += len(part)
    return Batched(rows, batch)
