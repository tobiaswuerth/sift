from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from sift.domain.arrays import ints
from sift.domain.codec.codes import Floats, Halves

type Run = Callable[[list[str]], Floats]


@dataclass(frozen=True, slots=True)
class Shape:
    width: int
    batch: int


def in_batches(
    texts: Sequence[str], lengths: Sequence[int], shape: Shape, run: Run, too_large: Callable[[BaseException], bool]
) -> Halves:
    order = np.argsort(-np.array(lengths, np.int64), kind="stable")
    rows, batch = np.empty((len(texts), shape.width), np.float16), shape.batch
    start = 0
    while start < len(texts):
        part = order[start : start + batch]
        try:
            rows[part] = run([texts[at] for at in ints(part)]).astype(np.float16)
        except Exception as error:
            if batch == 1 or not too_large(error):
                raise
            batch //= 2
            continue
        start += len(part)
    return rows
