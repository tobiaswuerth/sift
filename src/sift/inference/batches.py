from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from sift.api.errors import ModelUnavailableError
from sift.domain.arrays import ints
from sift.domain.codec.codes import Floats

AHEAD = 2
WAITED_SECONDS = 600.0
STUCK = "a batch was not ready within {seconds:.0f} s: the tokenizer is stuck"
STUCK_REMEDY = "stop sift and embed again: what it embedded so far is kept"


@dataclass(frozen=True, slots=True)
class BatchShape:
    width: int
    batch: int


@dataclass(frozen=True, slots=True)
class Attempt[P]:
    prepare: Callable[[list[str]], P]
    run: Callable[[P], Floats]
    too_large: Callable[[BaseException], bool]
    freed: Callable[[], None]


@dataclass(eq=False, frozen=True, slots=True)
class Batched:
    rows: Floats
    batch: int


def _ready[P](future: "Future[P]", waited: float) -> P:
    try:
        return future.result(timeout=waited)
    except TimeoutError:
        raise ModelUnavailableError(STUCK.format(seconds=waited), STUCK_REMEDY) from None


@contextmanager
def _ahead[P](prepare: Callable[[list[str]], P], parts: Sequence[list[str]], waited: float) -> Generator[Iterator[P]]:
    pool = ThreadPoolExecutor(1, thread_name_prefix="sift-prepare")
    pending = deque(pool.submit(prepare, part) for part in parts[:AHEAD])

    def taken() -> Iterator[P]:
        for at in range(len(parts)):
            future = pending.popleft()
            if at + AHEAD < len(parts):
                pending.append(pool.submit(prepare, parts[at + AHEAD]))
            yield _ready(future, waited)

    try:
        yield taken()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True, cancel_futures=True)


def in_batches[P](
    texts: Sequence[str],
    lengths: Sequence[int],
    shape: BatchShape,
    attempt: Attempt[P],
    waited: float = WAITED_SECONDS,
) -> Batched:
    order = np.argsort(-np.array(lengths, np.int64), kind="stable")
    rows, batch, start = np.empty((len(texts), shape.width), np.float32), shape.batch, 0
    while start < len(texts):
        parts = [order[at : at + batch] for at in range(start, len(texts), batch)]
        failed = None
        with _ahead(attempt.prepare, [[texts[at] for at in ints(part)] for part in parts], waited) as prepared:
            for made, part in zip(prepared, parts, strict=True):
                try:
                    rows[part] = attempt.run(made)
                except RecursionError:
                    raise
                except Exception as error:
                    if len(part) == 1 or not attempt.too_large(error):
                        raise
                    failed = part
                    break
                start += len(part)
        if failed is not None:
            attempt.freed()
            batch = len(failed) // 2
    return Batched(rows, batch)
