"""The worker processes sift starts, to read files (`indexing`), build keyword postings (`postings`) and finish the
stores of an old index (`upgrade`): how many this machine gives, and the pools of them, made here alone.
"""

import multiprocessing
import os
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

WINDOWS = os.name == "nt"
MOST_ON_WINDOWS = 61  # Windows waits for 64 handles at once, some of them the pool's own: more is a ValueError there


def cpus() -> int:
    """CPUs this process may use, honouring affinity and container limits where the platform reports them."""
    process_cpu_count: Callable[[], int | None] | None = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count:
        return process_cpu_count() or 1
    sched_getaffinity: Callable[[int], set[int]] | None = getattr(os, "sched_getaffinity", None)  # Linux
    if sched_getaffinity:
        return len(sched_getaffinity(0))
    return os.cpu_count() or 1


def pool(n: int) -> ProcessPoolExecutor:
    """`n` worker processes, or as many as the system lets a pool have. Spawned everywhere, as on Windows and macOS:
    forking a process that has threads can deadlock."""
    return ProcessPoolExecutor(
        min(n, MOST_ON_WINDOWS) if WINDOWS else n, mp_context=multiprocessing.get_context("spawn")
    )


@contextmanager
def running(n: int) -> Generator[ProcessPoolExecutor]:
    """A pool (see `pool`) for the block. Left by an error, or stopped, it runs nothing it hasn't started: a pool left
    as its own context runs all it was handed first, and the error shows after."""
    made = pool(n)
    try:
        yield made
    finally:
        made.shutdown(cancel_futures=True)
