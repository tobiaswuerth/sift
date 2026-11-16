"""The scan of the vectors' codes (see `vectors`), compiled by Numba: each code read once, its Hamming distance to the
query counted where it is read, and the codes at each distance tallied in the same pass, in threads of Numba's own.
NumPy took three passes, writing and reading two arrays as large as the codes between them: over 12.9 million arXiv
codes, 43 ms against 133 (docs/benchmarks.md). Loaded by vectors.compile_scan, as Numba takes a second or two."""

import threading

import numba
import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic

PARTS = 64  # pieces of the codes the threads take in turn, each tallying its own distances
_ONE_AT_A_TIME = threading.Lock()  # Numba's threads take one scan at a time (its workqueue layer takes no more)


@intrinsic
def _popcount(typingctx, word):
    """The bits set in a 64-bit word, as the CPU counts them in one instruction."""

    def codegen(context, builder, signature, args):
        return builder.ctpop(args[0])

    return types.uint64(types.uint64), codegen


@njit(parallel=True, cache=True, nogil=True)
def _scan(words, query, keep, far, out, tallies):
    n, width = words.shape
    parts = tallies.shape[0]
    step = (n + parts - 1) // parts
    for part in prange(parts):
        for i in range(part * step, min(n, (part + 1) * step)):
            if keep.size and not keep[i]:
                out[i] = far
                continue
            distance = 0
            for k in range(width):
                distance += _popcount(words[i, k] ^ query[k])
            out[i] = distance
            tallies[part, distance] += 1


def measure(words, query, keep, out, far, threads):
    """Write the Hamming distance of each code, `words` (rows of uint64), to the query's (uint64) into `out`, `far`
    for those `keep` (bools, or empty for all) leaves out; and return how many codes kept are at each distance, from 0
    to every bit. In `threads` of Numba's."""
    n, width = words.shape
    if query.shape != (width,) or out.shape != (n,) or keep.shape not in ((0,), (n,)):
        raise ValueError(f"a scan of {n} codes of {width} words got a query of {query.shape}, out {out.shape}, keep "
                         f"{keep.shape}: compiled code checks no bounds")  # fmt: skip
    tallies = np.zeros((PARTS, 64 * width + 1), np.int64)
    with _ONE_AT_A_TIME:
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        _scan(words, query, keep, far, out, tallies)
    return tallies.sum(axis=0)
