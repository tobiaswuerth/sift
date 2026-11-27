from collections.abc import Sequence

import numpy as np

from sift import _native
from sift.domain.arrays import STORED_I64, Ascending, Ids, run_starts, sums_at, whole

type Occurrences = tuple[Ids, Ids, Ids]


def merged(places: Ids, counts: Ids) -> tuple[Ids, Ids]:
    order = np.argsort(places, kind="stable")
    places, counts = places[order], counts[order]
    starts = run_starts(places)
    return places[starts], sums_at(counts, starts) if len(starts) else counts


def phrase_counts(candidates: Ascending, tokens: Sequence[Occurrences]) -> Ids:
    told = [(whole(places), whole(counts), whole(positions)) for places, counts, positions in tokens]
    found = _native.phrase_counts(whole(candidates.values), told)
    return np.frombuffer(found, STORED_I64).astype(np.int64, copy=False)
