from collections.abc import Sequence

import numpy as np

from sift.domain.arrays import Ids, run_starts, sums_at

STRIDE = 1 << 40

type Occurrences = tuple[Ids, Ids, Ids]


def merged(places: Ids, counts: Ids) -> tuple[Ids, Ids]:
    order = np.argsort(places, kind="stable")
    places, counts = places[order], counts[order]
    starts = run_starts(places)
    return places[starts], sums_at(counts, starts) if len(starts) else counts


def _keys(candidates: Ids, held: Occurrences, shift: int) -> Ids:
    places, counts, positions = held
    owners = np.searchsorted(candidates, np.repeat(places, counts))
    kept = positions >= shift
    return owners[kept] * STRIDE + positions[kept] - shift


def phrase_counts(candidates: Ids, tokens: Sequence[Occurrences]) -> Ids:
    found = _keys(candidates, tokens[0], 0)
    for shift, held in enumerate(tokens[1:], 1):
        found = found[np.isin(found, _keys(candidates, held, shift))]
    return np.bincount(found // STRIDE, minlength=len(candidates)).astype(np.int64)
