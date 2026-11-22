from collections.abc import Sequence

import numpy as np

from sift.domain.arrays import Ids, largest, run_starts, sums_at

MOST_KEY = 1 << 62

type Occurrences = tuple[Ids, Ids, Ids]


def merged(places: Ids, counts: Ids) -> tuple[Ids, Ids]:
    order = np.argsort(places, kind="stable")
    places, counts = places[order], counts[order]
    starts = run_starts(places)
    return places[starts], sums_at(counts, starts) if len(starts) else counts


def _keys(candidates: Ids, held: Occurrences, shift: int, stride: int) -> Ids:
    places, counts, positions = held
    each = np.repeat(places, counts)
    kept = (positions >= shift) & np.isin(each, candidates)
    return np.searchsorted(candidates, each[kept]) * stride + positions[kept] - shift


def _counted(candidates: Ids, tokens: Sequence[Occurrences], stride: int) -> Ids:
    found = _keys(candidates, tokens[0], 0, stride)
    for shift, held in enumerate(tokens[1:], 1):
        found = found[np.isin(found, _keys(candidates, held, shift, stride))]
    return np.bincount(found // stride, minlength=len(candidates)).astype(np.int64)


def phrase_counts(candidates: Ids, tokens: Sequence[Occurrences]) -> Ids:
    stride = 1 + max((largest(positions) for _, _, positions in tokens if len(positions)), default=0)
    group = max(1, MOST_KEY // stride)
    parts = [_counted(candidates[at : at + group], tokens, stride) for at in range(0, len(candidates), group)]
    return np.concatenate([np.zeros(0, np.int64), *parts])
