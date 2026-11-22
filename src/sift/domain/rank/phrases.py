from collections.abc import Sequence

import numpy as np

from sift.domain.arrays import Ascending, Ids, among, ascending, largest, run_starts, sums_at

MOST_KEY = 1 << 62

type Occurrences = tuple[Ids, Ids, Ids]


def merged(places: Ids, counts: Ids) -> tuple[Ids, Ids]:
    order = np.argsort(places, kind="stable")
    places, counts = places[order], counts[order]
    starts = run_starts(places)
    return places[starts], sums_at(counts, starts) if len(starts) else counts


def _keys(candidates: Ascending, held: Occurrences, shift: int, stride: int) -> Ids:
    places, counts, positions = held
    each = np.repeat(places, counts)
    kept = (positions >= shift) & among(each, candidates)
    return np.searchsorted(candidates.values, each[kept]) * stride + positions[kept] - shift


def _counted(candidates: Ascending, tokens: Sequence[Occurrences], stride: int) -> Ids:
    found = _keys(candidates, tokens[0], 0, stride)
    for shift, held in enumerate(tokens[1:], 1):
        found = found[among(found, ascending(_keys(candidates, held, shift, stride)))]
    return np.bincount(found // stride, minlength=len(candidates.values)).astype(np.int64)


def phrase_counts(candidates: Ascending, tokens: Sequence[Occurrences]) -> Ids:
    stride = 1 + max((largest(positions) for _, _, positions in tokens if len(positions)), default=0)
    group = max(1, MOST_KEY // stride)
    held = candidates.values
    parts = [_counted(Ascending(held[at : at + group]), tokens, stride) for at in range(0, len(held), group)]
    return np.concatenate([np.zeros(0, np.int64), *parts])
