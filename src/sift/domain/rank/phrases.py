from collections.abc import Sequence

from sift import _native
from sift.domain.arrays import Ascending, Ids, integers, whole

type Occurrences = tuple[Ids, Ids, Ids]


def merged(places: Ids, counts: Ids) -> tuple[Ids, Ids]:
    found, sums = _native.summed_by_id(whole(places), whole(counts))
    return integers(found), integers(sums)


def phrase_counts(candidates: Ascending, tokens: Sequence[Occurrences]) -> Ids:
    told = [(whole(places), whole(counts), whole(positions)) for places, counts, positions in tokens]
    return integers(_native.phrase_counts(whole(candidates.values), told))
