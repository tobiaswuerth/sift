from dataclasses import dataclass, field

import numpy as np

from sift.domain.arrays import Ids, Mask


@dataclass(slots=True)
class Chosen:
    ids: Ids
    asked: list[int] = field(default_factory=list[int])
    listings: list[bool] = field(default_factory=list[bool])

    def checked(self, ids: Ids) -> Mask:
        self.asked.append(len(ids))
        return np.isin(ids, self.ids)

    def listed(self) -> Ids:
        self.listings.append(True)
        return np.sort(self.ids)
