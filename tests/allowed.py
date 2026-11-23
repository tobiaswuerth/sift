from dataclasses import dataclass, field

import numpy as np

from sift.domain.arrays import Ids, Mask


@dataclass(slots=True)
class Chosen:
    ids: Ids
    owners: Ids = field(default_factory=lambda: np.zeros(0, np.int64))
    asked: list[int] = field(default_factory=list[int])
    listings: list[bool] = field(default_factory=list[bool])

    def documents(self, documents: Ids) -> Mask:
        self.asked.append(len(documents))
        return np.isin(documents, self.owners)

    def checked(self, ids: Ids) -> Mask:
        self.asked.append(len(ids))
        return np.isin(ids, self.ids)

    def listed(self) -> Ids:
        self.listings.append(True)
        return np.sort(self.ids)
