from typing import Protocol

from sift.domain.arrays import Ids, Mask


class Allowed(Protocol):
    def checked(self, ids: Ids) -> Mask: ...

    def listed(self) -> Ids: ...
