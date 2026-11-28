from typing import Protocol

from sift.domain.arrays import Ids, Mask


class AllowedDocuments(Protocol):
    def documents(self, documents: Ids) -> Mask: ...


class AllowedChunks(Protocol):
    def checked(self, ids: Ids) -> Mask: ...

    def listed(self) -> Ids: ...
