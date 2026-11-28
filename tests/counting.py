from dataclasses import dataclass, field

from sift.domain.arrays import Ids
from sift.domain.codec.codes import Halves
from sift.storage.database import Reading
from sift.storage.vectors import documents_embedded, vectors_of


@dataclass(slots=True)
class VectorReads:
    read: list[int] = field(default_factory=list[int])

    def __call__(self, transaction: Reading, chunks: Ids) -> tuple[Ids, Halves]:
        self.read.append(len(chunks))
        return vectors_of(transaction, chunks)


@dataclass(slots=True)
class DocumentCounts:
    counted: list[int] = field(default_factory=list[int])

    def __call__(self, transaction: Reading) -> int:
        self.counted.append(documents_embedded(transaction))
        return self.counted[-1]
