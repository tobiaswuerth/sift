from collections.abc import Mapping, Sequence

from sift.api.ids import DocumentId
from sift.domain.text.embedding import embedded
from sift.storage.database import Reading, Writing
from sift.storage.vectors import copy_vectors, held_vectors, placed_chunks

type KeptVectors = Mapping[str, tuple[bytes, bytes]]


def vectors_held(transaction: Reading, documents: Sequence[DocumentId]) -> KeptVectors:
    return {
        embedded(each.title, each.text, first=each.first): (each.vector, each.bits)
        for each in held_vectors(transaction, documents)
    }


def vectors_kept(transaction: Writing, documents: Sequence[DocumentId], held: KeptVectors) -> None:
    if not held:
        return
    placed = placed_chunks(transaction, documents)
    keys = ((each.chunk, embedded(each.title, each.text, first=each.first)) for each in placed)
    copy_vectors(transaction, [(chunk, *held[key]) for chunk, key in keys if key in held])
