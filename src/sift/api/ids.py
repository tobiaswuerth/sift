from typing import NewType

CollectionId = NewType("CollectionId", int)
DocumentId = NewType("DocumentId", int)
ChunkId = NewType("ChunkId", int)

ROW_BITS = 40
MOST_COLLECTIONS = (1 << (63 - ROW_BITS)) - 1


def first_row(collection: CollectionId) -> int:
    if not 0 < collection <= MOST_COLLECTIONS:
        raise ValueError(f"a collection id lies between 1 and {MOST_COLLECTIONS}, got {collection}")
    return collection << ROW_BITS


def owner(row: int) -> CollectionId:
    return CollectionId(row >> ROW_BITS)
