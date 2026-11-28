from collections.abc import Callable
from pathlib import Path

from sift.api.ids import CollectionId
from sift.storage.catalog import catalog_of, collections, same_place
from sift.storage.database import reading, writing
from sift.storage.ledger import forgive, owed, reclaimed
from sift.storage.store_files import unlinked


def tidied(home: Path, holding: CollectionId, unlink: Callable[[Path], None] = unlinked) -> tuple[Path, ...]:
    with catalog_of(home) as connection:
        with reading(connection) as transaction:
            if not owed(transaction):
                return ()
        with writing(connection) as transaction:
            known = collections(transaction)
            held = {one.id for one in known}
            left: list[Path] = []
            for owing in owed(transaction):
                store = Path(owing.store)
                if owing.collection != holding and owing.collection in held:
                    continue
                if any(same_place(store, one.store) for one in known):
                    forgive(transaction, store)
                else:
                    left.extend(reclaimed(transaction, owing, unlink))
    return tuple(left)
