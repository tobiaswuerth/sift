from pathlib import Path

from sift.api.browsing import Browsed, BrowseFolder, ListedDocument
from sift.api.collections import Audience
from sift.api.errors import InvalidRequestError, field_at_fault
from sift.domain.text.kinds import public_id
from sift.storage.catalog import known_to, named, refuse_inactive
from sift.storage.listing import documents_in, folders_in
from sift.storage.store import read_store

PARENT = ".."
CLIMBING_REMEDY = 'name a folder inside the collection, e.g. "2024/"'


def _folder(written: str) -> str:
    parts = [part for part in written.split("/") if part]
    if PARENT in parts:
        raise InvalidRequestError(f"{written!r} climbs out of the collection", CLIMBING_REMEDY, "folder")
    return "".join(f"{part}/" for part in parts)


def browse(home: Path, request: BrowseFolder, *, audience: Audience = Audience.PEOPLE) -> Browsed:
    with field_at_fault("collection"):
        one = refuse_inactive(named(known_to(home, audience), request.collection))
    folder = _folder(request.folder)
    with read_store(one.store, one.id) as transaction:
        folders = folders_in(transaction, folder)
        found = documents_in(transaction, folder, request.k + 1)
    listed = tuple(ListedDocument(public_id(one.name, row.path, row.key), row.title) for row in found[: request.k])
    return Browsed(one.name, folder, tuple(folders), listed, more=len(found) > request.k)
