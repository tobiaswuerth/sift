from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sift.api.collections import MoveCollections, StoreMoved, StoresMoved
from sift.api.errors import InvalidRequestError
from sift.api.events import Advanced, CollectionStarted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.ids import CollectionId
from sift.app.collections import chosen
from sift.app.paths import absolute
from sift.storage.catalog import (
    STORE_SUFFIX,
    Collection,
    known_collections,
    owe_files,
    owed_to,
    paid,
    same_place,
    set_store,
)
from sift.storage.leftovers import tidied
from sift.storage.moving import TRANSFER, Copied, Transfer, held_alone, transferred
from sift.storage.store import read_store
from sift.storage.store_files import store_files_left, unlinked
from sift.storage.writer import sole_writer

MB = 1 << 20
ONE_FILE_REMEDY = "name one collection, or give a folder"
TAKEN_REMEDY = "move it away, or move the collection elsewhere"


@dataclass(frozen=True, slots=True)
class Moving:
    transfer: Transfer = TRANSFER
    unlink: Callable[[Path], None] = unlinked
    repoint: Callable[[Path, CollectionId, Path], None] = set_store


MOVING = Moving()


def _megabytes(count: int) -> int:
    return -(-count // MB)


def _reported(report: Reporter) -> Copied:
    def copied(done: int, total: int) -> None:
        report(Advanced(done // MB) if done else PhaseStarted(Phase.COPY, _megabytes(total)))

    return copied


def _planned(home: Path, request: MoveCollections) -> list[tuple[Collection, Path]]:
    to = absolute(request.to, "to").resolve()
    found = chosen(home, request.names)
    if to.name.lower().endswith(STORE_SUFFIX):
        if len(found) != 1:
            raise InvalidRequestError(f"{to} is the file of one collection's store", ONE_FILE_REMEDY, "to")
        return [(found[0], to)]
    return [(one, to / one.store.name) for one in found]


def _taken(new: Path) -> InvalidRequestError:
    return InvalidRequestError(f"{new} exists already", TAKEN_REMEDY, "to")


def _refused(home: Path, one: Collection, new: Path) -> None:
    with read_store(one.store, one.id):
        pass
    others = [other.store for other in known_collections(home) if other.id != one.id]
    left_by_a_move = any(same_place(new, owed) for owed in owed_to(home, one.id))
    if (new.exists() and not left_by_a_move) or any(same_place(other, new) for other in others):
        raise _taken(new)


def _moved(home: Path, one: Collection, new: Path, report: Reporter, how: Moving) -> StoreMoved:
    report(CollectionStarted(one.name, str(new)))
    with sole_writer(one.store, "sift move"), held_alone(one.store) as held:
        tidied(home, one.id, how.unlink)
        if new.exists():
            raise _taken(new)
        owe_files(home, new, one.id)
        copied = transferred(held, one.store, new, _reported(report), how.transfer)
        try:
            how.repoint(home, one.id, new)
        except BaseException:
            store_files_left(new)
            raise
    if copied:
        report(PhaseDone(_megabytes(new.stat().st_size)))
    left = tuple(str(file) for file in store_files_left(one.store, how.unlink))
    if not left:
        paid(home, one.store)
    return StoreMoved(one.name, str(one.store), str(new), copied=copied, left=left)


def move(home: Path, request: MoveCollections, report: Reporter, how: Moving = MOVING) -> StoresMoved:
    planned = _planned(home, request)
    waiting = [(one, new) for one, new in planned if not same_place(one.store, new)]
    for one, new in waiting:
        _refused(home, one, new)
    moved = tuple(_moved(home, one, new, report, how) for one, new in waiting)
    return StoresMoved(moved, tuple(one.name for one, new in planned if same_place(one.store, new)))
