import errno
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from sift.api.collections import AddCollection, MoveCollections, StoreMoved, StoresMoved
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Event, Phase, PhaseDone, PhaseStarted
from sift.api.ids import CollectionId
from sift.api.indexing import IndexCollections
from sift.api.search import SearchKeywords, SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.moving import Moving, move
from sift.app.paths import ABSOLUTE_REMEDY
from sift.app.search import search
from sift.app.vsearch import vsearch
from sift.storage.catalog import Collection, known_collections
from sift.storage.moving import Transfer
from sift.storage.store import create_store
from sift.storage.store_files import delete_store
from tests.files import write
from tests.models import Hashed


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    write(tmp_path / "notes" / "rl.md", "# Reward shaping\n\nSpeeds up learning.\n")
    add(home, AddCollection(str(tmp_path / "notes")))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    return home


def notes(home: Path) -> Collection:
    (found,) = [one for one in known_collections(home) if one.name == "notes"]
    return found


def searched(home: Path) -> tuple[list[str], list[str]]:
    words = [hit.id for hit in search(home, SearchKeywords("learning")).hits]
    return words, [hit.id for hit in vsearch(home, SearchVectors("reward shaping", k=1), Hashed()).hits]


def moved(home: Path, *names: str, to: Path, how: Moving | None = None, said: list[Event] | None = None) -> StoresMoved:
    report = (said if said is not None else []).append
    return move(home, MoveCollections(names, to=str(to)), report, how or Moving())


def no_link(_old: Path, _new: Path) -> None:
    raise OSError(errno.EXDEV, "Invalid cross-device link")


def test_a_store_moves_to_another_folder_and_is_searched_there(home: Path, tmp_path: Path) -> None:
    old, before = notes(home).store, searched(home)
    new = tmp_path / "elsewhere" / "notes.sqlite"
    assert moved(home, to=tmp_path / "elsewhere") == StoresMoved(
        (StoreMoved("notes", str(old), str(new), copied=False),), ()
    )
    assert notes(home).store == new
    assert not list(old.parent.glob(f"{old.name}*"))
    assert searched(home) == before == (["notes/rl.md"], ["notes/rl.md"])


def test_a_store_moves_to_a_file_of_another_name(home: Path, tmp_path: Path) -> None:
    moved(home, "notes", to=tmp_path / "data_notes.sqlite")
    assert notes(home).store == tmp_path / "data_notes.sqlite"
    assert searched(home) == (["notes/rl.md"], ["notes/rl.md"])


def test_a_store_moves_to_another_drive_as_a_copy_saying_how_far_it_got(home: Path, tmp_path: Path) -> None:
    before, said = searched(home), list[Event]()
    result = moved(home, "notes", to=tmp_path / "other drive", how=Moving(Transfer(no_link, pages=2)), said=said)
    assert result.moved[0].copied
    assert said[:2] == [
        CollectionStarted("notes", str(tmp_path / "other drive" / "notes.sqlite")),
        PhaseStarted(Phase.COPY, 1),
    ]
    assert any(isinstance(each, Advanced) for each in said)
    assert isinstance(said[-1], PhaseDone)
    assert not list((tmp_path / "other drive").glob("*.part"))
    assert searched(home) == before


def test_a_copy_stopped_leaves_the_store_where_it_was(home: Path, tmp_path: Path) -> None:
    old = notes(home).store

    def stopped(event: Event) -> None:
        if isinstance(event, Advanced):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        move(
            home,
            MoveCollections(("notes",), to=str(tmp_path / "other drive")),
            stopped,
            Moving(Transfer(no_link, pages=1)),
        )
    assert notes(home).store == old
    assert list((tmp_path / "other drive").iterdir()) == []
    assert searched(home) == (["notes/rl.md"], ["notes/rl.md"])


def test_a_store_the_catalog_could_not_follow_stays_where_it_was(home: Path, tmp_path: Path) -> None:
    old = notes(home).store

    def full(_home: Path, _identity: CollectionId, _store: Path) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(OSError, match="No space left"):
        moved(home, "notes", to=tmp_path / "elsewhere", how=Moving(repoint=full))
    assert notes(home).store == old
    assert list((tmp_path / "elsewhere").iterdir()) == []
    assert searched(home) == (["notes/rl.md"], ["notes/rl.md"])


def test_a_store_another_sift_has_open_stays(home: Path, tmp_path: Path) -> None:
    with closing(sqlite3.connect(notes(home).store)) as reader:
        reader.execute("SELECT count(*) FROM documents").fetchone()
        with pytest.raises(StoreUnavailableError) as refused:
            moved(home, to=tmp_path / "elsewhere")
    assert refused.value.code is Problem.BUSY
    assert notes(home).store == home / "collections" / "notes.sqlite"
    assert not (tmp_path / "elsewhere" / "notes.sqlite").exists()


def test_a_store_is_not_moved_onto_a_file(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "elsewhere" / "notes.sqlite", "mine")
    with pytest.raises(InvalidRequestError, match="exists already"):
        moved(home, to=tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "notes.sqlite").read_text() == "mine"


def test_several_stores_move_into_a_folder_only(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "other" / "a.md", "# A\n")
    add(home, AddCollection(str(tmp_path / "other")))
    with pytest.raises(InvalidRequestError, match="the file of one collection's store"):
        moved(home, to=tmp_path / "all.sqlite")
    assert sorted(each.name for each in moved(home, to=tmp_path / "all").moved) == ["notes", "other"]


def test_a_store_where_it_is_already_stays(home: Path) -> None:
    assert moved(home, "notes", to=home / "collections") == StoresMoved((), ("notes",))


def test_a_missing_store_cannot_be_moved(home: Path, tmp_path: Path) -> None:
    notes(home).store.unlink()
    with pytest.raises(StoreUnavailableError) as refused:
        moved(home, "notes", to=tmp_path / "elsewhere")
    assert refused.value.code is Problem.STORE_MISSING


def test_a_store_moved_but_still_open_at_its_old_place_is_left_to_delete(home: Path, tmp_path: Path) -> None:
    old = notes(home).store

    def held_open(file: Path) -> None:
        if file == old:
            raise PermissionError(13, "The process cannot access the file", str(file))
        file.unlink()

    (result,) = moved(home, "notes", to=tmp_path / "elsewhere", how=Moving(unlink=held_open)).moved
    assert result.left == (str(old),)
    assert notes(home).store == tmp_path / "elsewhere" / "notes.sqlite"


def linked_then_stopped(old: Path, new: Path) -> None:
    os.link(old, new)
    raise KeyboardInterrupt


def stopped(_file: Path) -> None:
    raise KeyboardInterrupt


def test_a_move_stopped_before_its_catalog_changed_is_finished_by_the_next(home: Path, tmp_path: Path) -> None:
    old, to = notes(home).store, tmp_path / "elsewhere"
    with pytest.raises(KeyboardInterrupt):
        moved(home, "notes", to=to, how=Moving(transfer=Transfer(link=linked_then_stopped)))
    assert notes(home).store == old
    (result,) = moved(home, "notes", to=to).moved
    assert (result.new, result.left) == (str(to / "notes.sqlite"), ())
    assert notes(home).store == to / "notes.sqlite"
    assert not old.exists()
    assert searched(home) == (["notes/rl.md"], ["notes/rl.md"])


def first_move_stopped_before_the_old_copy_went(home: Path, tmp_path: Path) -> Path:
    old = notes(home).store
    with pytest.raises(KeyboardInterrupt):
        moved(home, "notes", to=tmp_path / "first", how=Moving(unlink=stopped))
    assert notes(home).store == tmp_path / "first" / "notes.sqlite"
    assert old.exists()
    return old


def test_the_old_copy_of_a_stopped_move_is_deleted_by_the_next_move(home: Path, tmp_path: Path) -> None:
    old = first_move_stopped_before_the_old_copy_went(home, tmp_path)
    moved(home, "notes", to=tmp_path / "second")
    assert not old.exists()
    assert not (tmp_path / "first" / "notes.sqlite").exists()
    assert searched(home) == (["notes/rl.md"], ["notes/rl.md"])


def test_the_old_copy_of_a_stopped_move_is_deleted_by_the_next_index(home: Path, tmp_path: Path) -> None:
    old = first_move_stopped_before_the_old_copy_went(home, tmp_path)
    index(home, IndexCollections(), lambda _: None)
    assert sorted(old.parent.iterdir()) == []
    assert searched(home) == (["notes/rl.md"], ["notes/rl.md"])


def test_a_store_of_another_collection_where_a_move_left_its_old_copy_is_kept(home: Path, tmp_path: Path) -> None:
    old = first_move_stopped_before_the_old_copy_went(home, tmp_path)
    delete_store(old)
    create_store(old, CollectionId(99))
    index(home, IndexCollections(), lambda _: None)
    index(home, IndexCollections(), lambda _: None)
    assert old.is_file()


def test_what_a_stopped_move_left_is_tidied_by_its_own_collection_alone(home: Path, tmp_path: Path) -> None:
    old = first_move_stopped_before_the_old_copy_went(home, tmp_path)
    write(tmp_path / "diary" / "d.md", "# Diary")
    add(home, AddCollection(str(tmp_path / "diary")))
    index(home, IndexCollections(("diary",)), lambda _: None)
    assert old.exists()
    index(home, IndexCollections(("notes",)), lambda _: None)
    assert not old.exists()


def test_a_place_a_stopped_move_left_is_kept_once_another_collection_is_stored_there(
    home: Path, tmp_path: Path
) -> None:
    place = tmp_path / "elsewhere" / "notes.sqlite"
    with pytest.raises(KeyboardInterrupt):
        moved(home, "notes", to=place.parent, how=Moving(transfer=Transfer(link=linked_then_stopped)))
    delete_store(place)
    write(tmp_path / "diary" / "d.md", "# Diary")
    add(home, AddCollection(str(tmp_path / "diary"), store=str(place)))
    index(home, IndexCollections(), lambda _: None)
    index(home, IndexCollections(), lambda _: None)
    assert place.is_file()
    assert [hit.id for hit in search(home, SearchKeywords("diary")).hits] == ["diary/d.md"]


def test_a_place_a_stopped_move_left_open_elsewhere_is_not_moved_onto(home: Path, tmp_path: Path) -> None:
    place = tmp_path / "elsewhere" / "notes.sqlite"
    with pytest.raises(KeyboardInterrupt):
        moved(home, "notes", to=place.parent, how=Moving(transfer=Transfer(link=linked_then_stopped)))

    def held_open(file: Path) -> None:
        if file == place:
            raise PermissionError(13, "The process cannot access the file", str(file))
        file.unlink()

    with pytest.raises(InvalidRequestError, match="exists already"):
        moved(home, "notes", to=place.parent, how=Moving(unlink=held_open))
    assert moved(home, "notes", to=place.parent).moved[0].new == str(place)


def test_a_place_to_move_to_that_is_not_absolute_is_refused(tmp_path: Path) -> None:
    home = tmp_path / "home"
    folder = tmp_path / "notes"
    folder.mkdir()
    add(home, AddCollection(str(folder)))
    with pytest.raises(InvalidRequestError, match="is no absolute path") as refused:
        move(home, MoveCollections(to="elsewhere"), lambda _: None)
    assert (refused.value.field, refused.value.remedy) == ("to", ABSOLUTE_REMEDY)
