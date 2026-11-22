import contextlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.collections import AddCollection, Counted, ShowStatus, TitleFrom, UnreadableFile
from sift.api.errors import NotFoundError
from sift.api.events import Advanced, CollectionStarted, Event, Finished, Noted, Phase, PhaseDone, PhaseStarted
from sift.api.ids import CollectionId
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import add, status
from sift.app.indexing import FOLDER_MISSING_REMEDY, Limits, index
from sift.app.problems import LEFT_AS_INDEXED
from sift.storage.database import reading, rows
from sift.storage.documents import stored
from sift.storage.store import MISSING_REMEDY, open_store


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def notes(tmp_path: Path, home: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "rl.md", "# Reward shaping\n\nReward shaping speeds up learning.\n")
    write(root / "plain.txt", "plain text\n")
    write(root / "robots" / "brush.md", "# Brush\n\nSee [[rl]].\n")
    write(root / ".hidden" / "secret.md", "# Secret\n")
    add(home, AddCollection(str(root)))
    return root


def indexed(home: Path, *names: str, report: Callable[[Event], None] = lambda _: None) -> IndexResult:
    return index(home, IndexCollections(names), report)


def outcome(home: Path, *names: str) -> Indexed | Skipped:
    (only,) = indexed(home, *names).collections
    return only.outcome


class Titled(NamedTuple):
    path: str
    title: str


def unreadable_files(home: Path) -> tuple[UnreadableFile, ...]:
    (shown,) = status(home, ShowStatus(unreadable=True)).collections
    assert isinstance(shown.contents, Counted)
    assert shown.contents.unreadable == len(shown.contents.unreadable_files)
    return shown.contents.unreadable_files


def titles(home: Path, name: str = "notes", identity: int = 1) -> dict[str, str]:
    (shown,) = [each for each in status(home, ShowStatus()).collections if each.name == name]
    store = open_store(Path(shown.store), CollectionId(identity))
    with contextlib.closing(store) as connection, reading(connection) as transaction:
        found = rows(transaction, Titled, "SELECT s.path, d.title FROM documents d JOIN sources s ON s.id = d.source")
        return {each.path: each.title for each in found}


@pytest.mark.usefixtures("notes")
def test_a_collection_s_text_and_markdown_files_are_read(home: Path) -> None:
    assert outcome(home) == Indexed(files=3, added=3, changed=0, removed=0, unreadable=0)
    assert titles(home) == {"plain.txt": "plain", "rl.md": "Reward shaping", "robots/brush.md": "Brush"}


def test_indexing_says_what_it_does_as_it_goes(home: Path, notes: Path) -> None:
    said: list[Event] = []
    indexed(home, report=said.append)
    assert said == [
        CollectionStarted("notes", str(notes.resolve())),
        PhaseStarted(Phase.SCAN, None),
        PhaseDone(3),
        PhaseStarted(Phase.READ, 3),
        Advanced(1),
        Advanced(2),
        Advanced(3),
        PhaseDone(3),
        Finished(3, 0, 0, 0),
    ]


def test_indexing_again_reads_only_what_changed(home: Path, notes: Path) -> None:
    indexed(home)
    write(notes / "rl.md", "# Reward shaping\n\nChanged.\n")
    (notes / "plain.txt").unlink()
    write(notes / "new.md", "# New\n")
    os.utime(notes / "robots" / "brush.md", ns=(10**18, 10**18))
    assert outcome(home) == Indexed(files=3, added=1, changed=1, removed=1, unreadable=0)
    assert set(titles(home)) == {"rl.md", "robots/brush.md", "new.md"}
    assert outcome(home) == Indexed(files=3, added=0, changed=0, removed=0, unreadable=0)


@pytest.mark.usefixtures("notes")
def test_files_are_written_in_batches(home: Path) -> None:
    said: list[Event] = []
    index(home, IndexCollections(), said.append, Limits(most_files_a_write=2))
    assert len(titles(home)) == 3
    assert said.count(Advanced(3)) == 1


def test_titles_come_from_file_names_if_the_collection_says_so(tmp_path: Path, home: Path) -> None:
    vault = tmp_path / "vault"
    write(vault / "Hans Müller.md", "## Summary\n")
    add(home, AddCollection(str(vault), title_from=TitleFrom.NAME))
    indexed(home)
    assert titles(home, "vault") == {"Hans Müller.md": "Hans Müller"}


@pytest.mark.usefixtures("notes")
def test_only_the_collections_named_are_read_and_an_unknown_name_stops_all(tmp_path: Path, home: Path) -> None:
    write(tmp_path / "other" / "a.md", "# A\n")
    add(home, AddCollection(str(tmp_path / "other")))
    assert [each.name for each in indexed(home, "other", "OTHER").collections] == ["other"]
    with pytest.raises(NotFoundError, match="no collection is named 'nope'"):
        indexed(home, "notes", "nope")
    assert titles(home) == {}


def test_a_collection_whose_folder_is_missing_is_skipped_and_kept(home: Path, notes: Path) -> None:
    indexed(home)
    notes.rename(notes.with_name("moved"))
    said: list[Event] = []
    result = indexed(home, report=said.append)
    reason = f"its folder is missing: {notes.resolve()}"
    assert result.collections == (CollectionIndexed("notes", Skipped(reason, FOLDER_MISSING_REMEDY)),)
    assert said[-1] == Noted(reason, FOLDER_MISSING_REMEDY)
    assert len(titles(home)) == 3


@pytest.mark.usefixtures("notes")
def test_a_collection_whose_store_is_missing_is_skipped(home: Path) -> None:
    (store,) = [Path(each.store) for each in status(home, ShowStatus()).collections]
    store.unlink()
    skipped = outcome(home)
    assert isinstance(skipped, Skipped)
    assert (skipped.reason, skipped.remedy) == (f"the store is missing: {store}", MISSING_REMEDY)


def test_a_folder_left_without_any_file_keeps_its_documents_until_one_is_back(home: Path, notes: Path) -> None:
    indexed(home)
    shutil.rmtree(notes)
    notes.mkdir()
    said: list[Event] = []
    assert indexed(home, report=said.append).collections[0].outcome == Indexed(0, 0, 0, 0, 0)
    assert Noted(LEFT_AS_INDEXED.format(files="3 files", folder=notes.resolve())) in said
    write(notes / "rl.md", "# Reward shaping\n\nWritten anew.\n")
    assert outcome(home) == Indexed(files=1, added=0, changed=1, removed=2, unreadable=0)


def test_a_file_that_cannot_be_read_is_counted_listed_and_read_again_later(home: Path, notes: Path) -> None:
    def vanish(event: Event) -> None:
        if event == PhaseStarted(Phase.READ, 3):
            (notes / "plain.txt").rename(notes / "away")

    assert indexed(home, report=vanish).collections[0].outcome == Indexed(3, 2, 0, 0, 1)
    assert [entry.path for entry in unreadable_files(home)] == ["plain.txt"]
    (notes / "away").rename(notes / "plain.txt")
    assert outcome(home) == Indexed(files=3, added=1, changed=0, removed=0, unreadable=0)
    assert unreadable_files(home) == ()


def test_an_unreadable_file_gone_for_good_is_listed_no_more(home: Path, notes: Path) -> None:
    def vanish(event: Event) -> None:
        if event == PhaseStarted(Phase.READ, 3):
            (notes / "plain.txt").unlink()

    indexed(home, report=vanish)
    assert len(unreadable_files(home)) == 1
    indexed(home)
    assert unreadable_files(home) == ()


def test_a_file_larger_than_a_document_may_be_is_left_out_until_it_changes(home: Path, notes: Path) -> None:
    write(notes / "huge.txt", "x" * 100)
    small = Limits(most_text_bytes=60)
    (result,) = index(home, IndexCollections(), lambda _: None, small).collections
    assert result.outcome == Indexed(files=4, added=3, changed=0, removed=0, unreadable=1)
    problems = [entry.problem for entry in unreadable_files(home)]
    assert problems == ["100 bytes: a document holds at most 60; split the file"]
    (again,) = index(home, IndexCollections(), lambda _: None, small).collections
    assert again.outcome == Indexed(files=4, added=0, changed=0, removed=0, unreadable=0)


def test_a_file_grown_too_large_loses_its_documents(home: Path, notes: Path) -> None:
    indexed(home)
    write(notes / "plain.txt", "x" * 61)
    (result,) = index(home, IndexCollections(), lambda _: None, Limits(most_text_bytes=60)).collections
    assert result.outcome == Indexed(files=3, added=0, changed=0, removed=1, unreadable=1)


def test_a_nul_in_a_file_is_stored_as_a_replacement_character(home: Path, notes: Path) -> None:
    (notes / "pasted.md").write_bytes(b"# Num\x00bers\n")
    indexed(home)
    assert titles(home)["pasted.md"] == "Num\N{REPLACEMENT CHARACTER}bers"


@pytest.mark.usefixtures("notes")
def test_a_file_s_documents_are_stored_whole(home: Path) -> None:
    indexed(home)
    (shown,) = status(home, ShowStatus()).collections
    store = open_store(Path(shown.store), CollectionId(1))
    with contextlib.closing(store) as connection, reading(connection) as transaction:
        found = stored(transaction, "robots/brush.md")
    assert found is not None
    assert found.text == "# Brush\n\nSee [[rl]].\n"
