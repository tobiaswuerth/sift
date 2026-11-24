import contextlib
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.collections import AddCollection, Counted, ShowStatus, TitleFrom, UnreadableFile
from sift.api.documents import GetDocuments
from sift.api.errors import NotFoundError
from sift.api.events import Advanced, CollectionStarted, Event, Finished, Noted, Phase, PhaseDone, PhaseStarted
from sift.api.ids import CollectionId
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import add, status
from sift.app.documents import documents
from sift.app.filing import LIMITS, Limits, Readers
from sift.app.indexing import FOLDER_MISSING_REMEDY, OUT_OF_REACH_REMEDY, index
from sift.app.problems import EMPTY_INSIDE, EMPTY_INSIDE_REMEDY, LEFT_AS_INDEXED
from sift.domain.plan.segments import Policy
from sift.storage.database import Batches, reading, rows
from sift.storage.documents import stored
from sift.storage.keywords import settle
from sift.storage.store import MISSING_REMEDY, open_store
from sift.storage.texts import text_of
from sift.storage.unusable import DAMAGED_REMEDY
from tests.damage import damage_table
from tests.files import write
from tests.later import LATER


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
    return index(home, IndexCollections(names), report, LIMITS, LATER)


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
    assert (skipped.message, skipped.remedy) == (f"the store is missing: {store}", MISSING_REMEDY)


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


def test_a_file_that_could_not_be_read_is_read_again_though_it_is_as_it_was(home: Path, notes: Path) -> None:
    plain = notes / "plain.txt"
    indexed(home)
    as_it_was = plain.stat().st_mtime_ns

    def vanish(event: Event) -> None:
        if event == PhaseStarted(Phase.READ, 1):
            plain.rename(notes / "away")

    write(plain, "changed and then out of reach\n")
    assert indexed(home, report=vanish).collections[0].outcome == Indexed(3, 0, 0, 0, 1)
    (notes / "away").unlink()
    write(plain, "plain text\n")
    os.utime(plain, ns=(as_it_was, as_it_was))
    assert outcome(home) == Indexed(files=3, added=0, changed=0, removed=0, unreadable=0)
    assert titles(home)["plain.txt"] == "plain"


@pytest.mark.usefixtures("notes")
def test_files_are_counted_as_they_are_read_before_their_batch_is_written(home: Path) -> None:
    held: list[tuple[int, int]] = []

    def count(event: Event) -> None:
        if isinstance(event, Advanced):
            held.append((event.done, len(titles(home))))

    index(home, IndexCollections(), count, Limits(most_files_a_write=2))
    assert held == [(1, 0), (2, 0), (3, 2)]


@pytest.mark.usefixtures("notes")
def test_an_index_interrupted_keeps_the_batches_written_and_goes_on_the_next_time(home: Path) -> None:
    def interrupt(event: Event) -> None:
        if event == Advanced(3):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        index(home, IndexCollections(), interrupt, Limits(most_files_a_write=2))
    assert len(titles(home)) == 2
    assert outcome(home) == Indexed(files=3, added=1, changed=0, removed=0, unreadable=0)


def test_a_folder_inside_that_is_empty_keeps_its_documents_until_it_is_deleted(home: Path, notes: Path) -> None:
    indexed(home)
    (notes / "robots" / "brush.md").unlink()
    said: list[Event] = []
    assert indexed(home, report=said.append).collections[0].outcome == Indexed(2, 0, 0, 0, 0)
    assert Noted(EMPTY_INSIDE.format(folder="robots/", files="1 file"), EMPTY_INSIDE_REMEDY) in said
    assert "robots/brush.md" in titles(home)
    (notes / "robots").rmdir()
    assert outcome(home) == Indexed(files=2, added=0, changed=0, removed=1, unreadable=0)


def two_collections(tmp_path: Path, home: Path) -> None:
    for name in ("a", "b"):
        write(tmp_path / name / "note.md", f"# {name}\n")
        add(home, AddCollection(str(tmp_path / name)))


def test_a_store_damaged_where_indexing_reads_it_is_skipped_and_the_others_are_indexed(
    tmp_path: Path, home: Path
) -> None:
    two_collections(tmp_path, home)
    damage_table(home / "collections" / "a.sqlite", "sources")
    damaged, sound = indexed(home).collections
    assert isinstance(damaged.outcome, Skipped)
    assert (damaged.outcome.message.startswith("the store "), damaged.outcome.remedy) == (True, DAMAGED_REMEDY)
    assert "is damaged" in damaged.outcome.message
    assert sound == CollectionIndexed("b", Indexed(1, 1, 0, 0, 0))


def test_a_collection_out_of_reach_is_skipped_and_the_others_are_indexed(tmp_path: Path, home: Path) -> None:
    two_collections(tmp_path, home)

    def denied_while_scanning(event: Event) -> None:
        if event == PhaseDone(1) and not seen:
            seen.append(event)
            raise PermissionError(13, "Access is denied")

    seen: list[Event] = []
    out_of_reach, sound = indexed(home, report=denied_while_scanning).collections
    assert out_of_reach == CollectionIndexed("a", Skipped("it is out of reach (Access is denied)", OUT_OF_REACH_REMEDY))
    assert sound == CollectionIndexed("b", Indexed(1, 1, 0, 0, 0))


def test_indexing_where_there_is_no_catalog_makes_none(home: Path) -> None:
    assert indexed(home) == IndexResult(())
    with pytest.raises(NotFoundError, match="no collection is named 'notes'"):
        indexed(home, "notes")
    assert not home.exists()


@pytest.mark.usefixtures("notes")
def test_a_store_that_cannot_be_written_is_passed_over_and_the_next_collection_indexed(
    home: Path, tmp_path: Path
) -> None:
    write(tmp_path / "more" / "a.md", "# A\n")
    add(home, AddCollection(str(tmp_path / "more")))
    store = home / "collections" / "notes.sqlite"
    store.chmod(stat.S_IREAD)
    try:
        found = {each.name: each.outcome for each in indexed(home).collections}
    finally:
        store.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert isinstance(found["more"], Indexed)
    assert isinstance(found["notes"], Skipped)
    assert found["notes"].message.startswith(f"the store {store.resolve()} cannot be written")


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
        assert text_of(transaction, found.id) == "# Brush\n\nSee [[rl]].\n"


def test_merges_too_large_for_a_write_run_as_their_own_phase(home: Path, notes: Path) -> None:
    write(notes / "more.md", "# More\n\nOne more note.\n")
    said: list[Event] = []
    batches = Batches(chunks_a_settle=0, merging=Policy(factor=2, floor=1, span=0.0))
    limits = Limits(most_files_a_write=1, batches=batches)
    index(home, IndexCollections(), said.append, limits)
    assert [event.phase for event in said if isinstance(event, PhaseStarted)] == [Phase.SCAN, Phase.READ, Phase.MERGE]
    assert [event for event in said if isinstance(event, PhaseDone)][-1] == PhaseDone(8)
    assert isinstance(said[-1], Finished)
    again: list[Event] = []
    index(home, IndexCollections(), again.append, limits)
    assert Phase.MERGE not in [event.phase for event in again if isinstance(event, PhaseStarted)]


def test_merges_left_by_a_remake_run_though_the_folder_is_gone(home: Path, notes: Path) -> None:
    batches = Batches(chunks_a_segment=1, chunks_a_settle=0, merging=Policy(factor=2, floor=1, span=0.0))
    index(home, IndexCollections(), lambda _: None, Limits(batches=batches))
    (store,) = [Path(each.store) for each in status(home, ShowStatus()).collections]
    with contextlib.closing(open_store(store, CollectionId(1))) as connection, connection:
        connection.execute("UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
    shutil.rmtree(notes)
    said: list[Event] = []
    held: list[int] = []

    def seen(event: Event) -> None:
        said.append(event)
        if isinstance(event, PhaseStarted) and event.phase is Phase.MERGE:
            held.append(segments(store))

    index(home, IndexCollections(), seen, Limits(batches=batches))
    assert [event.phase for event in said if isinstance(event, PhaseStarted)] == [Phase.REMAKE, Phase.MERGE]
    assert isinstance(said[-1], Noted)
    assert held[0] > segments(store)


def test_a_merge_phase_another_sift_settled_meanwhile_still_ends(home: Path, notes: Path) -> None:
    write(notes / "more.md", "# More\n\nOne more note.\n")
    batches = Batches(chunks_a_settle=0, merging=Policy(factor=2, floor=1, span=0.0))
    said: list[Event] = []

    def seen(event: Event) -> None:
        said.append(event)
        if isinstance(event, PhaseStarted) and event.phase is Phase.MERGE:
            (store,) = [Path(each.store) for each in status(home, ShowStatus()).collections]
            with contextlib.closing(open_store(store, CollectionId(1))) as connection:
                assert list(settle(connection, batches))[-1] > 0

    index(home, IndexCollections(), seen, Limits(most_files_a_write=1, batches=batches))
    phases = [event for event in said if isinstance(event, (PhaseStarted, PhaseDone))]
    assert phases[-2:] == [PhaseStarted(Phase.MERGE, None), PhaseDone(0)]


class Count(NamedTuple):
    n: int


def segments(store: Path) -> int:
    with contextlib.closing(open_store(store, CollectionId(1))) as connection, reading(connection) as transaction:
        (found,) = rows(transaction, Count, "SELECT count(*) FROM keyword_segments")
        return found.n


def test_a_file_rewritten_within_the_grain_of_its_scan_is_read_again(home: Path, notes: Path) -> None:
    tick = notes / "tick.md"
    write(tick, "# Tick\n\nalpha\n")
    stamp = tick.stat().st_mtime_ns
    index(home, IndexCollections(), lambda _: None, LIMITS, Readers(clock=lambda: stamp))
    write(tick, "# Tick\n\nomega\n")
    os.utime(tick, ns=(stamp, stamp))
    index(home, IndexCollections(), lambda _: None, LIMITS, LATER)
    (found,) = documents(home, GetDocuments(("notes/tick.md",))).found
    assert "omega" in found.text
