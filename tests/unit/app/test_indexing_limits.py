import shutil
import tracemalloc
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.events import Event, Finished, Phase, PhaseDone, PhaseStarted
from sift.api.indexing import IndexCollections, Indexed
from sift.app.collections import add
from sift.app.filing import Limits, Readers
from sift.app.indexing import index
from sift.ingest.files import CHANGED, Identity, pieces_of
from sift.storage.database import Batches
from tests.files import write
from tests.homes import text_of, unreadable_files
from tests.later import LATER, later
from tests.mails import sample

SMALL = Limits(most_file_bytes=2000)
TOO_LARGE = "sift reads documents of at most 2,000 bytes; split the file"


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def notes(tmp_path: Path, home: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "rl.md", "# Reward shaping\n\nReward shaping speeds up learning.\n")
    write(root / "plain.txt", "plain text\n")
    write(root / "robots" / "brush.md", "# Brush\n\nSee [[rl]].\n")
    add(home, AddCollection(str(root)))
    return root


def indexed(home: Path, limits: Limits = SMALL, readers: Readers = LATER, said: list[Event] | None = None) -> Indexed:
    heard: list[Event] = [] if said is None else said
    (result,) = index(home, IndexCollections(), heard.append, limits, readers).collections
    assert isinstance(result.outcome, Indexed)
    return result.outcome


def test_a_document_larger_than_sift_reads_is_left_out_until_it_changes(home: Path, notes: Path) -> None:
    shutil.copy(sample(".docx"), notes / "huge.docx")
    assert indexed(home) == Indexed(files=4, added=3, changed=0, removed=0, unreadable=1)
    assert [(each.path, each.problem) for each in unreadable_files(home)] == [
        ("huge.docx", f"36,749 bytes: {TOO_LARGE}")
    ]
    said: list[Event] = []
    assert indexed(home, said=said) == Indexed(files=4, added=0, changed=0, removed=0, unreadable=1)
    assert (PhaseStarted(Phase.READ, 0) in said, said[-1]) == (True, Finished(0, 0, 0, 1))


def test_a_document_grown_too_large_loses_its_documents(home: Path, notes: Path) -> None:
    shutil.copy(sample(".pdf"), notes / "a.pdf")
    indexed(home)
    shutil.copy(sample(".docx"), notes / "a.pdf")
    assert indexed(home) == Indexed(files=4, added=0, changed=0, removed=1, unreadable=1)


def test_a_document_grown_too_large_after_the_scan_is_read_no_further_than_the_limit(home: Path, notes: Path) -> None:
    shutil.copy(sample(".pdf"), notes / "a.pdf")
    asked: list[int] = []

    def grown(_path: Path, most: int, _scanned: Identity | None) -> bytes:
        asked.append(most)
        return b"x" * (most + 1)

    assert indexed(home, readers=Readers(whole=grown, clock=later)).unreadable == 1
    assert asked == [2000]
    assert [each.problem for each in unreadable_files(home)] == [f"2,001 bytes: {TOO_LARGE}"]


def test_a_note_swapped_after_the_scan_is_not_read_but_read_again_at_the_next_index(
    home: Path, notes: Path, tmp_path: Path
) -> None:
    other = write(tmp_path / "other.md", "# Elsewhere\n\nswapped in\n")

    @contextmanager
    def swapped(path: Path, scanned: Identity | None) -> Generator[Iterable[bytes]]:
        if path.name == "rl.md":
            other.replace(path)
        with pieces_of(path, scanned) as parts:
            yield parts

    assert indexed(home, readers=Readers(pieces=swapped, clock=later)) == Indexed(
        files=3, added=2, changed=0, removed=0, unreadable=1
    )
    assert [(each.path, each.problem) for each in unreadable_files(home)] == [("rl.md", CHANGED)]
    assert indexed(home) == Indexed(files=3, added=1, changed=0, removed=0, unreadable=0)
    assert (notes / "rl.md").read_text(encoding="utf-8").startswith("# Elsewhere")


def test_a_note_that_cannot_be_read_to_its_end_keeps_what_it_held_and_is_read_again(home: Path, notes: Path) -> None:
    indexed(home)
    write(notes / "rl.md", "# Reward shaping\n\nChanged.\n")

    @contextmanager
    def broken(path: Path, scanned: Identity | None) -> Generator[Iterable[bytes]]:
        def parts() -> Iterable[bytes]:
            yield b"# Reward"
            raise OSError(5, "the drive went away")

        with pieces_of(path, scanned):
            yield parts()

    assert indexed(home, readers=Readers(pieces=broken, clock=later)).unreadable == 1
    assert text_of(home, "notes/rl.md") == "# Reward shaping\n\nReward shaping speeds up learning.\n"
    assert indexed(home) == Indexed(files=3, added=0, changed=1, removed=0, unreadable=0)


def test_a_note_touched_but_unchanged_is_left_as_it_was(home: Path, notes: Path) -> None:
    indexed(home)
    write(notes / "rl.md", "# Reward shaping\n\nReward shaping speeds up learning.\n")
    assert indexed(home) == Indexed(files=3, added=0, changed=0, removed=0, unreadable=0)


def test_a_note_of_any_size_is_read_in_pieces_in_bounded_memory(home: Path, notes: Path) -> None:
    line = "Reward shaping speeds up learning, and brushes paint strokes.\n"
    count = (24 << 20) // len(line)
    (notes / "large.md").write_text("# Large\n\n" + line * count + "the last word is ultramarine\n", encoding="utf-8")
    small = Batches(chunks_a_segment=100, postings_a_merge=1000, bytes_a_merge=64 << 10, chunks_a_settle=2000)
    phases: list[Phase] = []
    peaks: dict[Phase, int] = {}

    def measured(event: Event) -> None:
        match event:
            case PhaseStarted():
                phases.append(event.phase)
                tracemalloc.reset_peak()
            case PhaseDone():
                peaks[phases[-1]] = tracemalloc.get_traced_memory()[1]
            case _:
                pass

    tracemalloc.start()
    try:
        index(home, IndexCollections(), measured, Limits(batches=small), LATER)
    finally:
        tracemalloc.stop()
    assert peaks[Phase.READ] < 16 << 20
    assert text_of(home, "notes/large.md").endswith("the last word is ultramarine\n")
