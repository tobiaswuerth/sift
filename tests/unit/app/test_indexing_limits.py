from pathlib import Path

import pytest

from sift.api.collections import AddCollection, UnreadableFile
from sift.api.events import Event, Finished, Phase, PhaseStarted
from sift.api.indexing import IndexCollections, Indexed
from sift.app.collections import add
from sift.app.filing import Limits, Readers
from sift.app.indexing import index
from sift.ingest.files import CHANGED, Identity, at_most
from tests.files import write
from tests.homes import unreadable_files
from tests.later import LATER, later

SMALL = Limits(most_text_bytes=60)
TOO_LARGE = "a document holds at most 60; split the file"


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


def test_a_file_larger_than_a_document_may_be_is_left_out_until_it_changes(home: Path, notes: Path) -> None:
    write(notes / "huge.txt", "x" * 100)
    assert indexed(home) == Indexed(files=4, added=3, changed=0, removed=0, unreadable=1)
    assert unreadable_files(home) == (UnreadableFile("huge.txt", f"100 bytes: {TOO_LARGE}"),)
    said: list[Event] = []
    assert indexed(home, said=said) == Indexed(files=4, added=0, changed=0, removed=0, unreadable=1)
    assert (PhaseStarted(Phase.READ, 0) in said, said[-1]) == (True, Finished(0, 0, 0, 1))


def test_a_file_that_is_too_large_only_once_decoded_is_left_out_until_it_changes(home: Path, notes: Path) -> None:
    (notes / "wide.txt").write_bytes(
        "\N{BYTE ORDER MARK}".encode("utf-16-le") + ("\N{CJK UNIFIED IDEOGRAPH-65E5}" * 25).encode("utf-16-le")
    )
    assert indexed(home) == Indexed(files=4, added=3, changed=0, removed=0, unreadable=1)
    assert unreadable_files(home) == (UnreadableFile("wide.txt", f"75 bytes as text: {TOO_LARGE}"),)
    said: list[Event] = []
    indexed(home, said=said)
    assert PhaseStarted(Phase.READ, 0) in said


def test_a_file_grown_too_large_loses_its_documents(home: Path, notes: Path) -> None:
    indexed(home, Limits())
    write(notes / "plain.txt", "x" * 61)
    assert indexed(home) == Indexed(files=3, added=0, changed=0, removed=1, unreadable=1)


@pytest.mark.usefixtures("notes")
def test_a_file_grown_too_large_after_the_scan_is_read_no_further_than_the_limit(home: Path) -> None:
    asked: list[int] = []

    def grown(_path: Path, most: int, _scanned: Identity | None) -> bytes:
        asked.append(most)
        return b"x" * (most + 1)

    assert indexed(home, readers=Readers(whole=grown, clock=later)) == Indexed(
        files=3, added=0, changed=0, removed=0, unreadable=3
    )
    assert asked == [60, 60, 60]
    assert {entry.problem for entry in unreadable_files(home)} == {f"more than 60 bytes: {TOO_LARGE}"}


def test_a_file_swapped_after_the_scan_is_not_read_but_read_again_at_the_next_index(
    home: Path, notes: Path, tmp_path: Path
) -> None:
    other = write(tmp_path / "other.md", "# Elsewhere\n\nswapped in\n")

    def swapped(path: Path, most: int, scanned: Identity | None) -> bytes:
        if path.name == "rl.md":
            other.replace(path)
        return at_most(path, most, scanned)

    assert indexed(home, readers=Readers(whole=swapped, clock=later)) == Indexed(
        files=3, added=2, changed=0, removed=0, unreadable=1
    )
    assert unreadable_files(home) == (UnreadableFile("rl.md", CHANGED),)
    assert indexed(home) == Indexed(files=3, added=1, changed=0, removed=0, unreadable=0)
    assert (notes / "rl.md").read_text(encoding="utf-8").startswith("# Elsewhere")
