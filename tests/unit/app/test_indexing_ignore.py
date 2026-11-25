from collections.abc import Callable
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.errors import Problem
from sift.api.events import Event, Noted
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import add
from sift.app.documents import documents
from sift.app.filing import LIMITS
from sift.app.indexing import IGNORE_REMEDY, index
from sift.app.problems import LEFT_AS_INDEXED
from sift.ingest.ignore import IGNORE_FILE
from tests.files import write
from tests.homes import asked, quiet
from tests.later import LATER

PATHS = ("notes/rl.md", "notes/plain.txt", "notes/robots/brush.md", "other/x.md")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def notes(tmp_path: Path, home: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "rl.md", "# Reward shaping\n")
    write(root / "plain.txt", "plain text\n")
    write(root / "robots" / "brush.md", "# Brush\n")
    add(home, AddCollection(str(root)))
    return root


def indexed(home: Path, report: Callable[[Event], None] = quiet) -> IndexResult:
    return index(home, IndexCollections(()), report, LIMITS, LATER)


def held(home: Path) -> list[str]:
    return [each.id for each in documents(home, asked(*PATHS)).found]


def test_the_index_follows_siftignore_as_it_changes(home: Path, notes: Path) -> None:
    indexed(home)
    write(notes / IGNORE_FILE, "# drafts and exports\nrobots/\n*.txt\n")
    assert indexed(home).collections == (CollectionIndexed("notes", Indexed(1, 0, 0, 2, 0)),)
    assert held(home) == ["notes/rl.md"]
    write(notes / IGNORE_FILE, "robots/\n")
    assert indexed(home).collections == (CollectionIndexed("notes", Indexed(2, 1, 0, 0, 0)),)
    assert held(home) == ["notes/rl.md", "notes/plain.txt"]


def test_a_collection_whose_siftignore_cannot_be_read_is_skipped_and_keeps_its_documents(
    tmp_path: Path, home: Path, notes: Path
) -> None:
    indexed(home)
    write(tmp_path / "other" / "x.md", "# X\n")
    add(home, AddCollection(str(tmp_path / "other")))
    (notes / IGNORE_FILE).mkdir()
    said: list[Event] = []
    skipped, other = indexed(home, said.append).collections
    reason = f"its {IGNORE_FILE} can't be read (it is not a file)"
    assert skipped == CollectionIndexed("notes", Skipped(Problem.OS, reason, IGNORE_REMEDY))
    assert Noted(reason, IGNORE_REMEDY, Problem.OS) in said
    assert other == CollectionIndexed("other", Indexed(1, 1, 0, 0, 0))
    assert held(home) == list(PATHS)


def test_a_siftignore_that_leaves_out_every_file_drops_them_all(home: Path, notes: Path) -> None:
    indexed(home)
    write(notes / IGNORE_FILE, "*.md\n*.txt\n")
    said: list[Event] = []
    assert indexed(home, said.append).collections == (CollectionIndexed("notes", Indexed(0, 0, 0, 3, 0)),)
    assert held(home) == []
    kept = LEFT_AS_INDEXED.split("{")[0]
    assert not [each for each in said if isinstance(each, Noted) and each.text.startswith(kept)]
