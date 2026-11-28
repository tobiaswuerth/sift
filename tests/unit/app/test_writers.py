import contextlib
import dataclasses
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.events import Advanced, Event, FilesKept
from sift.api.indexing import IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.filing import READERS
from sift.app.indexing import index
from sift.domain.text.documents import Outcome
from sift.ingest.readers import ToRead
from sift.ingest.sandbox import FileReading
from sift.storage.catalog import known_collections
from sift.storage.store import read_store
from sift.storage.vector_meta import left
from sift.storage.writer import sole_writer
from tests.files import write
from tests.homes import quiet
from tests.keyword_stores import vectors_by_text
from tests.models import Hashed


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "rl.md", "# Reward shaping\n\nReward shaping speeds up learning.\n")
    write(root / "plain.txt", "plain text\n")
    write(root / "robots" / "brush.md", "# Brush\n\nSee [[rl]].\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(root)))
    return home


def indexed(home: Path) -> IndexResult:
    return index(home, IndexCollections(), quiet)


def test_a_collection_another_sift_writes_now_is_skipped_at_once_saying_so(home: Path) -> None:
    (one,) = known_collections(home)
    with sole_writer(one.store, "sift embed"):
        found = indexed(home)
    (skipped,) = found.collections
    assert isinstance(skipped.outcome, Skipped)
    assert skipped.outcome.message.startswith("another sift writes it now (sift embed, process ")
    assert indexed(home).collections[0].outcome == Indexed(3, 3, 0, 0, 0)


def test_a_collection_another_sift_writes_now_is_not_embedded_and_said_to_be_busy(home: Path) -> None:
    indexed(home)
    (one,) = known_collections(home)
    for reset in (False, True):
        with sole_writer(one.store, "sift index"):
            found = embed(home, EmbedCollections(reset=reset), quiet, Hashed())
        (skipped,) = found.collections
        assert isinstance(skipped.outcome, Skipped)
        assert "another sift writes it now (sift index, process " in skipped.outcome.message


def test_a_file_renamed_or_moved_keeps_the_vectors_of_its_passages(home: Path, tmp_path: Path) -> None:
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    (one,) = known_collections(home)
    with read_store(one.store, one.id) as transaction:
        before = vectors_by_text(transaction)
    (tmp_path / "notes" / "rl.md").rename(tmp_path / "notes" / "robots" / "reward.md")
    assert indexed(home).collections[0].outcome == Indexed(3, 1, 0, 1, 0)
    with read_store(one.store, one.id) as transaction:
        assert (left(transaction), vectors_by_text(transaction)) == (0, before)


@contextlib.contextmanager
def stopped_reading() -> Generator[FileReading]:
    def read(asked: Sequence[ToRead]) -> Iterator[tuple[int, Outcome]]:
        raise KeyboardInterrupt(len(asked))

    yield read


def test_a_file_renamed_keeps_its_vectors_though_the_index_that_found_it_renamed_is_stopped(
    home: Path, tmp_path: Path
) -> None:
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    (one,) = known_collections(home)
    with read_store(one.store, one.id) as transaction:
        before = vectors_by_text(transaction)
    (tmp_path / "notes" / "rl.md").rename(tmp_path / "notes" / "robots" / "reward.md")
    with pytest.raises(KeyboardInterrupt):
        index(home, IndexCollections(), quiet, readers=dataclasses.replace(READERS, files=stopped_reading))
    assert indexed(home).collections[0].outcome == Indexed(3, 1, 0, 1, 0)
    with read_store(one.store, one.id) as transaction:
        assert (left(transaction), vectors_by_text(transaction)) == (0, before)


@pytest.mark.parametrize("named", [("rl.md", "robots/reward.md"), ("robots/reward.md", "rl.md")])
def test_a_file_renamed_keeps_its_vectors_when_both_its_names_are_read_now_in_either_order(
    home: Path, tmp_path: Path, named: tuple[str, str]
) -> None:
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    (one,) = known_collections(home)
    with read_store(one.store, one.id) as transaction:
        before = vectors_by_text(transaction)
    (tmp_path / "notes" / "rl.md").rename(tmp_path / "notes" / "robots" / "reward.md")
    files = tuple(f"notes/{name}" for name in named)
    states = {each.id: each.state for each in index(home, IndexCollections(files=files), quiet).collections[0].named}
    assert states == {"notes/rl.md": "removed", "notes/robots/reward.md": "added"}
    with read_store(one.store, one.id) as transaction:
        assert (left(transaction), vectors_by_text(transaction)) == (0, before)


def test_files_read_now_are_counted_one_after_another(home: Path, tmp_path: Path) -> None:
    indexed(home)
    for name in ("rl.md", "plain.txt", "robots/brush.md"):
        write(tmp_path / "notes" / name, f"# Changed\n\n{name} reads otherwise now.\n")
    events = list[Event]()
    files = ("notes/rl.md", "notes/plain.txt", "notes/robots/brush.md")
    index(home, IndexCollections(files=files), events.append)
    assert [each.done for each in events if isinstance(each, Advanced)] == [1, 2, 3]
    assert [each.files for each in events if isinstance(each, FilesKept)] == [1, 2, 3]
