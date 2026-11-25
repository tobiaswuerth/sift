from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.indexing import IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
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
