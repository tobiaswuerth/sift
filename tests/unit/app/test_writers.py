from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.indexing import IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.storage.catalog import known_collections
from sift.storage.writer import sole_writer
from tests.files import write
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
    return index(home, IndexCollections(), lambda _: None)


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
            found = embed(home, EmbedCollections(reset=reset), lambda _: None, Hashed())
        (skipped,) = found.collections
        assert isinstance(skipped.outcome, Skipped)
        assert "another sift writes it now (sift index, process " in skipped.outcome.message
