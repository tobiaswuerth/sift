import contextlib
import sqlite3
from pathlib import Path

import pytest

from sift.api.embedding import EmbedCollections, Embedded
from sift.api.errors import Problem, StoreUnavailableError
from sift.api.importing import Brought, ImportHome
from sift.api.indexing import Skipped
from sift.app.embedding import embed
from sift.app.importing import imported
from sift.storage.catalog import known_collections
from sift.storage.former import former_store
from sift.storage.writer import sole_writer
from tests.damage import damage_table
from tests.files import write
from tests.former_homes import former_home, hashed
from tests.homes import quiet

NOTES = {
    "rl.md": "# Reward shaping\n\nReward shaping speeds up reinforcement learning.\n",
    "brush.md": "# Brush\n\nA painting robot plans its brush strokes.\n\n## Paint\n\nPigments mix on canvas.\n",
}


@pytest.fixture
def former(tmp_path: Path) -> Path:
    for name, text in NOTES.items():
        write(tmp_path / "notes" / name, text)
    return former_home(tmp_path, {"notes": tmp_path / "notes"}, hashed())


def changed(former: Path, *statements: str) -> None:
    with contextlib.closing(sqlite3.connect(former / "collections" / "notes.sqlite")) as connection, connection:
        for statement in statements:
            connection.execute(statement)


def brought(home: Path, former: Path) -> Brought | Skipped:
    (one,) = imported(home, ImportHome(str(former)), quiet).collections
    return one.outcome


def test_what_a_store_of_sift_1_held_is_what_it_counted_not_what_reading_all_its_vectors_counts(
    former: Path, tmp_path: Path
) -> None:
    changed(former, "UPDATE counts SET n = 7 WHERE name = 'vectors'", "UPDATE counts SET n = 5 WHERE name = 'docs'")
    done = brought(tmp_path / "home", former)
    assert isinstance(done, Brought)
    assert (done.documents_before, done.passages_before, done.vectors_before, done.vectors) == (5, 2, 7, 2)


def test_a_passage_is_embedded_in_sift_1_where_it_has_a_code_as_sift_1_15_itself_holds(
    former: Path, tmp_path: Path
) -> None:
    changed(former, "DELETE FROM codes WHERE chunk = (SELECT min(chunk) FROM codes)")
    home = tmp_path / "home"
    done = brought(home, former)
    assert isinstance(done, Brought)
    assert (done.vectors, embed(home, EmbedCollections(), quiet, hashed()).collections[0].outcome) == (
        1,
        Embedded(1, 0),
    )


def test_a_code_of_sift_1_whose_vector_is_gone_leaves_its_passage_to_embed_and_the_others_carried(
    former: Path, tmp_path: Path
) -> None:
    changed(
        former,
        "DROP TRIGGER IF EXISTS codes_go_with_vectors",
        "DELETE FROM vectors WHERE chunk = (SELECT min(chunk) FROM vectors)",
    )
    done = brought(tmp_path / "home", former)
    assert isinstance(done, Brought)
    assert (done.passages, done.vectors) == (2, 1)


def test_vectors_of_sift_1_of_several_lengths_are_damage_of_its_store(former: Path, tmp_path: Path) -> None:
    changed(former, "UPDATE vectors SET vec = zeroblob(length(vec) / 2) WHERE chunk = (SELECT min(chunk) FROM vectors)")
    done = brought(tmp_path / "home", former)
    assert isinstance(done, Skipped)
    store = former / "collections" / "notes.sqlite"
    assert done.message == f"the store of sift 1.x {store} is damaged: its vectors are of several lengths"


def test_a_store_of_sift_1_without_its_counts_is_said_not_to_be_as_1_15_writes_it(former: Path, tmp_path: Path) -> None:
    changed(former, "DELETE FROM counts WHERE name = 'vectors'")
    done = brought(tmp_path / "home", former)
    assert isinstance(done, Skipped)
    assert "is not as sift 1.15 writes its stores: it counts no vectors" in done.message


def test_an_import_run_again_reads_no_passage_of_sift_1_where_none_waits_for_a_vector(
    former: Path, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    assert isinstance(brought(home, former), Brought)
    damage_table(former / "collections" / "notes.sqlite", "docs")
    again = brought(home, former)
    assert isinstance(again, Brought)
    assert (again.vectors_before, again.vectors) == (2, 2)


def test_damage_of_sift_1_s_store_found_as_its_vectors_are_carried_is_said_of_that_store(
    former: Path, tmp_path: Path
) -> None:
    damage_table(former / "collections" / "notes.sqlite", "vectors")
    done = brought(tmp_path / "home", former)
    assert isinstance(done, Skipped)
    assert done.code is Problem.STORE_UNAVAILABLE
    assert done.message.startswith(f"the store of sift 1.x {former / 'collections' / 'notes.sqlite'} ")
    assert done.remedy == "embed the collection: its files are indexed here, and its vectors were in that store alone"


def test_a_collection_inactive_in_sift_1_is_inactive_here_though_its_import_stopped_before_its_index(
    tmp_path: Path,
) -> None:
    write(tmp_path / "old" / "a.md", "# A\n\nAn old note.\n")
    former = former_home(tmp_path, {"old": tmp_path / "old"}, hashed(), inactive=("old",))
    home = tmp_path / "home"
    with sole_writer(former / "collections" / "old-v2.sqlite", "an embedding"):
        stopped = brought(home, former)
    assert isinstance(stopped, Skipped)
    assert (stopped.code, [(one.name, one.active) for one in known_collections(home)]) == (
        Problem.BUSY,
        [("old", False)],
    )


def test_a_store_of_sift_1_is_read_as_it_was_when_its_import_began(former: Path) -> None:
    store = former / "collections" / "notes.sqlite"
    changed(former, "PRAGMA journal_mode = WAL")
    with former_store(store) as old:
        before = old.counts()
        changed(former, "UPDATE counts SET n = n + 40 WHERE name = 'vectors'")
        assert old.counts() == before
    with former_store(store) as later:
        assert later.counts().vectors == before.vectors + 40


def test_a_store_of_sift_1_damaged_where_its_passages_lie_is_said_as_they_are_read(former: Path) -> None:
    store = former / "collections" / "notes.sqlite"
    damage_table(store, "chunks")
    with pytest.raises(StoreUnavailableError, match=r"the store of sift 1\.x"), former_store(store) as old:
        list(old.passages())
