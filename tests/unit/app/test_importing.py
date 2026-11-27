import contextlib
import json
import sqlite3
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from sift.api.collections import AddCollection, TitleFrom
from sift.api.embedding import EmbedCollections, Embedded
from sift.api.errors import NotFoundError, Problem, StoreUnavailableError
from sift.api.importing import Brought, ImportHome
from sift.api.indexing import Skipped
from sift.api.search import SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.importing import imported
from sift.app.vsearch import vsearch
from sift.storage.catalog import known_collections
from sift.storage.former import former_collections, former_store
from sift.storage.store_files import FORMER_CATALOG
from tests.files import write
from tests.former_homes import PINNED_FAKE, former_home, hashed
from tests.homes import quiet

NOTES = {
    "rl.md": "# Reward shaping\n\nReward shaping speeds up reinforcement learning.\n",
    "brush.md": "# Brush\n\nA painting robot plans its brush strokes.\n\n## Paint\n\nPigments mix on canvas.\n",
}


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    for name, text in NOTES.items():
        write(tmp_path / "notes" / name, text)
    return tmp_path / "notes"


def vectors_of(store: Path) -> dict[tuple[str, int], bytes]:
    sql = """SELECT s.path, c.start, v.vector FROM vectors v JOIN chunks c ON c.id = v.chunk
             JOIN documents d ON d.id = c.document JOIN sources s ON s.id = d.source"""
    with contextlib.closing(sqlite3.connect(store)) as connection:
        found = cast("list[tuple[str, int, bytes]]", connection.execute(sql).fetchall())
    return {(path, start): vector for path, start, vector in found}


def brought(home: Path, source: Path, *names: str) -> tuple[Brought | Skipped, ...]:
    return tuple(each.outcome for each in imported(home, ImportHome(str(source), names), quiet).collections)


def test_a_home_of_sift_1_is_imported_with_its_counts_and_every_vector_kept(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    home = tmp_path / "home"
    (done,) = brought(home, former)
    assert isinstance(done, Brought)
    assert (done.documents_before, done.passages_before, done.vectors_before) == (2, 2, 2)
    assert (done.documents, done.passages, done.vectors, done.model) == (2, 2, 2, PINNED_FAKE)
    (one,) = known_collections(home)
    assert (one.store, Path(done.store)) == (former / "collections" / "notes-v2.sqlite", one.store)
    assert vectors_of(one.store) == vectors_of(tmp_path / "made" / "collections" / "notes.sqlite")
    assert embed(home, EmbedCollections(), quiet, hashed()).collections[0].outcome == Embedded(0, 0)
    assert [hit.id for hit in vsearch(home, SearchVectors("painting robot", k=1), hashed()).hits] == ["notes/brush.md"]


def test_the_files_of_sift_1_stay_as_they_were(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    kept = {path: path.read_bytes() for path in former.rglob("*") if path.is_file()}
    brought(tmp_path / "home", former)
    assert {path: path.read_bytes() for path in kept} == kept


def test_passages_changed_since_are_left_to_embed_and_the_others_carried(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    write(notes / "rl.md", "# Reward shaping\n\nReward shaping, now with curiosity.\n")
    home = tmp_path / "home"
    (done,) = brought(home, former)
    assert isinstance(done, Brought)
    assert (done.passages, done.vectors) == (2, 1)
    assert embed(home, EmbedCollections(), quiet, hashed()).collections[0].outcome == Embedded(1, 0)


def test_a_store_never_embedded_is_imported_without_vectors(tmp_path: Path, notes: Path) -> None:
    (done,) = brought(tmp_path / "home", former_home(tmp_path, {"notes": notes}))
    assert isinstance(done, Brought)
    assert (done.vectors_before, done.vectors, done.model) == (0, 0, None)


def test_vectors_of_a_model_named_without_a_whole_commit_are_not_carried(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    store = former / "collections" / "notes.sqlite"
    with contextlib.closing(sqlite3.connect(store)) as connection, connection:
        connection.execute("UPDATE meta SET value = 'owner/model' WHERE key IN ('model', 'model_at')")
    (done,) = brought(tmp_path / "home", former)
    assert isinstance(done, Brought)
    assert (done.vectors_before, done.vectors, done.model) == (2, 0, "owner/model")


def test_a_collection_s_choices_and_its_being_inactive_are_kept(tmp_path: Path, notes: Path) -> None:
    write(tmp_path / "papers" / "p.jsonl", json.dumps({"key": "a", "heading": "A paper", "body": "On brushes."}) + "\n")
    choices = {
        "notes": {"title_from": "name", "links": "follow"},
        "papers": {"id": "key", "title": "heading", "text": ["body"]},
    }
    former = former_home(
        tmp_path, {"notes": notes, "papers": tmp_path / "papers"}, choices=choices, inactive=("papers",)
    )
    home = tmp_path / "home"
    brought(home, former)
    held = {one.name: one for one in known_collections(home)}
    notes_choices, papers = held["notes"].choices, held["papers"]
    assert (notes_choices.title_from, notes_choices.follow_links) == (TitleFrom.NAME, True)
    records = papers.choices.records
    assert (papers.active, records.id, records.title, records.text) == (False, "key", "heading", ("body",))


def test_a_vault_that_took_its_titles_from_headings_keeps_them_and_every_vector(tmp_path: Path, notes: Path) -> None:
    (notes / ".obsidian").mkdir()
    write(
        notes / "long.md",
        "# Long\n\n" + "".join(f"## Part {n}\n\n{'Brushes and pigments. ' * 40}\n\n" for n in range(9)),
    )
    former, home = former_home(tmp_path, {"notes": notes}, hashed()), tmp_path / "home"
    (done,) = brought(home, former)
    (one,) = known_collections(home)
    assert isinstance(done, Brought)
    assert (one.choices.title_from, done.vectors) == (TitleFrom.HEADING, done.vectors_before)
    assert done.passages > len(NOTES) + 1


def test_an_import_run_again_goes_on_and_adds_nothing_twice(tmp_path: Path, notes: Path) -> None:
    former, home = former_home(tmp_path, {"notes": notes}, hashed()), tmp_path / "home"
    first = brought(home, former)
    assert brought(home, former) == first
    assert [one.name for one in known_collections(home)] == ["notes"]


def test_a_collection_of_that_name_not_imported_from_that_home_is_left_as_it_is(tmp_path: Path, notes: Path) -> None:
    former, home = former_home(tmp_path, {"notes": notes}, hashed()), tmp_path / "home"
    write(tmp_path / "other" / "x.md", "# X\n")
    add(home, AddCollection(str(tmp_path / "other"), "notes"))
    (done,) = brought(home, former)
    assert isinstance(done, Skipped)
    assert (done.code, done.collection) == (Problem.TAKEN, "notes")


def test_a_collection_named_alone_is_imported_and_one_unknown_is_refused(tmp_path: Path, notes: Path) -> None:
    write(tmp_path / "more" / "m.md", "# More\n")
    former, home = former_home(tmp_path, {"notes": notes, "more": tmp_path / "more"}), tmp_path / "home"
    assert len(brought(home, former, "more")) == 1
    assert [one.name for one in known_collections(home)] == ["more"]
    with pytest.raises(NotFoundError, match="holds no collection nope"):
        brought(home, former, "nope")


def test_a_collection_whose_folder_is_gone_is_said_skipped(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes})
    for path in notes.iterdir():
        path.unlink()
    notes.rmdir()
    (done,) = brought(tmp_path / "home", former)
    assert isinstance(done, Skipped)


def test_an_index_of_sift_older_than_1_8_is_refused_saying_what_to_do(tmp_path: Path) -> None:
    (tmp_path / "old").mkdir()
    with contextlib.closing(sqlite3.connect(tmp_path / "old" / FORMER_CATALOG)) as catalog:
        catalog.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, text TEXT)")
    with pytest.raises(StoreUnavailableError, match=r"older than 1.8") as refused:
        brought(tmp_path / "home", tmp_path / "old")
    assert (
        refused.value.remedy
        == "open it once with sift 1.15, which splits it into a store per collection, then import it"
    )


def test_a_store_of_another_layout_is_skipped(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes})
    store = former / "collections" / "notes.sqlite"
    with contextlib.closing(sqlite3.connect(store)) as connection, connection:
        connection.execute("DROP TABLE vectors")
    (done,) = brought(tmp_path / "home", former)
    assert isinstance(done, Skipped)
    assert "older than 1.8" in done.message


NO_DATABASE = b"no database at all " * 64
CATALOG_REMEDY = "add the folders of its collections to this sift, then index and embed them"
STORE_REMEDY = "embed the collection: its files are indexed here, and its vectors were in that store alone"


def test_a_catalog_of_sift_1_that_is_damaged_is_refused_saying_what_to_do(tmp_path: Path) -> None:
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / FORMER_CATALOG).write_bytes(NO_DATABASE)
    with pytest.raises(StoreUnavailableError, match=r"the catalog of sift 1\.x .* is damaged") as refused:
        brought(tmp_path / "home", tmp_path / "old")
    assert (refused.value.code, refused.value.remedy) == (Problem.STORE_UNAVAILABLE, CATALOG_REMEDY)


def test_a_catalog_of_sift_1_whose_choices_are_no_json_is_damaged(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes})
    with contextlib.closing(sqlite3.connect(former / FORMER_CATALOG)) as catalog, catalog:
        catalog.execute("UPDATE collections SET fields = '{nope'")
    with pytest.raises(StoreUnavailableError, match=r"the catalog of sift 1\.x .* is damaged") as refused:
        brought(tmp_path / "home", former)
    assert (refused.value.code, refused.value.remedy) == (Problem.STORE_UNAVAILABLE, CATALOG_REMEDY)


def test_a_store_of_sift_1_that_is_damaged_is_said_and_the_other_collections_are_imported(
    tmp_path: Path, notes: Path
) -> None:
    write(tmp_path / "papers" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    former = former_home(tmp_path, {"notes": notes, "papers": tmp_path / "papers"}, hashed())
    (former / "collections" / "notes.sqlite").write_bytes(NO_DATABASE)
    home = tmp_path / "home"
    damaged, whole = brought(home, former)
    assert isinstance(damaged, Skipped)
    assert (damaged.code, damaged.remedy) == (Problem.STORE_UNAVAILABLE, STORE_REMEDY)
    assert f"the store of sift 1.x {former / 'collections' / 'notes.sqlite'} is damaged" in damaged.message
    assert isinstance(whole, Brought)
    assert (whole.documents, whole.vectors) == (1, 1)
    assert [one.name for one in known_collections(home)] == ["notes", "papers"]


@pytest.mark.parametrize(
    "change",
    [
        "ALTER TABLE chunks DROP COLUMN pos",
        "ALTER TABLE docs DROP COLUMN title",
        "ALTER TABLE vectors RENAME COLUMN vec TO v",
        "ALTER TABLE meta RENAME COLUMN value TO held",
    ],
)
def test_a_store_of_sift_1_lacking_what_1_15_writes_is_said_and_the_other_collections_are_imported(
    tmp_path: Path, notes: Path, change: str
) -> None:
    write(tmp_path / "papers" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    former = former_home(tmp_path, {"notes": notes, "papers": tmp_path / "papers"}, hashed())
    with contextlib.closing(sqlite3.connect(former / "collections" / "notes.sqlite")) as connection, connection:
        connection.execute(change)
    older, whole = brought(tmp_path / "home", former)
    assert isinstance(older, Skipped)
    assert (
        f"{former / 'collections' / 'notes.sqlite'} is not as sift 1.15 writes its stores: no such column"
        in older.message
    )
    assert older.remedy == "open that home once with sift 1.15, which brings its stores up to date, then import it"
    assert isinstance(whole, Brought)
    assert (whole.documents, whole.vectors) == (1, 1)


def counted_while_written(store: Path) -> None:
    with former_store(store) as old, contextlib.closing(sqlite3.connect(store, isolation_level=None)) as writing:
        old.connection.execute("PRAGMA busy_timeout = 0")
        writing.execute("BEGIN EXCLUSIVE")
        old.counts()


def test_a_store_of_sift_1_that_is_written_just_then_is_said_to_be_busy(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes})
    with pytest.raises(StoreUnavailableError, match=r"another sift is writing to the store of sift 1\.x") as refused:
        counted_while_written(former / "collections" / "notes.sqlite")
    assert refused.value.remedy == "wait for it to finish, then run sift again"


def test_a_home_of_sift_1_without_collections_imports_nothing(tmp_path: Path) -> None:
    former = former_home(tmp_path, {})
    assert brought(tmp_path / "home", former) == ()


def test_a_folder_holding_no_home_of_sift_1_is_refused(tmp_path: Path) -> None:
    (tmp_path / "nothing").mkdir()
    with pytest.raises(NotFoundError, match=r"holds no index of sift 1\.x: there is no sift\.db in it") as refused:
        brought(tmp_path / "home", tmp_path / "nothing")
    assert refused.value.field == "home"


def test_a_catalog_of_sift_1_that_cannot_be_opened_is_said_to_be_unreadable(tmp_path: Path) -> None:
    with pytest.raises(StoreUnavailableError, match=r"sift\.db cannot be read") as refused:
        former_collections(tmp_path)
    assert refused.value.remedy == "check that this user may read the sift 1.x home and its stores"


def test_vectors_of_passages_whose_documents_are_gone_are_passed_over(tmp_path: Path, notes: Path) -> None:
    former, home = former_home(tmp_path, {"notes": notes}, hashed()), tmp_path / "home"
    with contextlib.closing(sqlite3.connect(former / "collections" / "notes.sqlite")) as connection, connection:
        first = "SELECT c.pos, c.len, length(v.vec) FROM chunks c JOIN vectors v ON v.chunk = c.id ORDER BY c.id"
        start, length, size = cast("tuple[int, int, int]", connection.execute(first).fetchone())
        for chunk, document in ((900, 0), (901, 1 << 62)):
            spanned = (chunk, document, start, length)
            connection.execute("INSERT INTO chunks(id, doc, seq, pos, len) VALUES(?, ?, 0, ?, ?)", spanned)
            connection.execute("INSERT INTO vectors(chunk, vec) VALUES(?, ?)", (chunk, bytes(size)))
            connection.execute("INSERT INTO codes(chunk, bits) VALUES(?, zeroblob(8))", (chunk,))
            connection.execute("UPDATE counts SET n = n + 1 WHERE name = 'vectors'")
    (done,) = brought(home, former)
    assert isinstance(done, Brought)
    assert (done.vectors_before, done.vectors) == (4, 2)
    (one,) = known_collections(home)
    assert vectors_of(one.store) == vectors_of(tmp_path / "made" / "collections" / "notes.sqlite")


def test_passages_all_changed_since_carry_no_vector(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    for name in NOTES:
        write(notes / name, f"# {name}\n\nWritten anew.\n")
    (done,) = brought(tmp_path / "home", former)
    assert isinstance(done, Brought)
    assert (done.passages, done.vectors) == (2, 0)


def test_an_import_run_again_over_a_folder_now_gone_says_so(tmp_path: Path, notes: Path) -> None:
    former, home = former_home(tmp_path, {"notes": notes}), tmp_path / "home"
    brought(home, former)
    for path in notes.iterdir():
        path.unlink()
    notes.rmdir()
    (done,) = brought(home, former)
    assert isinstance(done, Skipped)
    assert done.code == Problem.FOLDER_MISSING
    assert done.remedy == (
        "bring its folder back, then import it again: sift reads a collection's files anew, "
        "and the vectors of sift 1.x follow the text it finds"
    )


def test_vectors_sift_1_14_named_by_their_model_alone_are_of_the_commit_it_pinned(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    with contextlib.closing(sqlite3.connect(former / "collections" / "notes.sqlite")) as connection, connection:
        connection.execute("UPDATE meta SET value = 'google/embeddinggemma-2' WHERE key = 'model'")
        connection.execute("DELETE FROM meta WHERE key = 'model_at'")
    (done,) = brought(tmp_path / "home", former)
    assert isinstance(done, Brought)
    assert (done.model, done.vectors) == ("google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e", 2)


def test_a_file_where_an_imported_store_goes_is_left_and_its_collection_skipped(tmp_path: Path, notes: Path) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    (former / "collections" / "notes-v2.sqlite").write_bytes(b"someone's file")
    (done,) = brought(tmp_path / "home", former)
    assert isinstance(done, Skipped)
    assert ("exists already" in done.message, (former / "collections" / "notes-v2.sqlite").read_bytes()) == (
        True,
        b"someone's file",
    )


def test_a_broken_vector_of_sift_1_is_left_to_embed_and_the_intact_ones_are_carried(
    tmp_path: Path, notes: Path
) -> None:
    former = former_home(tmp_path, {"notes": notes}, hashed())
    with contextlib.closing(sqlite3.connect(former / "collections" / "notes.sqlite")) as connection, connection:
        (chunk, size) = cast("tuple[int, int]", connection.execute("SELECT chunk, length(vec) FROM vectors").fetchone())
        broken = np.full(size // 2, np.nan, np.float16).tobytes()
        connection.execute("UPDATE vectors SET vec = ? WHERE chunk = ?", (broken, chunk))
    home = tmp_path / "home"
    (done,) = brought(home, former)
    assert isinstance(done, Brought)
    assert (done.vectors_before, done.vectors) == (2, 1)
    assert embed(home, EmbedCollections(), quiet, hashed()).collections[0].outcome == Embedded(1, 0)
