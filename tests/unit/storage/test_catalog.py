import contextlib
import sqlite3
import stat
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.collections import TitleFrom
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, StoreUnavailableError
from sift.api.ids import CollectionId
from sift.storage.catalog import (
    CATALOG_DAMAGED_REMEDY,
    Choices,
    Collection,
    NewCollection,
    add_collection,
    catalog_of,
    collections,
    known_collections,
    named,
    open_catalog,
    store_path,
)
from sift.storage.catalog import LAYOUT as CATALOG_LAYOUT
from sift.storage.database import connect, read_only, reading, rows, run, writing
from sift.storage.store import create_store, open_store
from sift.storage.store_files import CATALOG
from tests.damage import damage_pages
from tests.unit.storage.catalogs import folder, known


class Version(NamedTuple):
    number: int


def test_the_catalog_lies_in_the_home_folder_and_records_its_layout(home: Path, catalog: sqlite3.Connection) -> None:
    assert (home / CATALOG).is_file()
    with reading(catalog) as transaction:
        assert rows(transaction, Version, "PRAGMA user_version") == [Version(len(CATALOG_LAYOUT))]


def test_what_the_catalog_is_told_is_on_the_disk_once_told(catalog: sqlite3.Connection) -> None:
    with reading(catalog) as transaction:
        assert rows(transaction, Version, "PRAGMA synchronous") == [Version(2)]


def test_a_catalog_of_a_newer_sift_asks_for_an_upgrade(home: Path) -> None:
    with contextlib.closing(connect(home / CATALOG, create=True)) as connection, writing(connection) as transaction:
        run(transaction, "PRAGMA user_version = 99")
    with pytest.raises(StoreUnavailableError, match="newer sift"):
        open_catalog(home)


def test_a_collection_is_named_after_its_folder_and_stored_in_the_home_folder(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection
) -> None:
    notes = folder(tmp_path, "notes")
    added = add_collection(catalog, home, NewCollection(notes))
    assert added == Collection(
        CollectionId(1),
        "notes",
        notes.resolve(),
        (home / "collections" / "notes.sqlite").resolve(),
        active=True,
        choices=Choices(),
    )
    assert known(catalog) == (added,)
    with contextlib.closing(open_store(added.store, added.id)):
        pass


def test_a_store_goes_into_the_folder_or_file_given(tmp_path: Path, home: Path) -> None:
    assert store_path(home, "notes", tmp_path / "stores") == (tmp_path / "stores" / "notes.sqlite").resolve()
    assert store_path(home, "notes", tmp_path / "x" / "Mine.SQLITE") == (tmp_path / "x" / "Mine.SQLITE").resolve()


def test_choices_are_kept(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    chosen = Choices(title_from=TitleFrom.NAME, follow_links=True)
    add_collection(catalog, home, NewCollection(folder(tmp_path, "vault"), choices=chosen))
    add_collection(catalog, home, NewCollection(folder(tmp_path, "plain")))
    with reading(catalog) as transaction:
        assert named(collections(transaction), "vault").choices == chosen
        assert named(collections(transaction), "plain").choices == Choices(
            title_from=TitleFrom.HEADING, follow_links=False
        )


@pytest.mark.parametrize(
    "choices",
    ['{"title_from": "future"}', '{"follow_links": "yes"}', '{"text_fields": "body"}', '{"id_field": 7}', "[]"],
)
def test_choices_this_sift_cannot_read_name_their_collection_and_ask_for_an_upgrade(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection, choices: str
) -> None:
    add_collection(catalog, home, NewCollection(folder(tmp_path, "vault")))
    with writing(catalog) as transaction:
        run(transaction, "UPDATE collections SET choices = ?", (choices,))
    with pytest.raises(StoreUnavailableError, match="the catalog's choices for vault") as refused:
        known(catalog)
    assert refused.value.remedy == "upgrade sift"


def test_collections_are_listed_by_name(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    for name in ("b", "a", "c"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, name)))
    assert [c.name for c in known(catalog)] == ["a", "b", "c"]


def test_an_unknown_collection_names_those_there_are(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    with pytest.raises(NotFoundError) as missing, reading(catalog) as transaction:
        named(collections(transaction), "notes")
    assert missing.value.remedy == "add a folder as a collection first"
    add_collection(catalog, home, NewCollection(folder(tmp_path, "papers")))
    add_collection(catalog, home, NewCollection(folder(tmp_path, "mail")))
    with pytest.raises(NotFoundError, match="no collection is named 'notes'") as missing, reading(catalog) as tx:
        named(collections(tx), "notes")
    assert missing.value.remedy == "the collections are: mail, papers"


def test_the_collections_of_a_home_are_read_without_making_a_catalog(tmp_path: Path, home: Path) -> None:
    assert known_collections(home) == ()
    assert not home.exists()
    with contextlib.closing(open_catalog(home)) as catalog:
        added = add_collection(catalog, home, NewCollection(folder(tmp_path, "papers")))
    assert known_collections(home) == (added,)
    assert named(known_collections(home), "PAPERS") == added
    with pytest.raises(NotFoundError, match="no collection is named 'x'"):
        named((), "x")


def test_what_is_no_folder_is_refused(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    for path in (tmp_path / "gone", tmp_path / "file.txt"):
        with pytest.raises(InvalidRequestError, match="is no folder"):
            add_collection(catalog, home, NewCollection(path))
    assert known(catalog) == ()


def test_a_folder_is_one_collection(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    add_collection(catalog, home, NewCollection(folder(tmp_path, "notes")))
    with pytest.raises(InvalidRequestError, match="is the collection 'notes' already"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "notes"), name="again"))


@pytest.mark.parametrize("inner, outer", [("notes/sub", "notes"), ("notes", "notes/sub")])
def test_a_folder_inside_a_collection_or_around_one_is_added_only_as_nested(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection, inner: str, outer: str
) -> None:
    add_collection(catalog, home, NewCollection(folder(tmp_path, outer), name="first"))
    with pytest.raises(InvalidRequestError, match="found twice") as refused:
        add_collection(catalog, home, NewCollection(folder(tmp_path, inner), name="second"))
    assert refused.value.remedy == "add it as nested, if both should hold them"
    assert (
        add_collection(catalog, home, NewCollection(folder(tmp_path, inner), name="second", nested=True)).name
        == "second"
    )


def test_a_store_never_takes_a_file_there_already(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    taken = tmp_path / "taken.sqlite"
    taken.write_bytes(b"someone's")
    with pytest.raises(InvalidRequestError, match="exists already"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "notes"), store=taken))
    assert (known(catalog), taken.read_bytes()) == ((), b"someone's")


def test_a_store_an_add_left_behind_without_documents_is_made_anew(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection
) -> None:
    left = tmp_path / "left.sqlite"
    create_store(left, CollectionId(1))
    added = add_collection(catalog, home, NewCollection(folder(tmp_path, "notes"), store=left))
    with contextlib.closing(open_store(left, added.id)) as connection:
        assert connection.execute("SELECT count(*) FROM documents").fetchone() == (0,)


@pytest.mark.parametrize("kind", ["another program's", "empty", "another collection's"])
def test_a_file_no_add_of_this_collection_left_is_never_taken(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection, kind: str
) -> None:
    taken = tmp_path / "taken.sqlite"
    if kind == "another program's":
        with contextlib.closing(sqlite3.connect(taken)) as database:
            database.execute("CREATE TABLE bookmarks(url TEXT)")
            database.commit()
    elif kind == "empty":
        taken.write_bytes(b"")
    else:
        create_store(taken, CollectionId(7))
    before = taken.read_bytes()
    with pytest.raises(InvalidRequestError, match="exists already"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "notes"), store=taken))
    assert (taken.read_bytes(), known(catalog)) == (before, ())


@pytest.mark.parametrize("left", [b"", None])
def test_a_store_an_add_was_stopped_making_in_the_home_is_made_anew(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection, left: bytes | None
) -> None:
    store = home / "collections" / "notes.sqlite"
    store.parent.mkdir(parents=True)
    if left is None:
        with contextlib.closing(sqlite3.connect(store)) as database:
            database.execute("PRAGMA journal_mode=WAL")
    else:
        store.write_bytes(left)
    added = add_collection(catalog, home, NewCollection(folder(tmp_path, "notes")))
    assert added.store == store
    with contextlib.closing(open_store(store, added.id)) as connection:
        assert connection.execute("SELECT count(*) FROM documents").fetchone() == (0,)


@pytest.mark.parametrize("kind", ["a database", "no database"])
def test_a_file_of_another_program_in_the_home_s_stores_is_never_taken(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection, kind: str
) -> None:
    store = home / "collections" / "notes.sqlite"
    store.parent.mkdir(parents=True)
    if kind == "a database":
        with contextlib.closing(sqlite3.connect(store)) as database:
            database.execute("CREATE TABLE bookmarks(url TEXT)")
            database.commit()
    else:
        store.write_bytes(b"someone's notes, no database at all" * 100)
    with pytest.raises(InvalidRequestError, match="exists already"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "notes")))


def test_a_store_holding_documents_is_never_taken(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    held = tmp_path / "held.sqlite"
    create_store(held, CollectionId(7))
    with contextlib.closing(open_store(held, CollectionId(7))) as connection:
        connection.execute("INSERT INTO sources(path, modified_ns, size, folded) VALUES ('a.md', 1, 1, 'a.md')")
        connection.execute("INSERT INTO documents(source, key, digest, title, fields) VALUES (1, '', 'd', 'a', '{}')")
    with pytest.raises(InvalidRequestError, match="exists already"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "notes"), store=held))


def test_the_store_of_a_collection_on_a_missing_drive_is_not_given_away(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection
) -> None:
    first = add_collection(catalog, home, NewCollection(folder(tmp_path, "a"), store=tmp_path / "s.sqlite"))
    first.store.unlink()
    with pytest.raises(InvalidRequestError, match="is the store of the collection 'a'"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "b"), store=tmp_path / "s.sqlite"))


def test_a_store_that_could_not_be_made_leaves_nothing(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    (tmp_path / "blocked").write_text("a file where a folder would go", encoding="utf-8")
    with pytest.raises(FileExistsError):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "notes"), store=tmp_path / "blocked" / "s.sqlite"))
    assert known(catalog) == ()


def test_an_id_is_never_given_twice(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    first = add_collection(catalog, home, NewCollection(folder(tmp_path, "a")))
    with writing(catalog) as transaction:
        run(transaction, "DELETE FROM collections")
    assert add_collection(catalog, home, NewCollection(folder(tmp_path, "b"))).id == first.id + 1


def test_a_catalog_is_opened_while_another_program_writes_to_it(home: Path, catalog: sqlite3.Connection) -> None:
    with writing(catalog), contextlib.closing(open_catalog(home)) as second, reading(second) as transaction:
        assert collections(transaction) == ()


def test_a_collection_whose_registration_fails_at_the_end_leaves_no_store(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection
) -> None:
    with writing(catalog) as transaction:
        run(transaction, "CREATE TABLE owed(id INTEGER REFERENCES collections(id) DEFERRABLE INITIALLY DEFERRED)")
        run(
            transaction,
            "CREATE TRIGGER owing AFTER INSERT ON collections BEGIN INSERT INTO owed(id) VALUES (NEW.id + 1); END",
        )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "notes")))
    assert known(catalog) == ()
    assert not (home / "collections" / "notes.sqlite").exists()


def test_choices_that_are_no_json_are_refused_by_the_catalog(catalog: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"), writing(catalog) as transaction:
        run(transaction, "INSERT INTO collections(name, folder, store, choices) VALUES ('a', 'f', 's', '{no')")


def test_a_damaged_catalog_is_unavailable_saying_what_to_do(home: Path) -> None:
    with contextlib.closing(open_catalog(home)):
        pass
    damage_pages(home / CATALOG)
    with pytest.raises(StoreUnavailableError, match=r"the catalog .* is damaged") as refused:
        known_collections(home)
    assert (refused.value.code, refused.value.remedy) == (Problem.CATALOG_DAMAGED, CATALOG_DAMAGED_REMEDY)


def test_a_catalog_another_program_writes_is_unavailable_saying_what_to_do(home: Path) -> None:
    with (
        contextlib.closing(open_catalog(home)) as other,
        writing(other),
        pytest.raises(StoreUnavailableError, match="another sift is writing to the catalog") as refused,
        catalog_of(home, timeout=0) as connection,
        writing(connection),
    ):
        pass
    assert refused.value.code == Problem.STORE_UNAVAILABLE


def test_the_collections_of_a_read_only_catalog_are_listed_and_it_is_left_as_it_was(tmp_path: Path, home: Path) -> None:
    with contextlib.closing(open_catalog(home)) as catalog:
        added = add_collection(catalog, home, NewCollection(folder(tmp_path, "notes")))
    path = home / CATALOG
    before = path.read_bytes()
    path.chmod(stat.S_IREAD)
    try:
        assert known_collections(home) == (added,)
        assert path.read_bytes() == before
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_a_catalog_of_an_older_layout_is_brought_up_to_date_when_its_collections_are_listed(home: Path) -> None:
    home.mkdir()
    (home / CATALOG).touch()
    assert known_collections(home) == ()
    with contextlib.closing(read_only(home / CATALOG)) as catalog, reading(catalog) as transaction:
        assert rows(transaction, Version, "PRAGMA user_version") == [Version(len(CATALOG_LAYOUT))]


def test_a_catalog_lacking_a_table_is_damaged(home: Path) -> None:
    with contextlib.closing(open_catalog(home)) as connection, writing(connection) as transaction:
        run(transaction, "DROP TABLE leftovers")
    with pytest.raises(StoreUnavailableError, match="it lacks the table leftovers") as read:
        known_collections(home)
    with pytest.raises(StoreUnavailableError, match="it lacks the table leftovers") as opened, catalog_of(home):
        pass
    assert (read.value.code, opened.value.code) == (Problem.CATALOG_DAMAGED, Problem.CATALOG_DAMAGED)
