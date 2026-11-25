import sqlite3
from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError, Problem
from sift.storage.catalog import NewCollection, add_collection, collections, named
from sift.storage.database import reading
from tests.unit.storage.catalogs import folder, known


@pytest.mark.parametrize("name", ["", "has space", "a/b", "semi;colon", "dots..ok?", "-x", ".", "..", ".hidden"])
def test_a_name_holds_letters_digits_underscores_dashes_and_dots(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection, name: str
) -> None:
    with pytest.raises(InvalidRequestError, match="no collection name"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, "x"), name=name or " "))
    assert add_collection(catalog, home, NewCollection(folder(tmp_path, "y"), name="Ok_name-1.0")).name == "Ok_name-1.0"


@pytest.mark.parametrize(
    "name, said",
    [
        ("x" * 101, "a collection name of 101 characters is too long"),
        ("CON", "'CON' is no collection name: Windows keeps it for a device"),
        ("nul", "'nul' is no collection name: Windows keeps it for a device"),
        ("com1", "'com1' is no collection name: Windows keeps it for a device"),
        ("LPT9", "'LPT9' is no collection name: Windows keeps it for a device"),
        ("Aux.notes", "'Aux.notes' is no collection name: Windows keeps it for a device"),
        ("COM\N{SUPERSCRIPT ONE}", "'COM\N{SUPERSCRIPT ONE}' is no collection name: Windows keeps it for a device"),
    ],
)
def test_a_name_too_long_or_kept_by_windows_is_refused_everywhere(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection, name: str, said: str
) -> None:
    with pytest.raises(InvalidRequestError) as refused:
        add_collection(catalog, home, NewCollection(folder(tmp_path, "x"), name=name))
    assert (refused.value.message, bool(refused.value.remedy)) == (said, True)
    for fine in ("y" * 100, "console", "com10", "notes.con"):
        assert add_collection(catalog, home, NewCollection(folder(tmp_path, fine), name=fine)).name == fine


def test_names_that_differ_in_case_alone_are_one_name(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    add_collection(catalog, home, NewCollection(folder(tmp_path, "Notes")))
    with pytest.raises(InvalidRequestError, match="named 'Notes' already") as refused:
        add_collection(catalog, home, NewCollection(folder(tmp_path, "other"), name="notes"))
    assert refused.value.code == Problem.TAKEN


def test_a_collection_is_found_by_its_name_in_any_case(tmp_path: Path, home: Path, catalog: sqlite3.Connection) -> None:
    add_collection(catalog, home, NewCollection(folder(tmp_path, "x"), name="Bücher"))
    with reading(catalog) as transaction:
        assert named(collections(transaction), "BÜCHER").name == "Bücher"
        assert named(collections(transaction), "bücher").name == "Bücher"


def test_collections_are_listed_by_name_whatever_its_case(
    tmp_path: Path, home: Path, catalog: sqlite3.Connection
) -> None:
    for name in ("b", "A", "c", "Ä"):
        add_collection(catalog, home, NewCollection(folder(tmp_path, f"folder-{name}"), name=name))
    assert [one.name for one in known(catalog)] == ["A", "b", "c", "Ä"]
