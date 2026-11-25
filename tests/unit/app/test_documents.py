from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.documents import Documents, DocumentText, GetDocuments, Unfound, WantedDocument
from sift.api.errors import InvalidRequestError, Problem
from sift.api.ids import CollectionId
from sift.api.indexing import IndexCollections
from sift.app.collections import add
from sift.app.documents import documents
from sift.app.indexing import index
from sift.storage.database import Reading
from sift.storage.store import read_store
from sift.storage.unusable import DAMAGED_REMEDY
from tests.damage import damage_table
from tests.files import write
from tests.homes import asked

UNKNOWN = "take the id from a search result"


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root, home = tmp_path / "notes", tmp_path / "home"
    write(root / "rl.md", "# Reward shaping\n\nReward shaping speeds up learning.\n")
    write(root / "robots" / "brush.md", "# Brush\n")
    for path in ("a/b/c.md", "x/b/c.md", "a/bc.md", "Twice.md"):
        write(root / path, f"# {path}\n")
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    return home


def text(home: Path, id_: str, first: int | None = None, last: int | None = None) -> str:
    result = documents(home, asked(id_, lines=(first, last)))
    assert result.unfound == ()
    return result.found[0].text


def unfound(home: Path, id_: str) -> Unfound:
    (missing,) = documents(home, asked(id_)).unfound
    return missing


def test_a_document_is_read_by_its_id_as_it_was_indexed(home: Path) -> None:
    assert documents(home, asked("notes/rl.md")) == Documents(
        (DocumentText("notes/rl.md", "Reward shaping", "# Reward shaping\n\nReward shaping speeds up learning.\n"),),
        (),
    )


def test_a_collection_is_named_in_any_case(home: Path) -> None:
    assert documents(home, asked("NOTES/rl.md")).found[0].id == "notes/rl.md"


def test_documents_asked_each_with_lines_of_their_own_come_back_each_with_its_own(home: Path) -> None:
    wanted = (
        WantedDocument("notes/rl.md", 1, 1),
        WantedDocument("notes/robots/brush.md"),
        WantedDocument("notes/rl.md", 3),
    )
    found = documents(home, GetDocuments(wanted)).found
    assert [each.text for each in found] == ["# Reward shaping\n", "# Brush\n", "Reward shaping speeds up learning.\n"]


def test_only_some_lines_are_read_if_asked(home: Path) -> None:
    assert text(home, "notes/rl.md", 1, 1) == "# Reward shaping\n"
    assert text(home, "notes/rl.md", 2, 9) == "\nReward shaping speeds up learning.\n"
    assert text(home, "notes/rl.md", None, 1) == "# Reward shaping\n"
    with pytest.raises(InvalidRequestError, match="0 is outside 1 to"):
        WantedDocument("notes/rl.md", 0)
    with pytest.raises(InvalidRequestError, match="line 2 comes before line 3"):
        documents(home, asked("notes/rl.md", lines=(3, 2)))
    (past,) = documents(home, asked("notes/rl.md", lines=(9, 9))).unfound
    assert past.message == "notes/rl.md has 3 lines; line 9 is past its end"


@pytest.mark.parametrize(
    "short, whole",
    [
        ("notes/brush.md", "notes/robots/brush.md"),
        ("notes/bc.md", "notes/a/bc.md"),
        ("notes/a/b/c.md", "notes/a/b/c.md"),
    ],
)
def test_a_shortened_id_is_the_end_of_a_path(home: Path, short: str, whole: str) -> None:
    assert documents(home, asked(short)).found[0].id == whole


def test_a_shortened_id_several_paths_end_in_asks_for_the_whole(home: Path) -> None:
    for several in ("notes/c.md", "notes/b/c.md"):
        missing = unfound(home, several)
        assert (missing.message, missing.remedy) == (
            f"{several!r} matches a/b/c.md, x/b/c.md",
            "give the whole id",
        )


@pytest.mark.parametrize(
    "unknown", ["notes/C.md", "notes/rush.md", "notes/z/b/c.md", "notes/robots", "missing/rl.md", "notes/", "notes"]
)
def test_an_id_that_names_no_document_whole_is_unknown(home: Path, unknown: str) -> None:
    assert unfound(home, unknown) == Unfound(
        unknown, f"unknown document {unknown!r}", UNKNOWN, Problem.UNKNOWN_DOCUMENT
    )


def test_a_document_is_read_by_its_file_s_absolute_path_in_any_case(tmp_path: Path, home: Path) -> None:
    root = tmp_path / "notes"
    assert documents(home, asked(str(root / "robots" / "brush.md"))).found[0].id == "notes/robots/brush.md"
    assert documents(home, asked(str(root / "Robots" / "BRUSH.md"))).found[0].id == "notes/robots/brush.md"
    assert documents(home, asked(str(root / "twice.md"))).found[0].id == "notes/Twice.md"
    for outside in (tmp_path / "rl.md", root / "brush.md"):
        assert unfound(home, str(outside)).message.startswith("unknown document")


def test_an_absolute_path_finds_its_document_whatever_the_case_of_its_letters_beyond_ascii(
    tmp_path: Path, home: Path
) -> None:
    write(tmp_path / "notes" / "Äpfel.md", "# Äpfel\n")
    index(home, IndexCollections(), lambda _: None)
    for written in ("Äpfel.md", "äpfel.md", "ÄPFEL.MD"):
        assert documents(home, asked(str(tmp_path / "notes" / written))).found[0].id == "notes/Äpfel.md"


def test_an_absolute_id_on_a_share_no_collection_lies_on_is_unknown(home: Path) -> None:
    elsewhere = str(Path("//host.invalid/share/notes/rl.md"))
    assert unfound(home, elsewhere).message == f"unknown document {elsewhere!r}"


def test_several_documents_are_read_at_once_and_those_not_found_said(home: Path) -> None:
    result = documents(home, asked("notes/rl.md", "notes/gone.md", "notes/brush.md"))
    assert [found.id for found in result.found] == ["notes/rl.md", "notes/robots/brush.md"]
    assert [missing.id for missing in result.unfound] == ["notes/gone.md"]


def test_a_document_is_read_as_indexed_until_the_next_index(tmp_path: Path, home: Path) -> None:
    write(tmp_path / "notes" / "rl.md", "# Rewritten\n")
    assert text(home, "notes/rl.md").startswith("# Reward shaping")
    (tmp_path / "notes" / "rl.md").unlink()
    index(home, IndexCollections(), lambda _: None)
    assert unfound(home, "notes/rl.md").message == "unknown document 'notes/rl.md'"


def test_a_shortened_id_many_paths_end_in_names_the_first_five(tmp_path: Path, home: Path) -> None:
    for folder in "abcdef":
        write(tmp_path / "notes" / folder / "same.md", f"# {folder}\n")
    index(home, IndexCollections(), lambda _: None)
    listed = ", ".join(f"{folder}/same.md" for folder in "abcde")
    assert unfound(home, "notes/same.md").message == f"'notes/same.md' matches {listed}, ..."


def test_a_document_of_a_damaged_store_is_not_found_and_says_why(home: Path) -> None:
    damage_table(home / "collections" / "notes.sqlite", "documents")
    missing = unfound(home, "notes/rl.md")
    assert ("is damaged" in missing.message, missing.remedy) == (True, DAMAGED_REMEDY)


def test_without_any_collection_every_id_is_unknown(tmp_path: Path) -> None:
    result = documents(tmp_path / "empty", asked("notes/rl.md"))
    assert [missing.message for missing in result.unfound] == ["unknown document 'notes/rl.md'"]
    assert not (tmp_path / "empty").exists()


def test_the_front_matter_asked_for_comes_with_each_document(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "notes" / "dated.md", "---\ncreated: 2024-06-15\nrank: 2\n---\n# Dated\n\nText.\n")
    index(home, IndexCollections(), lambda _: None)
    (found,) = documents(home, asked("notes/dated.md", fields=("rank", "missing"))).found
    assert found.fields == {"rank": 2}
    (every,) = documents(home, asked("notes/dated.md", fields=("*",))).found
    assert every.fields == {"created": "2024-06-15", "rank": 2}
    assert documents(home, asked("notes/dated.md")).found[0].fields is None
    with pytest.raises(InvalidRequestError, match="fields holds an empty text"):
        asked("notes/dated.md", fields=("",))


def test_documents_of_one_collection_are_read_with_its_store_opened_once(home: Path) -> None:
    opened = list[Path]()

    @contextmanager
    def counted(path: Path, collection: CollectionId) -> Generator[Reading]:
        opened.append(path)
        with read_store(path, collection) as transaction:
            yield transaction

    result = documents(home, asked(*["notes/rl.md", "notes/gone.md", "notes/robots/brush.md"] * 20), counted)
    assert (len(result.found), len(result.unfound), len(opened)) == (40, 20, 1)
    assert [each.id for each in result.found[:2]] == ["notes/rl.md", "notes/robots/brush.md"]
