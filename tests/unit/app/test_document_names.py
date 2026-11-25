import unicodedata
from pathlib import Path

from sift.api.collections import AddCollection
from sift.api.indexing import IndexCollections
from sift.api.search import SearchKeywords
from sift.app.collections import add
from sift.app.documents import documents
from sift.app.indexing import index
from sift.app.search import search
from tests.files import write
from tests.homes import asked


def indexed(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        write(tmp_path / "notes" / name, text)
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "notes")))
    index(home, IndexCollections(), lambda _: None)
    return home


def test_a_file_named_decomposed_is_read_and_linked_to_by_its_composed_id(tmp_path: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "Müller.md")
    home = indexed(tmp_path, {decomposed: "# Müller\n\nA note.\n", "a.md": "# A\n\nSee [[Müller]], a friend.\n"})
    (found,) = documents(home, asked("notes/Müller.md")).found
    assert unicodedata.normalize("NFC", found.id) == "notes/Müller.md"
    linking = search(home, SearchKeywords("friend", links_to=("notes/Müller.md",)))
    assert [hit.id for hit in linking.hits] == ["notes/a.md"]


def test_a_name_whose_casefold_is_not_its_lower_case_is_told_in_which_case_to_give_it(tmp_path: Path) -> None:
    home = indexed(tmp_path, {"Straße.md": "# Straße\n"})
    (missing,) = documents(home, asked("notes/strasse.md")).not_read
    assert missing.remedy == "ids keep their case: give notes/Straße.md"
