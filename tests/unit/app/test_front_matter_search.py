from pathlib import Path

from sift.api.collections import AddCollection
from sift.api.indexing import IndexCollections
from sift.api.search import SearchKeywords
from sift.app.collections import add
from sift.app.indexing import index
from sift.app.search import search
from tests.files import write

NESTED = "---\nauthor:\n  name: Zanzibarian\npeople:\n  - name: A\n    role: Wombatchief\n---\n# Note\n\nbody\n"


def test_a_search_finds_the_words_of_nested_front_matter(tmp_path: Path) -> None:
    write(tmp_path / "notes" / "n.md", NESTED)
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "notes")))
    index(home, IndexCollections(), lambda _: None)
    for word in ("Zanzibarian", "Wombatchief"):
        assert [hit.id for hit in search(home, SearchKeywords(word)).hits] == ["notes/n.md"]
