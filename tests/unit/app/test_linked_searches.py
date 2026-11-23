from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, NotFoundError
from sift.api.filters import Filter, written
from sift.api.indexing import IndexCollections
from sift.api.search import SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.query import query
from sift.app.search import search
from sift.app.vsearch import vsearch
from tests.models import Hashed

OFFER, ACME = "vault/Offer.md", "vault/Orgs/Acme AG.md"
ANNA, HANS = "vault/People/Anna Keller.md", "vault/People/Hans Müller.md"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collection(tmp_path: Path, home: Path, name: str, files: dict[str, str]) -> None:
    for path, text in files.items():
        write(tmp_path / name / path, text)
    add(home, AddCollection(str(tmp_path / name)))


@pytest.fixture
def linked(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    hans = '---\naliases: [Hans Mueller]\norg: "[[Acme AG]]"\n---\nNote: a customer since 2019; knows [[Peter Frei]].\n'
    collection(
        tmp_path,
        home,
        "vault",
        {
            "People/Hans Müller.md": hans,
            "People/Anna Keller.md": "Note: Anna works with [[Hans Müller]].\n",
            "Orgs/Acme AG.md": "Note: a customer, run by [[Hans Müller]].\n",
            "Offer.md": "Note: an offer for [[Hans Mueller]], see [[Wartungsvertrag 2027]].\n",
        },
    )
    work = {"contract.md": "Note: Hans Müller signed for Acme AG.\n", "Plan.md": "Note: see [[contract]].\n"}
    collection(tmp_path, home, "work", work)
    index(home, IndexCollections(), lambda _: None)
    return home


def ids(result: SearchResult) -> list[str]:
    return sorted(hit.id for hit in result.hits)


def notes(
    home: Path,
    links_to: tuple[str, ...] = (),
    linked_from: tuple[str, ...] = (),
    collection: str | None = None,
    filters: tuple[Filter, ...] = (),
) -> list[str]:
    asked = SearchKeywords(
        "note", k=20, links_to=links_to, linked_from=linked_from, collection=collection, filters=filters
    )
    return ids(search(home, asked))


def test_a_search_keeps_to_the_documents_linking_a_note(linked: Path) -> None:
    assert notes(linked, links_to=("Acme AG",)) == [HANS]
    assert search(linked, SearchKeywords("note", links_to=("Acme AG",))).searched == 2
    assert notes(linked, links_to=("[[Acme AG]]",)) == [HANS]
    assert notes(linked, links_to=(ACME,), collection="vault") == [HANS]
    assert notes(linked, links_to=("Hans Müller",)) == [OFFER, ACME, ANNA]
    assert notes(linked, links_to=("Acme AG", "Wartungsvertrag 2027")) == [OFFER, HANS]


def test_a_search_keeps_to_the_documents_a_note_links(linked: Path) -> None:
    assert notes(linked, linked_from=("Hans Müller",)) == [ACME]
    assert notes(linked, linked_from=(HANS,)) == [ACME]
    assert notes(linked, links_to=("Acme AG",), linked_from=("Anna Keller",)) == [HANS]
    assert notes(linked, links_to=("Hans Müller",), linked_from=("Anna Keller",)) == []


@pytest.mark.parametrize(
    "links_to, linked_from, refusal",
    [
        (("Nobody",), (), "no note 'Nobody' in the index, and no link to one"),
        (("vault/Nobody.md",), (), "no note 'vault/Nobody.md' in the index, and no link to one"),
        ((" ",), (), "--links-to takes a note's id or name, not ' '"),
        ((), ("Wartungsvertrag 2027",), "no note 'Wartungsvertrag 2027' in the index"),
        ((), ("",), "--linked-from takes a note's id or name, not ''"),
    ],
)
def test_notes_no_document_answers_to_are_refused(
    linked: Path, links_to: tuple[str, ...], linked_from: tuple[str, ...], refusal: str
) -> None:
    with pytest.raises((NotFoundError, InvalidRequestError), match=refusal):
        notes(linked, links_to=links_to, linked_from=linked_from)


def test_a_name_several_notes_answer_to_is_chosen_by_collection(linked: Path, tmp_path: Path) -> None:
    write(tmp_path / "work" / "Acme AG.md", "Note: another Acme.\n")
    index(linked, IndexCollections(), lambda _: None)
    with pytest.raises(InvalidRequestError, match=r"names several notes: vault/Orgs/Acme AG\.md, work/Acme AG\.md"):
        notes(linked, links_to=("Acme AG",))
    assert notes(linked, links_to=("Acme AG",), collection="vault") == [HANS]
    write(tmp_path / "work" / "Sub" / "Acme AG.md", "Note: a third.\n")
    index(linked, IndexCollections(), lambda _: None)
    with pytest.raises(InvalidRequestError, match=r"names several notes: work/Acme AG\.md, work/Sub/Acme AG\.md"):
        notes(linked, links_to=("Acme AG",), collection="work")


def test_notes_of_several_collections_keep_a_search_to_their_links_in_each(linked: Path) -> None:
    assert notes(linked, links_to=("Acme AG", "contract")) == [HANS, "work/Plan.md"]
    assert notes(linked, linked_from=("Hans Müller", "work/Plan.md")) == [ACME, "work/contract.md"]


def test_a_store_that_cannot_be_read_while_notes_are_looked_up_is_said_to_be_unsearched(linked: Path) -> None:
    store = linked / "collections" / "work.sqlite"
    store.write_bytes(b"\xff" * 4096 + store.read_bytes()[4096:])
    for links_to in (("Wartungsvertrag 2027",), ("Acme AG",)):
        found = search(linked, SearchKeywords("note", links_to=links_to))
        assert [each.collection for each in found.unsearched] == ["work"]
    with pytest.raises(NotFoundError, match="no note 'Nobody' in the index; not searched: work: "):
        notes(linked, linked_from=("Nobody",))


def test_every_search_takes_links(linked: Path) -> None:
    embed(linked, EmbedCollections(), lambda _: None, Hashed())
    assert ids(vsearch(linked, SearchVectors("customer", k=5, linked_from=("Hans Müller",)), Hashed())) == [ACME]
    assert ids(query(linked, SearchHybrid("customer since 2019", links_to=("Acme AG",)), Hashed())) == [HANS]


@pytest.fixture
def orgs(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    collection(
        tmp_path,
        home,
        "v",
        {
            "Orgs/Acme AG.md": "---\naliases: [Acme]\n---\nNote.\n",
            "a.md": '---\norg: "[[Acme AG]]"\n---\nNote.\n',
            "b.md": '---\norg: "[[Acme AG|Acme]]"\n---\nNote.\n',
            "c.md": "---\norg: [[acme ag]]\n---\nNote.\n",
            "d.md": '---\norg: ["[[Orgs/Acme AG]]", "[[Beta]]"]\n---\nNote.\n',
            "e.md": '---\norg: "[[Acme#History]]"\n---\nNote.\n',
            "f.md": '---\norg: "[[Beta]]"\n---\nNote.\n',
            "g.md": '---\nclient: "[[Acme AG]]"\n---\nNote.\n',
            "h.md": "---\norg: Acme AG\n---\nNote.\n",
        },
    )
    index(home, IndexCollections(), lambda _: None)
    return home


def filtered(home: Path, *texts: str) -> list[str]:
    return [each.removeprefix("v/").removesuffix(".md") for each in notes(home, filters=tuple(map(written, texts)))]


def test_a_wikilink_filter_value_matches_a_link_to_the_note_however_written(orgs: Path) -> None:
    acme = ["a", "b", "c", "d", "e"]
    assert filtered(orgs, "org=[[Acme AG|Acme]]") == acme
    assert filtered(orgs, "org=[[Acme AG]]") == acme
    assert filtered(orgs, "org=[[Orgs/Acme AG]]") == acme
    assert filtered(orgs, "org!=[[Acme AG]]") == ["Orgs/Acme AG", "f", "g", "h"]
    assert filtered(orgs, "org=Acme AG") == ["h"]
    assert filtered(orgs, 'org=["[[beta.md]]", "Acme AG"]') == ["d", "f", "h"]
    assert filtered(orgs, "client=[[Acme]]") == ["g"]
    assert filtered(orgs, "org=[[Acme AG]]", "org!=[[Beta]]") == ["a", "b", "c", "e"]
