from dataclasses import replace
from pathlib import Path

import pytest

from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, StoreUnavailableError
from sift.api.filters import Filter
from sift.api.search import SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app import searching
from sift.app.embedding import embed
from sift.app.linked import condition_in
from sift.app.notes import Note, read_in
from sift.app.query import query
from sift.app.search import search
from sift.app.vsearch import vsearch
from sift.interfaces.cli.written import written
from tests.damage import damage_pages
from tests.files import write
from tests.homes import collection, indexed, quiet
from tests.models import Hashed

OFFER, ACME = "vault/Offer.md", "vault/Orgs/Acme AG.md"
ANNA, HANS = "vault/People/Anna Keller.md", "vault/People/Hans Müller.md"


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
            "People/Anna Keller.md": "Note: Anna works with [[Hans Müller]] on [[Dup]].\n",
            "A/Dup.md": "Note: one.\n",
            "B/Dup.md": "Note: two.\n",
            "Orgs/Acme AG.md": "Note: a customer, run by [[Hans Müller]].\n",
            "Offer.md": "Note: an offer for [[Hans Mueller]], see [[Wartungsvertrag 2027]].\n",
        },
    )
    work = {
        "contract.md": "Note: Hans Müller signed for Acme AG.\n",
        "Plan.md": "Note: see [[contract]].\n",
        "old.md": "Note: see [[Archive/Wartungsvertrag 2027]].\n",
    }
    collection(tmp_path, home, "work", work)
    indexed(home)
    return home


def ids(result: SearchResult) -> list[str]:
    return sorted(hit.id for hit in result.hits)


def notes(
    home: Path,
    links_to: tuple[str, ...] = (),
    linked_from: tuple[str, ...] = (),
    collections: tuple[str, ...] = (),
    filters: tuple[Filter, ...] = (),
) -> list[str]:
    asked = SearchKeywords(
        "note", k=20, links_to=links_to, linked_from=linked_from, collections=collections, filters=filters
    )
    return ids(search(home, asked))


def test_a_search_keeps_to_the_documents_linking_a_note(linked: Path) -> None:
    assert notes(linked, links_to=("Acme AG",)) == [HANS]
    found = search(linked, SearchKeywords("note", links_to=("Acme AG",)))
    assert (found.searched, found.ruled_out) == (1, 1)
    assert notes(linked, links_to=("[[Acme AG]]",)) == [HANS]
    assert notes(linked, links_to=(ACME,), collections=("vault",)) == [HANS]
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
        (("Nobody",), (), "no note or document 'Nobody'"),
        (("vault/Nobody.md",), (), "no note or document 'vault/Nobody.md'"),
        ((" ",), (), "takes a note's id or name, not ' '"),
        ((), ("Wartungsvertrag 2027",), "no note or document 'Wartungsvertrag 2027'"),
        ((), ("",), "linked_from holds an empty text"),
    ],
)
def test_notes_no_document_answers_to_are_refused(
    linked: Path, links_to: tuple[str, ...], linked_from: tuple[str, ...], refusal: str
) -> None:
    with pytest.raises((NotFoundError, InvalidRequestError), match=refusal):
        notes(linked, links_to=links_to, linked_from=linked_from)


def test_a_name_several_notes_answer_to_is_refused_whichever_collections_are_searched(
    linked: Path, tmp_path: Path
) -> None:
    write(tmp_path / "work" / "Acme AG.md", "Note: another Acme.\n")
    indexed(linked)
    for collections in ((), ("vault",), ("work",)):
        with pytest.raises(
            InvalidRequestError, match=r"names several notes: vault/Orgs/Acme AG\.md, work/Acme AG\.md"
        ) as refused:
            notes(linked, links_to=("Acme AG",), collections=collections)
        assert refused.value.code == Problem.AMBIGUOUS
    assert notes(linked, links_to=(ACME,), collections=("vault",)) == [HANS]


def test_notes_of_several_collections_keep_a_search_to_their_links_in_each(linked: Path) -> None:
    assert notes(linked, links_to=("Acme AG", "contract")) == [HANS, "work/Plan.md"]
    assert notes(linked, linked_from=("Hans Müller", "work/Plan.md")) == [ACME, "work/contract.md"]


def test_a_store_that_cannot_be_read_while_notes_are_looked_up_is_said_to_be_unsearched(linked: Path) -> None:
    store = linked / "collections" / "work.sqlite"
    damage_pages(store)
    for links_to in (("Wartungsvertrag 2027",), ("Acme AG",)):
        found = search(linked, SearchKeywords("note", links_to=links_to))
        assert [each.collection for each in found.unsearched] == ["work"]
    with pytest.raises(NotFoundError, match="no note or document 'Nobody'; not searched: work: "):
        notes(linked, linked_from=("Nobody",))


def test_every_search_takes_links(linked: Path) -> None:
    embed(linked, EmbedCollections(), quiet, Hashed())
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
    indexed(home)
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


def test_a_note_named_by_id_keeps_other_collections_closed(linked: Path) -> None:
    store = linked / "collections" / "work.sqlite"
    damage_pages(store)
    found = search(linked, SearchKeywords("note", links_to=(ACME,)))
    assert ([hit.id for hit in found.hits], found.searched, found.ruled_out, found.unsearched) == ([HANS], 1, 1, ())


def test_a_note_named_by_id_in_a_store_that_cannot_be_read_is_the_lookup_s_error(linked: Path) -> None:
    store = linked / "collections" / "work.sqlite"
    damage_pages(store)
    for links_to, linked_from in ((("work/contract.md",), ()), ((), ("work/Plan.md",))):
        with pytest.raises(StoreUnavailableError, match="is damaged"):
            search(linked, SearchKeywords("note", links_to=links_to, linked_from=linked_from))


def test_an_ambiguous_link_and_a_link_by_another_path_lead_to_no_note_named(linked: Path) -> None:
    assert notes(linked, linked_from=("Anna Keller",)) == [HANS]
    assert notes(linked, links_to=("Wartungsvertrag 2027",)) == [OFFER]


def test_a_store_no_document_of_can_pass_the_links_is_not_searched(linked: Path) -> None:
    result = search(linked, SearchKeywords("note", links_to=("Hans Müller",), linked_from=("Anna Keller",)))
    assert (result.hits, result.searched, result.ruled_out) == ((), 0, 2)


def test_a_note_gone_before_its_store_is_read_lets_no_document_through(linked: Path) -> None:
    narrowing = searching.narrowed(linked, SearchKeywords("note", links_to=("Acme AG",)))
    assert narrowing.links_to is not None
    (pointed,) = narrowing.links_to
    assert pointed.note is not None
    gone = replace(narrowing, links_to=(replace(pointed, note=Note(pointed.note.collection, "Gone.md")),))
    (vault,) = gone.collections
    found = read_in(vault, lambda transaction: condition_in(gone, vault, transaction))
    assert found is not None
    assert found.within == "[]"


def test_a_note_linking_itself_is_left_out_of_its_own_links_both_ways(tmp_path: Path) -> None:
    home = tmp_path / "home"
    collection(
        tmp_path, home, "loop", {"Loop.md": "Note: see [[Loop]] and [[Other]].\n", "Other.md": "Note: [[Loop]].\n"}
    )
    indexed(home)
    assert notes(home, links_to=("Loop",)) == ["loop/Other.md"]
    assert notes(home, linked_from=("Loop",)) == ["loop/Other.md"]


@pytest.mark.parametrize(
    "written", ["Acme AG#Intro", "Acme AG|the firm", "Acme AG#Intro|the firm", "[[Acme AG#Intro|x]]"]
)
def test_a_note_named_with_a_heading_or_shown_text_is_the_note(linked: Path, written: str) -> None:
    assert notes(linked, links_to=(written,)) == [HANS]
    assert notes(linked, linked_from=(written.replace("Acme AG", "Hans Müller"),)) == [ACME]
