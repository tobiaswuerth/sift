from pathlib import Path

import pytest

from sift.api.collections import AddCollection, Counted, ShowStatus
from sift.api.errors import InvalidRequestError, NotFoundError
from sift.api.indexing import IndexCollections
from sift.api.links import FindMentions
from sift.api.search import SearchResult
from sift.app.collections import add, status
from sift.app.indexing import index
from sift.app.mentions import mentions

HANS, ACME = "vault/People/Hans Müller.md", "vault/Orgs/Acme AG.md"
ACME_LINK = '"[[Acme AG]]"'


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collection(tmp_path: Path, home: Path, name: str, files: dict[str, str]) -> None:
    for path, text in files.items():
        write(tmp_path / name / path, text)
    add(home, AddCollection(str(tmp_path / name)))


@pytest.fixture
def named(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    collection(
        tmp_path,
        home,
        "vault",
        {
            "People/Hans Müller.md": f"---\naliases: [Hans Mueller]\norg: {ACME_LINK}\n---\nHans Müller, a customer.\n",
            "People/Anna Keller.md": "Anna works with [[Hans Müller]].\n",
            "Orgs/Acme AG.md": "Acme AG is run by [[Hans Müller]]; Hans Müller founded it.\n",
            "Offer.md": "An offer for [[Hans Mueller]] from Acme AG.\n",
        },
    )
    collection(tmp_path, home, "work", {"contract.md": "Hans Müller signed for Acme AG.\n"})
    index(home, IndexCollections(), lambda _: None)
    return home


def ids(result: SearchResult) -> list[str]:
    return sorted(hit.id for hit in result.hits)


def test_documents_naming_a_note_without_linking_it_are_found(named: Path) -> None:
    assert ids(mentions(named, FindMentions(HANS))) == ["work/contract.md"]
    assert ids(mentions(named, FindMentions(ACME))) == ["vault/Offer.md", "work/contract.md"]
    assert ids(mentions(named, FindMentions("Hans Mueller"))) == ["work/contract.md"]
    assert ids(mentions(named, FindMentions(HANS, collection="vault"))) == []
    (hit,) = mentions(named, FindMentions(HANS)).hits
    assert hit.text is not None
    assert "Hans Müller" in hit.text
    (short,) = mentions(named, FindMentions(HANS, compact=True, fields=("*",))).hits
    assert (short.text, short.fields) == (None, {})


def test_a_note_is_mentioned_by_every_name_it_has(tmp_path: Path) -> None:
    home = tmp_path / "home"
    collection(
        tmp_path,
        home,
        "v",
        {
            "Jürg Bühler.md": '---\ntitle: "Dr. J. Bühler"\naliases: [JB Consulting]\nemails: [jb@example.com]\n---\n',
            "a.md": "Met Juerg Buehler today.",
            "b.md": "Met Jurg Buhler today.",
            "c.md": "Invoice from JB Consulting.",
            "d.md": "Letter to Dr. J. Bühler.",
            "e.md": "Mail from jb@example.com.",
            "f.md": "Nothing about anyone.",
            "AI.md": "",
        },
    )
    index(home, IndexCollections(), lambda _: None)
    assert ids(mentions(home, FindMentions("v/Jürg Bühler.md"))) == ["v/a.md", "v/b.md", "v/c.md", "v/d.md"]
    assert "v/e.md" in ids(mentions(home, FindMentions("v/Jürg Bühler.md", also=("emails",))))
    assert mentions(home, FindMentions("v/AI.md")).hits == ()


def test_mentions_of_what_is_no_note_or_by_no_field_are_refused(named: Path) -> None:
    with pytest.raises(NotFoundError, match="unknown document 'Nobody'"):
        mentions(named, FindMentions("Nobody"))
    for blank in ("", " "):
        with pytest.raises(InvalidRequestError, match=r"--also names front matter fields, e\.g\. emails, not "):
            mentions(named, FindMentions(HANS, also=(blank,)))


def test_a_name_that_starts_like_an_id_is_looked_up_as_a_name(named: Path, tmp_path: Path) -> None:
    write(tmp_path / "vault" / "work" / "Plan.md", "Plan of the work.\n")
    write(tmp_path / "vault" / "Notes.md", "Read the work/Plan today.\n")
    index(named, IndexCollections(), lambda _: None)
    assert ids(mentions(named, FindMentions("work/Plan"))) == ["vault/Notes.md"]


def test_a_store_a_name_could_not_be_looked_up_in_is_said(named: Path, tmp_path: Path) -> None:
    collection(tmp_path, named, "other", {"x.md": "Hans Müller again.\n"})
    index(named, IndexCollections(), lambda _: None)
    store = named / "collections" / "other.sqlite"
    store.write_bytes(b"\xff" * 4096 + store.read_bytes()[4096:])
    found = mentions(named, FindMentions("Hans Mueller", collection="work"))
    assert (ids(found), [each.collection for each in found.unsearched]) == (["work/contract.md"], ["other"])


def test_status_counts_the_links_of_each_collection_and_those_leading_nowhere(named: Path, tmp_path: Path) -> None:
    write(tmp_path / "work" / "plan.md", "[[contract]], [[Gone]] and [[gone.md]]\n")
    index(named, IndexCollections(), lambda _: None)
    vault, work = (each.contents for each in status(named, ShowStatus()).collections)
    assert isinstance(vault, Counted)
    assert isinstance(work, Counted)
    assert ((vault.links, vault.unresolved), (work.links, work.unresolved)) == ((4, 0), (3, 2))
