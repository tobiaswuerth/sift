import shutil
from pathlib import Path

import pytest

from sift.api.collections import AddCollection, UnreadableFile
from sift.api.embedding import EmbedCollections
from sift.api.events import Event, Noted
from sift.api.filters import written
from sift.api.indexing import IndexCollections, Indexed
from sift.api.search import SearchKeywords
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.filing import Limits, Readers
from sift.app.indexing import index
from sift.app.search import search
from sift.domain.text.documents import Conversion, Unread
from sift.ingest.documents import convert
from sift.ingest.files import Identity
from sift.storage.catalog import known_collections
from sift.storage.store import read_store
from sift.storage.vectors import left
from tests.homes import text_of, unreadable_files
from tests.keyword_stores import vectors_by_text
from tests.later import later
from tests.mails import PARAGRAPH, mail_bytes, message, only, sample
from tests.models import Hashed

PDF = "application/pdf"


def indexed(
    home: Path, limits: Limits | None = None, said: list[Event] | None = None, readers: Readers | None = None
) -> Indexed:
    heard: list[Event] = [] if said is None else said
    (one,) = index(
        home, IndexCollections(), heard.append, limits or Limits(), readers or Readers(clock=later)
    ).collections
    assert isinstance(one.outcome, Indexed)
    return one.outcome


def found(home: Path, query: str, filters: tuple[str, ...] = ()) -> list[str]:
    return sorted(
        hit.id for hit in search(home, SearchKeywords(query, k=20, filters=tuple(map(written, filters)))).hits
    )


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root, home = tmp_path / "library", tmp_path / "home"
    root.mkdir()
    for extension in (".pdf", ".docx", ".xlsx"):
        shutil.copy(sample(extension), root / f"pigments{extension}")
    add(home, AddCollection(str(root)))
    return home


def _offer(root: Path, body: str = "Plain words.", attached: bytes | None = None) -> Path:
    path = root / "offer.eml"
    contract = sample(".pdf").read_bytes() if attached is None else attached
    headers = {"Subject": "Offer", "From": "Ada Example <ada@example.com>"}
    path.write_bytes(mail_bytes(message(body=body, headers=headers, attachments=[("contract.pdf", contract, PDF)])))
    return path


@pytest.fixture
def post(tmp_path: Path) -> Path:
    root, home = tmp_path / "mail", tmp_path / "home"
    root.mkdir()
    _offer(root)
    add(home, AddCollection(str(root)))
    return home


def test_documents_are_indexed_as_markdown_and_found_by_their_text_and_properties(library: Path) -> None:
    assert indexed(library) == Indexed(files=3, added=3, changed=0, removed=0, unreadable=0)
    assert found(library, "absorbs scatters") == [
        "library/pigments.docx",
        "library/pigments.pdf",
        "library/pigments.xlsx",
    ]
    assert found(library, "Ultramarine") == ["library/pigments.xlsx"]
    assert found(library, "absorbs", ("authors=Ada Example",)) == found(library, "absorbs scatters")
    assert text_of(library, "library/pigments.docx") == only(sample(".docx").read_bytes(), "pigments.docx").text


def test_a_damaged_document_is_listed_not_read_again_until_it_changes_then_read(library: Path, tmp_path: Path) -> None:
    root = tmp_path / "library"
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 the rest is missing")
    assert indexed(library).unreadable == 1
    (listed,) = unreadable_files(library)
    assert listed.path == "broken.pdf"
    assert listed.problem.startswith("Parsing error: ")
    converted: list[str] = []

    def counted(raw: bytes, name: str) -> Conversion:
        converted.append(name)
        return convert(raw, name)

    assert indexed(library, readers=Readers(clock=later, convert=counted)).unreadable == 1
    assert converted == []
    shutil.copy(sample(".pdf"), root / "broken.pdf")
    assert indexed(library) == Indexed(files=4, added=1, changed=0, removed=0, unreadable=0)
    assert PARAGRAPH in text_of(library, "library/broken.pdf")


def test_a_document_that_stops_converting_loses_its_document(library: Path) -> None:
    indexed(library)

    def failing(_raw: bytes, _name: str) -> Conversion:
        return Unread("the converter failed: index out of bounds")

    (library.parent / "library" / "pigments.pdf").touch()
    assert indexed(library, readers=Readers(clock=later, convert=failing)).removed == 1
    assert unreadable_files(library) == (UnreadableFile("pigments.pdf", "the converter failed: index out of bounds"),)


def test_a_document_that_cannot_be_read_is_read_again_at_the_next_index(library: Path) -> None:
    def locked(_path: Path, _most: int, _scanned: Identity | None) -> bytes:
        raise PermissionError(13, "Permission denied")

    assert indexed(library, readers=Readers(whole=locked, clock=later)).unreadable == 3
    assert {each.problem for each in unreadable_files(library)} == {"Permission denied"}
    assert indexed(library) == Indexed(files=3, added=3, changed=0, removed=0, unreadable=0)


def test_a_mail_and_its_attachments_are_documents_found_and_filtered_by_their_ids(post: Path) -> None:
    assert indexed(post) == Indexed(files=1, added=2, changed=0, removed=0, unreadable=0)
    assert found(post, "Kubelka-Munk") == ["mail/offer.eml#contract.pdf"]
    assert found(post, "offer", ("from=ada@example.com",)) == ["mail/offer.eml"]
    assert PARAGRAPH in text_of(post, "mail/offer.eml#contract.pdf")


def test_a_changed_mail_keeps_its_unchanged_attachments_and_their_vectors(post: Path, tmp_path: Path) -> None:
    indexed(post)
    embed(post, EmbedCollections(), lambda _: None, Hashed())
    (one,) = known_collections(post)
    with read_store(one.store, one.id) as transaction:
        before = vectors_by_text(transaction)
    _offer(tmp_path / "mail", body="Other words.")
    assert indexed(post) == Indexed(files=1, added=0, changed=1, removed=0, unreadable=0)
    with read_store(one.store, one.id) as transaction:
        after = vectors_by_text(transaction)
        assert (len(after), left(transaction)) == (len(before) - 1, 1)
        assert all(before[text] == vector for text, vector in after.items())
    _offer(tmp_path / "mail", attached=b"%PDF-1.4 broken")
    said: list[Event] = []
    assert indexed(post, said=said) == Indexed(files=1, added=0, changed=1, removed=1, unreadable=1)
    (problem,) = [event.text for event in said if isinstance(event, Noted)]
    assert problem.startswith("offer.eml: contract.pdf: Parsing error: ")
    assert unreadable_files(post)[0].problem == problem.removeprefix("offer.eml: ")
    assert found(post, "offer") == ["mail/offer.eml"]
