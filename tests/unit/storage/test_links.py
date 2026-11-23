import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from sift.api.ids import CollectionId, DocumentId
from sift.domain.text.links import LinkKind
from sift.domain.text.notes import note
from sift.domain.text.resolving import Candidate
from sift.storage.database import reading
from sift.storage.documents import add_documents, put_source
from sift.storage.keywords import writing_documents
from sift.storage.links import doubtful, keys_of, links_into, links_out, named
from sift.storage.store import create_store, open_store

NOTES = {
    "People/Hans.md": "---\naliases: [Hansi]\n---\n# Hans\n",
    "Projects/Plan.md": (
        '---\nlead: "[[Hans]]"\n---\n# Plan\n\n[[Hans]] and [x](../People/Hans.md).\n\n[[Nobody]] [[a/b]]\n'
    ),
    "Other.md": "# Other\n\nSee [[Hansi]].\n",
}


@pytest.fixture
def store(tmp_path: Path) -> Iterator[tuple[sqlite3.Connection, dict[str, DocumentId]]]:
    create_store(tmp_path / "s.sqlite", CollectionId(2))
    with contextlib.closing(open_store(tmp_path / "s.sqlite", CollectionId(2))) as connection:
        ids: dict[str, DocumentId] = {}
        for path, text in NOTES.items():
            with writing_documents(connection) as transaction:
                source = put_source(transaction, path, 1, 2)
                (ids[path],) = add_documents(transaction, source, path, (note(text.encode(), path.rsplit("/", 1)[-1]),))
        yield connection, ids


def test_the_files_answering_to_a_name_are_found_by_name_and_by_alias(
    store: tuple[sqlite3.Connection, dict[str, DocumentId]],
) -> None:
    connection, ids = store
    with reading(connection) as transaction:
        assert named(transaction, "hans") == ([Candidate(ids["People/Hans.md"], "People/Hans.md")], [])
        assert named(transaction, "hansi") == ([], [Candidate(ids["People/Hans.md"], "People/Hans.md")])
        assert keys_of(transaction, ids["People/Hans.md"]) == ["hans", "hansi"]


def test_the_links_of_a_note_come_in_order_and_those_into_it_by_path_and_line(
    store: tuple[sqlite3.Connection, dict[str, DocumentId]],
) -> None:
    connection, ids = store
    with reading(connection) as transaction:
        out = links_out(transaction, ids["Projects/Plan.md"], None)
        assert [(each.field, each.kind, each.target) for each in out] == [
            ("lead", LinkKind.WIKILINK, "Hans"),
            (None, LinkKind.WIKILINK, "Hans"),
            (None, LinkKind.MARKDOWN, "../People/Hans.md"),
            (None, LinkKind.WIKILINK, "Nobody"),
            (None, LinkKind.WIKILINK, "a/b"),
        ]
        assert [each.target for each in links_out(transaction, ids["Projects/Plan.md"], "lead")] == ["Hans"]
        hans = ids["People/Hans.md"]
        into = links_into(transaction, keys_of(transaction, hans), hans, None)
        assert [(each.path, each.target) for each in into] == [
            ("Other.md", "Hansi"),
            ("Projects/Plan.md", "Hans"),
            ("Projects/Plan.md", "Hans"),
            ("Projects/Plan.md", "../People/Hans.md"),
        ]
        assert len(links_into(transaction, ["hans"], hans, "lead")) == 1


def test_links_that_may_lead_nowhere_are_those_by_path_and_those_no_name_answers(
    store: tuple[sqlite3.Connection, dict[str, DocumentId]],
) -> None:
    connection, _ = store
    with reading(connection) as transaction:
        assert [each.target for each in doubtful(transaction)] == ["../People/Hans.md", "Nobody", "a/b"]
