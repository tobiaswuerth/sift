import contextlib
import dataclasses
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, cast

import pytest

from sift.api.fields import FieldValue
from sift.api.ids import ChunkId, CollectionId, DocumentId, first_row
from sift.domain.errors import DamagedError
from sift.domain.text.decoding import text_pieces
from sift.domain.text.links import LinkKind, NameKind
from sift.domain.text.markdown import MOST_LINE
from sift.domain.text.notes import MOST_HEAD, Document, note, read_note
from sift.storage.database import reading, rows, run, writing
from sift.storage.documents import (
    PATHS_NAMED,
    Counts,
    KeptDocument,
    SourceId,
    StoredChunk,
    Unreadable,
    add_documents,
    add_streamed,
    chunks_of,
    clear_unreadable,
    counts,
    fields_json,
    fields_of,
    kept,
    mark_unreadable,
    paths_named,
    put_source,
    set_digest,
    sources,
    stored,
    unreadable,
)
from sift.storage.forgetting import delete_documents, delete_sources
from sift.storage.keywords import writing_documents
from sift.storage.store import create_store, open_store
from sift.storage.texts import BLOCK, text_of

COLLECTION = CollectionId(2)
NOTE = (
    b'---\naliases: [Hansi]\norg: "[[Acme AG]]"\n---\n# Hans\n\n'
    b"See [[Acme AG#Team|Acme]], [x](Other.md), ![[scan.pdf]].\n"
)


class Count(NamedTuple):
    n: int


class LinkRow(NamedTuple):
    ordinal: int
    field: str | None
    line: int
    kind: str
    target: str
    key: str
    fragment: str | None
    shown: str | None


LINKS_OF = """SELECT ordinal, field, line, kind, target, key, fragment, shown FROM links
              WHERE document = ? ORDER BY ordinal"""


class NameRow(NamedTuple):
    key: str
    kind: str


@pytest.fixture
def store(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    with contextlib.closing(open_store(tmp_path / "s.sqlite", COLLECTION)) as connection:
        yield connection


COUNTS = MappingProxyType(
    {
        "sources": "SELECT count(*) FROM sources",
        "documents": "SELECT count(*) FROM documents",
        "chunks": "SELECT count(*) FROM chunks",
        "links": "SELECT count(*) FROM links",
        "names": "SELECT count(*) FROM names",
    }
)


def counted(connection: sqlite3.Connection, table: str) -> int:
    with reading(connection) as transaction:
        (found,) = rows(transaction, Count, COUNTS[table])
        return found.n


def added(connection: sqlite3.Connection, path: str, *documents: Document) -> tuple[SourceId, tuple[DocumentId, ...]]:
    with writing_documents(connection) as transaction:
        source = put_source(transaction, path, 10, 20)
        return source, add_documents(transaction, source, path, documents)


class DocumentRow(NamedTuple):
    key: str
    digest: str
    title: str
    fields: str


def held(connection: sqlite3.Connection, identity: DocumentId) -> tuple[object, ...]:
    with reading(connection) as transaction:
        row = rows(
            transaction, DocumentRow, "SELECT key, digest, title, fields FROM documents WHERE id = ?", (identity,)
        )
        spans = [(c.first_line, c.last_line, c.start, c.length) for c in chunks_of(transaction, identity)]
        links = rows(transaction, LinkRow, LINKS_OF, (identity,))
        named = rows(
            transaction, NameRow, "SELECT key, kind FROM names WHERE document = ? ORDER BY key, kind", (identity,)
        )
        return row, text_of(transaction, identity), spans, links, named


@pytest.mark.parametrize(
    "raw",
    [NOTE, b"---\ntitle: Front\n---\n", b"".join(b"[[n%d]]\n" % n for n in range(1500)) + b"# Late\n\nlast words\n"],
)
def test_a_note_stored_as_it_is_read_is_stored_as_the_note_read_whole(store: sqlite3.Connection, raw: bytes) -> None:
    whole = note(raw, "Hans.md")
    _, (first,) = added(store, "People/Hans.md", whole)
    before = held(store, first)
    with writing_documents(store) as transaction:
        delete_documents(transaction, [first])
        source = put_source(transaction, "People/Hans.md", 10, 20)
        second = add_streamed(transaction, source, "People/Hans.md", read_note(text_pieces((raw,)), "Hans.md"))
        set_digest(transaction, second, whole.digest)
    assert held(store, second) == before


def test_a_source_is_written_once_and_then_updated(store: sqlite3.Connection) -> None:
    with writing(store) as transaction:
        first = put_source(transaction, "a.md", 1, 2)
        assert put_source(transaction, "b.md", 1, 2) != first
        assert put_source(transaction, "a.md", 3, 4) == first
    with reading(store) as transaction:
        assert {path: (s.modified_ns, s.size) for path, s in sources(transaction).items()} == {
            "a.md": (3, 4),
            "b.md": (1, 2),
        }


def test_a_document_is_stored_with_its_chunks_links_and_names(store: sqlite3.Connection) -> None:
    document = note(NOTE, "Hans Müller.md")
    _, (identity,) = added(store, "People/Hans Müller.md", document)
    assert identity == first_row(COLLECTION) + 1
    with reading(store) as transaction:
        found = stored(transaction, "People/Hans Müller.md")
        assert found is not None
        assert (found.id, found.key, found.digest, found.title) == (identity, "", document.digest, "Hans")
        assert text_of(transaction, found.id) == document.text
        assert fields_of(found.fields) == {"aliases": ["Hansi"], "org": "[[Acme AG]]"}
        chunk = document.chunks[0]
        assert chunks_of(transaction, identity) == [
            StoredChunk(
                ChunkId(first_row(COLLECTION) + 1), chunk.first_line, chunk.last_line, chunk.start, chunk.length
            )
        ]


def test_a_document_s_links_and_names_are_stored_as_they_were_read(store: sqlite3.Connection) -> None:
    document = note(NOTE, "Hans Müller.md")
    _, (identity,) = added(store, "People/Hans Müller.md", document)
    with reading(store) as transaction:
        links = rows(
            transaction,
            LinkRow,
            LINKS_OF,
            (identity,),
        )
        names = rows(transaction, NameRow, "SELECT key, kind FROM names WHERE document = ? ORDER BY key", (identity,))
    assert links == [
        LinkRow(0, "org", 3, "wikilink", "Acme AG", "acme ag", None, None),
        LinkRow(1, None, 7, "wikilink", "Acme AG", "acme ag", "Team", "Acme"),
        LinkRow(2, None, 7, "markdown", "Other.md", "other", None, "x"),
        LinkRow(3, None, 7, "embed", "scan.pdf", "scan.pdf", None, None),
    ]
    assert names == [NameRow("hans müller", "name"), NameRow("hansi", "alias")]
    assert {link.kind for link in links} == set(LinkKind)
    assert {name.kind for name in names} == set(NameKind)


def test_a_record_of_a_file_answers_to_no_name(store: sqlite3.Connection) -> None:
    record = dataclasses.replace(note(b"text\n", "data.jsonl"), key="7")
    added(store, "data.jsonl", record)
    with reading(store) as transaction:
        assert stored(transaction, "data.jsonl", "7") is not None
        assert stored(transaction, "data.jsonl") is None
    assert counted(store, "names") == 0


def test_the_documents_of_a_source_are_listed_with_their_digests(store: sqlite3.Connection) -> None:
    first, second = (dataclasses.replace(note(text, "d.jsonl"), key=key) for text, key in ((b"a", "1"), (b"b", "2")))
    source, ids = added(store, "d.jsonl", first, second)
    with reading(store) as transaction:
        assert kept(transaction, source) == [
            KeptDocument(ids[0], "1", first.digest),
            KeptDocument(ids[1], "2", second.digest),
        ]


def test_deleting_a_document_deletes_all_that_is_its(store: sqlite3.Connection) -> None:
    _, (identity,) = added(store, "Hans.md", note(NOTE, "Hans.md"))
    added(store, "Other.md", note(b"# Other\n[[Hans]]\n", "Other.md"))
    with writing_documents(store) as transaction:
        delete_documents(transaction, [identity])
    assert [counted(store, table) for table in ("documents", "chunks", "links", "names")] == [1, 1, 1, 1]
    with reading(store) as transaction:
        assert stored(transaction, "Hans.md") is None
        assert stored(transaction, "Other.md") is not None


def test_deleting_a_source_deletes_its_documents_and_all_that_is_theirs(store: sqlite3.Connection) -> None:
    added(store, "Hans.md", note(NOTE, "Hans.md"))
    added(store, "kept.md", note(b"kept", "kept.md"))
    with writing_documents(store) as transaction:
        assert delete_sources(transaction, ["Hans.md", "gone.md"]) == 1
        assert delete_sources(transaction, ["kept.md"]) == 1
    assert [counted(store, table) for table in ("sources", "documents", "chunks", "links", "names")] == [0] * 5


def test_a_source_holds_one_document_of_each_key(store: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        added(store, "a.md", note(b"a", "a.md"), note(b"b", "a.md"))
    assert counted(store, "documents") == 0


@pytest.mark.parametrize(
    "statement, refusal",
    [
        (
            "INSERT INTO links(document, ordinal, line, kind, target, key) VALUES (:document, 9, 1, 'href', 't', 'k')",
            "CHECK constraint failed",
        ),
        ("INSERT INTO names(key, document, kind) VALUES ('k', :document, 'nickname')", "CHECK constraint failed"),
        (
            "INSERT INTO chunks(document, first_line, last_line, start, length) VALUES (:document + 99, 1, 1, 0, 1)",
            "FOREIGN KEY constraint failed",
        ),
        ("INSERT INTO sources(path, modified_ns, size) VALUES ('a', 'yesterday', 1)", "cannot store TEXT value"),
        (
            "INSERT INTO documents(source, key, digest, title, fields) VALUES (:source, 'k', 'd', 't', '{')",
            "CHECK constraint failed",
        ),
    ],
)
def test_the_store_refuses_rows_that_break_its_rules(store: sqlite3.Connection, statement: str, refusal: str) -> None:
    source, (document,) = added(store, "a.md", note(b"a", "a.md"))
    with pytest.raises(sqlite3.IntegrityError, match=refusal), writing(store) as transaction:
        run(transaction, statement, {"source": source, "document": document})


def test_fields_are_kept_as_json_whatever_their_mapping_and_sequence() -> None:
    written = fields_json(MappingProxyType({"a": (1, MappingProxyType({"b": "Zürich"})), "c": range(2)}))
    assert written == '{"a": [1, {"b": "Zürich"}], "c": [0, 1]}'
    assert fields_of(written) == {"a": [1, {"b": "Zürich"}], "c": [0, 1]}


@pytest.mark.parametrize("value", [float("nan"), cast("FieldValue", b"bytes")])
def test_what_json_cannot_keep_is_no_field_value(value: FieldValue) -> None:
    with pytest.raises((ValueError, TypeError)):
        fields_json({"x": value})


def record(key: str) -> Document:
    return dataclasses.replace(note(b"x", "data.jsonl"), key=key)


def kept_counts(connection: sqlite3.Connection) -> Counts:
    with reading(connection) as transaction:
        return counts(transaction)


@pytest.mark.parametrize("written", ["[]", '"text"', "7"])
def test_front_matter_that_is_no_mapping_is_damage(written: str) -> None:
    with pytest.raises(DamagedError, match="no mapping"):
        fields_of(written)


def test_the_largest_row_of_a_note_or_record_fits_a_row_sqlite_stores(store: sqlite3.Connection) -> None:
    most_characters_as_utf8, title_and_fields = 4, 2
    largest_row = max(BLOCK, MOST_LINE, MOST_HEAD) * most_characters_as_utf8 * title_and_fields
    assert largest_row < store.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)


def test_the_store_keeps_count_of_its_documents_and_chunks_as_they_come_and_go(store: sqlite3.Connection) -> None:
    assert kept_counts(store) == Counts(0, 0)
    _, (hans,) = added(store, "Hans.md", note(NOTE, "Hans.md"))
    added(store, "Other.md", note(b"# Other\n\n" + b"A line of words.\n" * 400, "Other.md"))
    expected = Counts(counted(store, "documents"), counted(store, "chunks"))
    assert kept_counts(store) == expected
    assert expected.chunks > expected.documents == 2
    with writing_documents(store) as transaction:
        delete_documents(transaction, [hans])
    assert kept_counts(store) == Counts(counted(store, "documents"), counted(store, "chunks"))
    with writing_documents(store) as transaction:
        delete_sources(transaction, ["Other.md"])
    assert kept_counts(store) == Counts(0, 0)


def test_a_count_that_would_fall_below_zero_is_refused(store: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"), writing(store) as transaction:
        run(transaction, "UPDATE counts SET documents = -1")


def test_a_document_id_is_never_handed_out_twice(store: sqlite3.Connection) -> None:
    _, (first_id,) = added(store, "a.md", note(b"a", "a.md"))
    with writing_documents(store) as transaction:
        delete_documents(transaction, [first_id])
    _, (second_id,) = added(store, "a.md", note(b"a", "a.md"))
    assert second_id == first_id + 1


def test_a_whole_file_is_found_by_the_last_part_of_its_path_in_any_case(store: sqlite3.Connection) -> None:
    for path in ("a/b/c.md", "x/b/c.md", "c.md", "a/bc.md", "a/b/C.MD"):
        added(store, path, note(b"x", path))
    with writing_documents(store) as transaction:
        put_source(transaction, "y/c.md", 1, 1)
        add_documents(transaction, put_source(transaction, "data.jsonl", 1, 1), "data.jsonl", [record("c.md")])
    with reading(store) as transaction:
        assert paths_named(transaction, "c.md") == ["a/b/C.MD", "a/b/c.md", "c.md", "x/b/c.md"]
        assert paths_named(transaction, "b") == []


def test_a_source_s_name_follows_its_path_to_the_end(store: sqlite3.Connection) -> None:
    for path in ("end/", "plain"):
        added(store, path, note(b"x", path))
    with reading(store) as transaction:
        assert (paths_named(transaction, ""), paths_named(transaction, "plain")) == (["end/"], ["plain"])


def test_unreadable_files_are_listed_with_their_problem_until_cleared(store: sqlite3.Connection) -> None:
    with writing(store) as transaction:
        mark_unreadable(transaction, "b.md", "denied", read_again=True)
        mark_unreadable(transaction, "a.md", "denied", read_again=True)
        mark_unreadable(transaction, "a.md", "too large", read_again=False)
    with reading(store) as transaction:
        assert unreadable(transaction) == [Unreadable("a.md", "too large", 0), Unreadable("b.md", "denied", 1)]
    with writing(store) as transaction:
        clear_unreadable(transaction, ["a.md", "gone.md"])
    with reading(store) as transaction:
        assert unreadable(transaction) == [Unreadable("b.md", "denied", 1)]


class Step(NamedTuple):
    id: int
    parent: int
    unused: int
    detail: str


def test_a_shortened_id_is_looked_up_by_the_index_of_names_not_by_reading_every_source(
    store: sqlite3.Connection,
) -> None:
    sql = f"EXPLAIN QUERY PLAN {PATHS_NAMED}"
    with reading(store) as transaction:
        steps = [step.detail for step in rows(transaction, Step, sql, ("c.md",))]
    assert steps[0] == "SEARCH s USING INDEX sources_name (name=?)"
    assert not [step for step in steps if step.startswith("SCAN")]
