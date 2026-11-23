import json
from collections.abc import Iterator
from contextlib import closing
from itertools import islice
from pathlib import Path

import pytest

from sift.api.collections import AddCollection, Counted, ShowStatus
from sift.api.documents import GetDocuments
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError
from sift.api.events import Event, Noted
from sift.api.filters import written
from sift.api.indexing import IndexCollections, Indexed
from sift.api.search import SearchKeywords
from sift.app.collections import add, status
from sift.app.documents import documents
from sift.app.embedding import embed
from sift.app.filing import Limits, Lines, Readers
from sift.app.indexing import index
from sift.app.search import search
from sift.domain.text.records import SkippedLine
from sift.ingest.files import Identity
from sift.ingest.records import lines_of
from sift.storage.catalog import known_collections
from sift.storage.database import reading
from sift.storage.documents import counts
from sift.storage.store import open_store
from sift.storage.vectors import left
from tests.files import jsonl, write
from tests.later import later
from tests.models import Hashed

PAPERS: list[dict[str, object]] = [
    {"id": "2101.00001", "title": "Brush stroke planning", "abstract": "A robot plans brush strokes.", "year": 2021},
    {"id": "2101.00002", "title": "Reward shaping", "abstract": "Shaping speeds up learning.", "year": 2021},
    {"id": "hep-th/9901001", "title": "String junctions", "abstract": "Heterotic strings and junctions.", "year": 1999},
]
BRUSH, SHAPING = "arxiv/2021/2021-01.jsonl#2101.00001", "arxiv/2021/2021-01.jsonl#2101.00002"
STRINGS = "arxiv/1999/1999-01.jsonl#hep-th/9901001"


def indexed(
    home: Path, limits: Limits | None = None, said: list[Event] | None = None, lines: Lines = lines_of
) -> Indexed:
    heard: list[Event] = [] if said is None else said
    (one,) = index(
        home, IndexCollections(), heard.append, limits or Limits(), Readers(lines=lines, clock=later)
    ).collections
    assert isinstance(one.outcome, Indexed)
    return one.outcome


def found(home: Path, query: str, filters: tuple[str, ...] = ()) -> list[str]:
    asked = SearchKeywords(query, k=20, filters=tuple(map(written, filters)))
    return sorted(hit.id for hit in search(home, asked).hits)


def text_of(home: Path, id_: str) -> str:
    (one,) = documents(home, GetDocuments((id_,))).found
    return one.text


@pytest.fixture
def papers(tmp_path: Path) -> Path:
    root, home = tmp_path / "arxiv", tmp_path / "home"
    write(root / "2021" / "2021-01.jsonl", jsonl(PAPERS[:2]))
    write(root / "1999" / "1999-01.jsonl", jsonl(PAPERS[2:]))
    write(root / "notes.md", "# Notes\n\nA Markdown file in the same collection.\n")
    add(home, AddCollection(str(root)))
    indexed(home)
    return home


def test_each_line_of_a_json_lines_file_is_a_document_named_by_its_file_and_id(papers: Path) -> None:
    assert found(papers, "brush") == [BRUSH]
    assert text_of(papers, BRUSH) == "# Brush stroke planning\n\nA robot plans brush strokes."
    assert found(papers, "strings") == [STRINGS]
    assert found(papers, "learning OR strokes OR junctions", ("year=1999",)) == [STRINGS]
    (shown,) = documents(papers, GetDocuments((BRUSH,), fields=("year", "authors"))).found
    assert shown.fields == {"year": 2021}


def test_only_changed_records_are_stored_again_and_gone_ones_leave(papers: Path, tmp_path: Path) -> None:
    root = tmp_path / "arxiv"
    changed = [PAPERS[0], {**PAPERS[1], "abstract": "Shaping is now different."}, {"id": "2101.00003", "text": "New."}]
    write(root / "2021" / "2021-01.jsonl", jsonl(changed))
    again = indexed(papers)
    assert (again.added, again.changed, again.removed) == (1, 1, 0)
    assert text_of(papers, SHAPING) == "# Reward shaping\n\nShaping is now different."
    assert found(papers, "speeds") == []
    write(root / "2021" / "2021-01.jsonl", jsonl(PAPERS[:1]))
    write(root / "1999" / "1999-01.jsonl", "")
    gone = indexed(papers)
    assert (gone.added, gone.changed, gone.removed) == (0, 0, 3)
    assert found(papers, "junctions OR shaping OR new") == []
    unchanged = indexed(papers)
    assert (unchanged.added, unchanged.changed, unchanged.removed) == (0, 0, 0)


def test_vectors_of_unchanged_records_are_kept(papers: Path, tmp_path: Path) -> None:
    embed(papers, EmbedCollections(), lambda _: None, Hashed())
    write(tmp_path / "arxiv" / "2021" / "2021-01.jsonl", jsonl([PAPERS[0], {**PAPERS[1], "abstract": "New text."}]))
    indexed(papers)
    (one,) = known_collections(papers)
    with closing(open_store(one.store, one.id)) as connection, reading(connection) as transaction:
        assert left(transaction) == 1


def test_lines_without_a_record_are_skipped_and_said(tmp_path: Path) -> None:
    lines = ["{not json", "[1, 2]", json.dumps({"id": 1}), json.dumps({"id": 2, "text": "kept"}), ""]
    lines += [json.dumps({"id": 2, "text": "again"}), json.dumps({"id": "a\x00b", "text": "nul"})]
    lines += [json.dumps({"id": "a\ufffdb", "text": "same key"}), "[]"]
    write(tmp_path / "c" / "r.jsonl", "\n".join(lines) + "\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "c")))
    said: list[Event] = []
    indexed(home, said=said)
    assert found(home, "kept OR again OR nul OR same") == ["c/r.jsonl#2", "c/r.jsonl#a\ufffdb"]
    assert text_of(home, "c/r.jsonl#a\ufffdb") == "same key"
    (noted,) = [each for each in said if isinstance(each, Noted)]
    assert noted.text.startswith("r.jsonl: 6 lines skipped (line 1: not JSON; line 2: not an object; line 3: no text")
    assert noted.text.endswith("; ...)")
    (contents,) = [each.contents for each in status(home, ShowStatus(unreadable=True)).collections]
    assert isinstance(contents, Counted)
    assert [(each.path, each.problem) for each in contents.unreadable_files] == [("r.jsonl", noted.text[9:])]


def test_fields_may_be_chosen_when_the_collection_is_added(tmp_path: Path) -> None:
    write(tmp_path / "c" / "qa.jsonl", json.dumps({"key": "q1", "q": "Why?", "a": "Because.", "text": "not this"}))
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "c"), id_field="key", title_field="q", text_fields=("q", "a")))
    indexed(home)
    assert text_of(home, "c/qa.jsonl#q1") == "# Why?\n\nWhy?\n\nBecause."


@pytest.mark.parametrize("id_field, title_field, text_fields", [(None, None, ("",)), (" ", None, ()), (None, "", ())])
def test_a_field_chosen_must_be_named(
    tmp_path: Path, id_field: str | None, title_field: str | None, text_fields: tuple[str, ...]
) -> None:
    (tmp_path / "c").mkdir()
    asked = AddCollection(str(tmp_path / "c"), id_field=id_field, title_field=title_field, text_fields=text_fields)
    with pytest.raises(InvalidRequestError, match="field"):
        add(tmp_path / "home", asked)


def test_a_large_file_is_stored_a_batch_at_a_time_as_it_is_read(tmp_path: Path) -> None:
    records = [{"id": n % 9, "text": f"record {n}"} for n in range(12)]
    write(tmp_path / "c" / "big.jsonl", jsonl(records))
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "c")))
    small = Limits(most_records_a_write=4)
    said: list[Event] = []
    held: list[int] = []

    def counted(path: Path, most: int, scanned: Identity | None) -> Iterator[tuple[int, str | SkippedLine]]:
        (one,) = known_collections(home)
        for line in lines_of(path, most, scanned):
            with closing(open_store(one.store, one.id)) as connection, reading(connection) as transaction:
                held.append(counts(transaction).documents)
            yield line

    first = indexed(home, small, said, counted)
    assert (first.added, first.removed, held[0], held[-1]) == (9, 0, 0, 8)
    assert text_of(home, "c/big.jsonl#3") == "record 3"
    assert any(isinstance(each, Noted) and "line 10: id 0 used twice" in each.text for each in said)
    write(tmp_path / "c" / "big.jsonl", jsonl([*records[:5], {"id": 5, "text": "changed"}]))
    second = indexed(home, small)
    assert (second.added, second.changed, second.removed) == (0, 1, 3)
    assert text_of(home, "c/big.jsonl#5") == "changed"


def test_a_record_longer_than_a_document_may_be_is_skipped_and_files_in_utf_16_are_read(tmp_path: Path) -> None:
    lines = [json.dumps({"id": 1, "text": "x" * 50}), '{"id": 2, "text": "' + "\N{EURO SIGN}" * 15 + '"}']
    write(tmp_path / "c" / "r.jsonl", "\n".join([*lines, json.dumps({"id": 3, "text": "short"})]) + "\n")
    (tmp_path / "c" / "wide.jsonl").write_bytes(jsonl([{"id": "w", "text": "wide text"}]).encode("utf-16"))
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "c")))
    said: list[Event] = []
    indexed(home, Limits(most_text_bytes=40), said)
    assert found(home, "short OR wide") == ["c/r.jsonl#3", "c/wide.jsonl#w"]
    (noted,) = [each.text for each in said if isinstance(each, Noted)]
    assert "line 1: longer than the 40 characters a line may hold" in noted
    assert "line 2: 45 bytes as text, more than the 40 a document holds" in noted


def broken(path: Path, most: int, scanned: Identity | None) -> Iterator[tuple[int, str | SkippedLine]]:
    yield from islice(lines_of(path, most, scanned), 2)
    raise OSError(5, "the drive went away")


def test_a_file_that_cannot_be_read_to_its_end_keeps_what_was_read_and_is_read_again(tmp_path: Path) -> None:
    write(tmp_path / "c" / "r.jsonl", jsonl([{"id": n, "text": f"part {n}"} for n in range(3)]))
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "c")))
    assert indexed(home, Limits(most_records_a_write=1), lines=broken).added == 2
    assert found(home, "part") == ["c/r.jsonl#0", "c/r.jsonl#1"]
    (contents,) = [each.contents for each in status(home, ShowStatus(unreadable=True)).collections]
    assert isinstance(contents, Counted)
    assert [(each.path, each.problem) for each in contents.unreadable_files] == [("r.jsonl", "the drive went away")]
    again = indexed(home)
    assert (again.added, again.changed, again.removed, again.unreadable) == (1, 0, 0, 0)
    assert found(home, "part") == ["c/r.jsonl#0", "c/r.jsonl#1", "c/r.jsonl#2"]
