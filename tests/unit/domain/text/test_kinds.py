import pytest

from sift.domain.text.kinds import KINDS, FileKind, keyed_path, kind_of, public_id, split_keyed_path, streamed


@pytest.mark.parametrize(
    "path, kind",
    [
        ("a.md", FileKind.NOTE),
        ("x/Notes.MD", FileKind.NOTE),
        ("a.markdown", FileKind.NOTE),
        ("a.TXT", FileKind.NOTE),
        ("data.jsonl", FileKind.RECORDS),
        ("x/d.NDJSON", FileKind.RECORDS),
        ("SCAN.PDF", FileKind.DOCUMENT),
        ("x/page.htm", FileKind.DOCUMENT),
        ("book.epub", FileKind.DOCUMENT),
        ("offer.EML", FileKind.MAIL),
        (".md", None),
        ("x/.jsonl", None),
        ("a.md/b", None),
        ("a.csv", None),
        ("a", None),
    ],
)
def test_a_file_s_kind_is_that_of_its_extension_in_any_case(path: str, kind: FileKind | None) -> None:
    assert kind_of(path) is kind


def test_every_kind_is_read_from_some_extension_and_only_records_are_streamed() -> None:
    assert set(KINDS.values()) == set(FileKind)
    named = ("a.md", "a.txt", "a.jsonl", "a.ndjson", "a.pdf", "a.eml")
    assert [path for path in named if streamed(path)] == ["a.jsonl", "a.ndjson"]


@pytest.mark.parametrize(
    "path, key, written",
    [
        ("a.md", "", "a.md"),
        ("data.jsonl", "2101.00001", "data.jsonl#2101.00001"),
        ("x/d.NDJSON", "hep-th/9901001#2", "x/d.NDJSON#hep-th/9901001#2"),
        ("a#b.md", "", "a#b.md"),
        ("c#d.jsonl", "1", "c#d.jsonl#1"),
        ("mail/offer.eml", "Fwd.eml/contract #2.pdf", "mail/offer.eml#Fwd.eml/contract #2.pdf"),
        ("scan.pdf", "", "scan.pdf"),
    ],
)
def test_a_document_s_path_names_its_file_and_its_key(path: str, key: str, written: str) -> None:
    assert keyed_path(path, key) == written
    assert split_keyed_path(written) == (path, key)
    assert public_id("c", path, key) == f"c/{written}"
