import multiprocessing
from pathlib import Path

from sift.api.collections import TitleFrom
from sift.domain.text.documents import Again, Conversion, Refused, Unread
from sift.ingest.readers import ToRead, read_file
from sift.ingest.sandbox import ORPHANED, ends_with
from tests.waits import joined


def panicked(_raw: bytes, _name: str) -> Conversion:
    return Unread("the converter failed: index out of bounds", next_release=True)


def refused(_raw: bytes, _name: str) -> Conversion:
    return Unread("Parsing error")


def test_a_file_a_library_failed_on_is_read_again_under_the_next_release_and_one_it_refused_never(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF")
    asked = ToRead(path, 4, 100, None, TitleFrom.HEADING)
    assert read_file(asked, converting=panicked) == Refused(
        "the converter failed: index out of bounds", Again.NEXT_RELEASE
    )
    assert read_file(asked, converting=refused) == Refused("Parsing error", Again.NEVER)


def test_a_reader_whose_parent_is_gone_ends_itself_saying_it_was_left() -> None:
    parent, ended = multiprocessing.get_context("spawn").Process(target=int), list[int]()
    parent.start()
    joined(parent, 30)
    ends_with(parent.sentinel, ended.append)
    assert ended == [ORPHANED]
