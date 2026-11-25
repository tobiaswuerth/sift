from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.collections import AddCollection, DeactivateCollections
from sift.api.errors import InvalidRequestError
from sift.api.indexing import FileIndexed, IndexCollections
from sift.app.collections import add, deactivate
from sift.app.filing import LIMITS, Limits, Readers
from sift.app.indexing import index
from sift.domain.plan.indexing import Found
from sift.ingest.scan import Reach, Scanned
from sift.storage.catalog import known_collections
from sift.storage.database import reading, rows
from sift.storage.store import connected
from tests.files import TORN, jsonl, linked, torn_lines, write
from tests.homes import found_ids, quiet
from tests.later import LATER, later


class Keyed(NamedTuple):
    key: str
    id: int


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    write(tmp_path / "papers" / "rl.md", "# Reward shaping\n\nShaped rewards.\n")
    write(tmp_path / "recipes" / "soup.md", "# Tomato soup\n\nWith salt.\n")
    write(tmp_path / "diary" / "monday.md", "# Monday\n\nThe robot broke.\n")
    write(tmp_path / "diary" / "ignored" / "x.md", "# X\n\nLeft out.\n")
    write(tmp_path / "diary" / ".siftignore", "ignored/\n")
    for name in ("papers", "recipes", "diary"):
        add(home, AddCollection(str(tmp_path / name)))
    index(home, IndexCollections(), quiet, readers=LATER)
    return home


def reindexed(home: Path, *files: str, readers: Readers = LATER, limits: Limits = LIMITS) -> list[FileIndexed]:
    result = index(home, IndexCollections(files=files), quiet, limits, readers)
    return [each for collection in result.collections for each in collection.named]


def test_files_of_several_collections_are_read_at_once_whatever_their_dates(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "papers" / "rl.md", "# Reward shaping\n\nShaped rewards, changed.\n")
    write(tmp_path / "recipes" / "soup.md", "# Tomato soup\n\nWith basil now.\n")
    write(tmp_path / "papers" / "new.md", "# New\n\nA new paper.\n")
    files = ("recipes/soup.md", str(tmp_path / "papers" / "rl.md"), "diary/monday.md", "papers/new.md")
    assert sorted((each.id, each.state) for each in reindexed(home, *files)) == [
        ("diary/monday.md", "unchanged"),
        ("papers/new.md", "added"),
        ("papers/rl.md", "changed"),
        ("recipes/soup.md", "changed"),
    ]
    assert found_ids(home, "basil") == ["recipes/soup.md"]


def test_a_file_gone_is_dropped_and_one_never_there_is_said_missing(home: Path, tmp_path: Path) -> None:
    (tmp_path / "diary" / "monday.md").unlink()
    assert [(each.id, each.state) for each in reindexed(home, "diary/monday.md", "diary/tuesday.md")] == [
        ("diary/monday.md", "removed"),
        ("diary/tuesday.md", "missing"),
    ]
    assert found_ids(home, "robot") == []


def test_a_file_the_collection_leaves_out_is_not_read(home: Path) -> None:
    ((left),) = reindexed(home, "diary/ignored/x.md")
    assert (left.state, left.note) == ("left out", "its collection's rules leave it out")
    assert found_ids(home, "left") == []


def test_a_file_of_an_inactive_collection_is_read_for_when_it_is_active_again(home: Path, tmp_path: Path) -> None:
    deactivate(home, DeactivateCollections(("diary",)))
    write(tmp_path / "diary" / "tuesday.md", "# Tuesday\n\nNothing broke.\n")
    assert [each.state for each in reindexed(home, "diary/tuesday.md")] == ["added"]


def test_a_file_that_cannot_be_looked_at_is_kept_and_said_unreadable(home: Path) -> None:
    def refused(_root: Path, relative: str, _reach: Reach) -> Scanned:
        return Scanned(Found({}, unlisted_files=(relative,)), refusals={relative: "Access is denied"})

    ((kept),) = reindexed(home, "diary/monday.md", readers=Readers(clock=later, scan_one=refused))
    assert (kept.state, kept.note) == ("unreadable", "Access is denied")
    assert found_ids(home, "robot") == ["diary/monday.md"]


def test_a_file_that_leads_outside_its_folder_is_left_out_and_dropped_if_it_was_in(home: Path) -> None:
    def outside(_root: Path, relative: str, _reach: Reach) -> Scanned:
        return Scanned(Found({}, outside=(relative,)))

    readers = Readers(clock=later, scan_one=outside)
    assert [(each.state, each.note) for each in reindexed(home, "diary/monday.md", "diary/x.md", readers=readers)] == [
        ("removed", "it leads outside its collection's folder"),
        ("missing", None),
    ]


def test_a_link_given_by_its_path_that_leads_outside_is_left_out_saying_so(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "elsewhere" / "x.md", "# Elsewhere")
    linked(tmp_path / "diary" / "link.md", str(tmp_path / "elsewhere" / "x.md"))
    ((left),) = reindexed(home, str(tmp_path / "diary" / "link.md"))
    assert (left.state, left.note) == ("left out", "it leads outside its collection's folder")


def test_a_file_its_reader_refuses_is_said_unreadable_with_why(home: Path, tmp_path: Path) -> None:
    (tmp_path / "papers" / "big.pdf").write_bytes(b"%PDF-1.4 " + b"x" * 200)
    ((refused),) = reindexed(home, "papers/big.pdf", limits=Limits(most_file_bytes=100))
    assert (refused.state, refused.note) == (
        "unreadable",
        "209 bytes: sift reads documents of at most 100 bytes; split the file",
    )


def test_records_named_say_the_lines_they_skip_whether_or_not_they_changed(home: Path, tmp_path: Path) -> None:
    records, skipped = tmp_path / "papers" / "p.jsonl", "1 line skipped (line 3: not an object)"
    write(records, jsonl([{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]) + "[1, 2]\n")
    assert reindexed(home, "papers/p.jsonl") == [FileIndexed("papers/p.jsonl", "added", skipped)]
    assert reindexed(home, "papers/p.jsonl") == [FileIndexed("papers/p.jsonl", "unchanged", skipped)]
    write(records, jsonl([{"id": "a", "text": "first"}]))
    assert reindexed(home, "papers/p.jsonl") == [FileIndexed("papers/p.jsonl", "changed")]
    assert found_ids(home, "second") == []


def test_records_named_but_not_read_to_their_end_are_said_unreadable_and_read_again(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "papers" / "p.jsonl", jsonl([{"id": n, "text": f"part {n}"} for n in range(3)]))
    torn = Readers(clock=later, lines=torn_lines)
    one_a_write = Limits(most_records_a_write=1)
    assert reindexed(home, "papers/p.jsonl", readers=torn, limits=one_a_write) == [
        FileIndexed("papers/p.jsonl", "unreadable", TORN)
    ]
    assert found_ids(home, "part") == ["papers/p.jsonl#0", "papers/p.jsonl#1"]
    assert reindexed(home, "papers/p.jsonl") == [FileIndexed("papers/p.jsonl", "changed")]
    assert found_ids(home, "part") == ["papers/p.jsonl#0", "papers/p.jsonl#1", "papers/p.jsonl#2"]


def test_records_changed_in_a_file_read_again_keep_the_others_as_they_were(home: Path, tmp_path: Path) -> None:
    records = tmp_path / "papers" / "p.jsonl"
    write(records, "".join(f'{{"id": "r{n}", "text": "record {n}"}}\n' for n in range(3)))
    reindexed(home, "papers/p.jsonl")
    (papers,) = [one for one in known_collections(home) if one.name == "papers"]
    sql = "SELECT key, id FROM documents WHERE key != '' ORDER BY key"
    with connected(papers.store, papers.id) as connection, reading(connection) as transaction:
        before = rows(transaction, Keyed, sql)
    write(records, "".join(f'{{"id": "r{n}", "text": "record {n}{" changed" * (n == 1)}"}}\n' for n in range(3)))
    assert [each.state for each in reindexed(home, "papers/p.jsonl#r1")] == ["changed"]
    with connected(papers.store, papers.id) as connection, reading(connection) as transaction:
        after = rows(transaction, Keyed, sql)
    assert [each for each in after if each.key != "r1"] == [each for each in before if each.key != "r1"]


def test_collections_and_files_are_not_named_together_and_a_file_must_lie_in_a_collection(
    home: Path, tmp_path: Path
) -> None:
    with pytest.raises(InvalidRequestError, match="collections or files, not both"):
        index(home, IndexCollections(("papers",), files=("papers/rl.md",)), quiet)
    write(tmp_path / "elsewhere.md", "x")
    for outside in (str(tmp_path / "elsewhere.md"), "nowhere/x.md"):
        with pytest.raises(InvalidRequestError, match="is in no collection"):
            index(home, IndexCollections(files=(outside,)), quiet)
