import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sift.domain.plan.ignore import NOTHING_LEFT_OUT, Ignore, left_out
from sift.ingest.ignore import IGNORE_FILE, NOT_A_FILE, IgnoreUnreadableError, ignore_at
from tests.files import linked


def written(folder: Path, raw: bytes, platform: str = "linux") -> Ignore:
    (folder / IGNORE_FILE).write_bytes(raw)
    return ignore_at(folder, platform=platform)


def test_a_folder_without_a_siftignore_leaves_nothing_out(tmp_path: Path) -> None:
    assert ignore_at(tmp_path) == NOTHING_LEFT_OUT


@pytest.mark.parametrize("platform, folded", [("win32", True), ("darwin", True), ("linux", False)])
def test_case_counts_in_a_siftignore_where_the_file_system_counts_it(
    tmp_path: Path, platform: str, *, folded: bool
) -> None:
    assert left_out(written(tmp_path, b"Drafts/\n", platform), "drafts", folder=True) == folded


def test_a_siftignore_is_read_in_the_encoding_it_is_written_in(tmp_path: Path) -> None:
    marked = written(tmp_path, "\ufeff# exports\r\n*.pdf\r\n".encode())
    assert (left_out(marked, "exports/x.pdf"), left_out(marked, "x.md")) == (True, False)
    assert left_out(written(tmp_path, "drafts/\r\n".encode("utf-16")), "drafts", folder=True)
    assert left_out(written(tmp_path, "Entwürfe/\n".encode("cp1252")), "Entwürfe", folder=True)


@pytest.mark.parametrize("separator", ["\u2028", "\x0c", "\x85"])
def test_a_siftignore_ends_lines_at_newlines_alone_as_git_does(tmp_path: Path, separator: str) -> None:
    ignore = written(tmp_path, f"Notes{separator}draft/\r\n*.log\r\n".encode())
    asked = ((f"Notes{separator}draft", True), ("draft", True), ("a.log", False))
    assert [left_out(ignore, path, folder=folder) for path, folder in asked] == [True, False, True]


def test_a_siftignore_that_is_no_file_cannot_be_read(tmp_path: Path) -> None:
    (tmp_path / IGNORE_FILE).mkdir()
    with pytest.raises(IgnoreUnreadableError, match=NOT_A_FILE):
        ignore_at(tmp_path)


def test_a_siftignore_larger_than_sift_reads_cannot_be_read(tmp_path: Path) -> None:
    (tmp_path / IGNORE_FILE).write_bytes(b"*.md\n" * 3)
    assert left_out(ignore_at(tmp_path, 15), "a.md")
    with pytest.raises(IgnoreUnreadableError, match="more than the 14 bytes sift reads of it"):
        ignore_at(tmp_path, 14)


def test_a_siftignore_linked_to_nothing_cannot_be_read(tmp_path: Path) -> None:
    linked(tmp_path / IGNORE_FILE, str(tmp_path / "gone"))
    with pytest.raises(FileNotFoundError):
        ignore_at(tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows'")
def test_a_siftignore_that_is_a_junction_to_nothing_cannot_be_read(tmp_path: Path) -> None:
    command, target = shutil.which("cmd"), tmp_path / "gone"
    assert command is not None
    target.mkdir()
    _ = subprocess.run([command, "/c", "mklink", "/J", str(tmp_path / IGNORE_FILE), str(target)], check=True)
    target.rmdir()
    with pytest.raises(FileNotFoundError):
        ignore_at(tmp_path)
