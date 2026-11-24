import shutil
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from sift.ingest.files import CHANGED, READ_AT_ONCE, ChangedError, at_most, identity_at
from tests.files import write


def test_a_file_is_read_only_while_it_is_the_one_scanned(tmp_path: Path) -> None:
    path, other = write(tmp_path / "a.md", "first"), write(tmp_path / "b.md", "second")
    scanned = identity_at(path)
    assert (at_most(path, 10, scanned), at_most(path, 10, None)) == (b"first", b"first")
    other.replace(path)
    with pytest.raises(ChangedError, match="read again at the next index"):
        at_most(path, 10, scanned)
    assert at_most(path, 3, None) == b"seco"


def test_a_small_file_is_read_without_a_buffer_the_size_of_the_limit(tmp_path: Path) -> None:
    path = write(tmp_path / "a.md", "small")
    tracemalloc.start()
    try:
        assert at_most(path, 64 << 20, None) == b"small"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * READ_AT_ONCE


def test_a_file_longer_than_one_read_is_read_whole_or_cut_one_byte_past_the_limit(tmp_path: Path) -> None:
    path = tmp_path / "long.md"
    path.write_bytes(bytes(range(256)) * (5 * READ_AT_ONCE // 512))
    whole = path.read_bytes()
    assert at_most(path, 4 * READ_AT_ONCE, None) == whole
    assert at_most(path, READ_AT_ONCE + 7, None) == whole[: READ_AT_ONCE + 8]


@pytest.mark.skipif(sys.platform == "win32", reason="named pipes on Windows live apart from files")
def test_a_pipe_where_a_file_was_scanned_is_named_changed_not_waited_for(tmp_path: Path) -> None:
    command = shutil.which("mkfifo")
    assert command is not None
    _ = subprocess.run([command, str(tmp_path / "note.md")], check=True)
    with pytest.raises(ChangedError, match=CHANGED):
        at_most(tmp_path / "note.md", 100, None)


def test_a_folder_where_a_file_was_scanned_is_named_changed(tmp_path: Path) -> None:
    (tmp_path / "note.md").mkdir()
    with pytest.raises(OSError, match=CHANGED if sys.platform != "win32" else "denied"):
        at_most(tmp_path / "note.md", 100, None)
