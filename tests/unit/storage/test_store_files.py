from pathlib import Path

import pytest

from sift.storage.store_files import delete_store, store_files_left


def test_deleting_a_store_deletes_its_database_log_and_shared_memory(tmp_path: Path) -> None:
    for name in ("s.sqlite", "s.sqlite-wal", "s.sqlite-shm", "s.sqlite.writer", "other.sqlite"):
        (tmp_path / name).write_bytes(b"x")
    delete_store(tmp_path / "s.sqlite")
    assert [path.name for path in tmp_path.iterdir()] == ["other.sqlite"]


def test_deleting_a_store_that_is_not_there_does_nothing(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"x")
    delete_store(tmp_path / "gone.sqlite")
    delete_store(tmp_path / "file" / "s.sqlite")
    assert store_files_left(tmp_path / "gone.sqlite") == ()
    assert [path.name for path in tmp_path.iterdir()] == ["file"]


def test_the_files_of_a_store_a_program_holds_open_are_left_and_named(tmp_path: Path) -> None:
    for name in ("s.sqlite", "s.sqlite-wal", "s.sqlite-shm"):
        (tmp_path / name).write_bytes(b"x")

    def held_open(file: Path) -> None:
        if file.name == "s.sqlite":
            raise PermissionError(13, "The process cannot access the file", str(file))
        file.unlink()

    assert store_files_left(tmp_path / "s.sqlite", held_open) == (tmp_path / "s.sqlite",)
    assert [path.name for path in tmp_path.iterdir()] == ["s.sqlite"]


def test_a_store_file_that_cannot_be_deleted_for_another_reason_says_that_reason(tmp_path: Path) -> None:
    (tmp_path / "s.sqlite").write_bytes(b"x")

    def broken(file: Path) -> None:
        raise OSError(5, "Input/output error", str(file))

    with pytest.raises(OSError, match="Input/output error"):
        store_files_left(tmp_path / "s.sqlite", broken)
