from collections.abc import Callable
from pathlib import Path

CATALOG = "catalog.sqlite"
FORMER_CATALOG = "sift.db"
WRITER_SUFFIX = ".writer"
PART_SUFFIX = ".part"
FILE_SUFFIXES = ("", "-wal", "-shm", WRITER_SUFFIX, f"{WRITER_SUFFIX}-journal", PART_SUFFIX, f"{PART_SUFFIX}-journal")
DATA_SUFFIXES = ("", "-wal")
LOG_SUFFIXES = ("-wal", "-shm")
OPENED_SUFFIXES = ("", *LOG_SUFFIXES)


def _files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return [file for file in (Path(f"{path}{suffix}") for suffix in suffixes) if file.is_file()]


def delete_store(path: Path) -> None:
    for file in _files(path, FILE_SUFFIXES):
        file.unlink()


def delete_logs(path: Path) -> None:
    for file in _files(path, LOG_SUFFIXES):
        file.unlink()


def unlinked(file: Path) -> None:
    file.unlink(missing_ok=True)


def store_files_left(
    path: Path, unlink: Callable[[Path], None] = unlinked, suffixes: tuple[str, ...] = FILE_SUFFIXES
) -> tuple[Path, ...]:
    left: list[Path] = []
    for file in _files(path, suffixes):
        try:
            unlink(file)
        except PermissionError:
            left.append(file)
    return tuple(left)


def _size(file: Path) -> int:
    try:
        return file.stat().st_size
    except FileNotFoundError:
        return 0


def stored_bytes(path: Path) -> int:
    return sum(_size(Path(f"{path}{suffix}")) for suffix in DATA_SUFFIXES)
