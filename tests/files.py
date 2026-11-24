import json
import stat
import sys
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from itertools import islice
from pathlib import Path
from types import MappingProxyType

import pytest

from sift.domain.text.records import SkippedLine
from sift.ingest.files import Identity
from sift.ingest.records import lines_of

TORN = "the drive went away"


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def jsonl(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(json.dumps(each) + "\n" for each in records)


def torn_lines(path: Path, most: int, scanned: Identity | None) -> Iterator[tuple[int, str | SkippedLine]]:
    yield from islice(lines_of(path, most, scanned), 2)
    raise OSError(5, TORN)


def linked(link: Path, target: str, *, folder: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=folder)
    except OSError as refused:
        pytest.skip(f"this system makes no symbolic links: {refused}")


@contextmanager
def _held_open(path: Path) -> Generator[None]:
    with path.open("rb"):
        yield


@contextmanager
def _in_a_read_only_folder(path: Path) -> Generator[None]:
    path.parent.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        yield
    finally:
        path.parent.chmod(stat.S_IRWXU)


UNDELETABLE: Mapping[str, Callable[[Path], AbstractContextManager[None]]] = MappingProxyType({"win32": _held_open})


def undeletable(path: Path) -> AbstractContextManager[None]:
    return UNDELETABLE.get(sys.platform, _in_a_read_only_folder)(path)
