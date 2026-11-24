import json
from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from pathlib import Path

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
