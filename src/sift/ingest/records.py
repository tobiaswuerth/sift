import io
from collections.abc import Iterator
from pathlib import Path

from sift.domain.text.decoding import ansi, encoding_of
from sift.domain.text.records import SkippedLine

MARK_BYTES = 4
LINE_END = "\r\n"


def _rest_skipped(text: io.TextIOWrapper, most: int) -> None:
    part = text.readline(most + 1)
    while part and not part.endswith("\n"):
        part = text.readline(most + 1)


def lines_of(path: Path, most: int) -> Iterator[tuple[int, str | SkippedLine]]:
    with path.open("rb") as raw:
        encoding, errors = encoding_of(raw.read(MARK_BYTES))
        raw.seek(0)
        with io.TextIOWrapper(raw, encoding=encoding, errors=errors, newline="\n") as text:
            number = 0
            while line := text.readline(most + len(LINE_END)):
                number += 1
                if len(line.rstrip(LINE_END)) > most:
                    if not line.endswith("\n"):
                        _rest_skipped(text, most)
                    yield number, SkippedLine(number, f"longer than the {most:,} characters a line may hold")
                else:
                    yield number, ansi(line).strip()
