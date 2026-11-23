import io
from collections.abc import Iterator
from pathlib import Path

from sift.domain.text.decoding import ansi, encoding_of

MARK_BYTES = 4


def lines_of(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("rb") as raw:
        encoding, errors = encoding_of(raw.read(MARK_BYTES))
        raw.seek(0)
        with io.TextIOWrapper(raw, encoding=encoding, errors=errors, newline="\n") as text:
            for number, line in enumerate(text, 1):
                yield number, ansi(line).strip()
