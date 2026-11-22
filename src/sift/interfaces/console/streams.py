import codecs
import contextlib
import errno
import io
import os
from collections.abc import Iterable
from typing import TextIO

UTF8 = "utf-8"
READER_GONE = frozenset({errno.EPIPE, errno.EINVAL})


class ReaderLeftError(Exception):
    stream: TextIO

    def __init__(self, stream: TextIO) -> None:
        super().__init__("whoever read the results has left")
        self.stream = stream


def utf8(stream: TextIO | None) -> None:
    if isinstance(stream, io.TextIOWrapper) and codecs.lookup(stream.encoding).name != UTF8:
        stream.reconfigure(encoding=UTF8)


def _text(lines: Iterable[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


def write_results(stream: TextIO | None, lines: Iterable[str]) -> None:
    if stream is None:
        return
    try:
        stream.write(_text(lines))
        stream.flush()
    except OSError as error:
        if error.errno in READER_GONE:
            raise ReaderLeftError(stream) from None
        raise


def say(stream: TextIO | None, lines: Iterable[str]) -> None:
    if stream is None:
        return
    with contextlib.suppress(OSError, ValueError):
        stream.write(_text(lines))
        stream.flush()


def let_go(stream: TextIO) -> None:
    try:
        target = stream.fileno()
    except (OSError, ValueError):
        return
    nowhere = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(nowhere, target)
    finally:
        os.close(nowhere)
