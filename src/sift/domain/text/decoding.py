import codecs
import itertools
import re
from collections.abc import Iterable, Iterator
from types import MappingProxyType

from sift.domain.text.json_values import LONE_SURROGATE

MARK_BYTES = 4
NEWLINE, WINDOWS_NEWLINE, CARRIAGE_RETURN = "\n", "\r\n", "\r"
UTF32_MARKS = (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")
UTF16_MARKS = (b"\xff\xfe", b"\xfe\xff")
ESCAPED_BYTES = range(0xDC80, 0xDD00)
ESCAPED = re.compile("[\udc80-\udcff]")
ANSI = MappingProxyType({point: bytes([point - 0xDC00]).decode("cp1252", "replace") for point in ESCAPED_BYTES})


def decoded(raw: bytes) -> str:
    if raw.startswith(UTF32_MARKS):
        return raw.decode("utf-32", "replace")
    if raw.startswith(UTF16_MARKS):
        return raw.decode("utf-16", "replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("utf-8-sig", "surrogateescape").translate(ANSI)


def _newlines(held: str) -> tuple[str, str]:
    kept = CARRIAGE_RETURN if held.endswith(CARRIAGE_RETURN) else ""
    return held[: len(held) - len(kept)].replace(WINDOWS_NEWLINE, NEWLINE), kept


def text_pieces(raw: Iterable[bytes]) -> Iterator[str]:
    parts = iter(raw)
    start = b""
    for part in parts:
        start += part
        if len(start) >= MARK_BYTES:
            break
    encoding, errors = encoding_of(start)
    decoder = codecs.getincrementaldecoder(encoding)(errors)
    held = ""
    for part in itertools.chain((start,), parts):
        piece, held = _newlines(held + ansi(decoder.decode(part)))
        if piece:
            yield piece
    last = (held + ansi(decoder.decode(b"", final=True))).replace(WINDOWS_NEWLINE, NEWLINE)
    if last:
        yield last


def encoding_of(start: bytes) -> tuple[str, str]:
    if start.startswith(UTF32_MARKS):
        return "utf-32", "replace"
    if start.startswith(UTF16_MARKS):
        return "utf-16", "replace"
    return "utf-8-sig", "surrogateescape"


def ansi(text: str) -> str:
    return text.translate(ANSI) if ESCAPED.search(text) else text


def encodable(text: str) -> bool:
    return LONE_SURROGATE.search(text) is None
