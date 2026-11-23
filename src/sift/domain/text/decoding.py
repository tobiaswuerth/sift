from types import MappingProxyType

UTF32_MARKS = (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")
UTF16_MARKS = (b"\xff\xfe", b"\xfe\xff")
ESCAPED_BYTES = range(0xDC80, 0xDD00)
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


def file_text(raw: bytes) -> str:
    return decoded(raw).replace("\r\n", "\n")


def encoding_of(start: bytes) -> tuple[str, str]:
    if start.startswith(UTF32_MARKS):
        return "utf-32", "replace"
    if start.startswith(UTF16_MARKS):
        return "utf-16", "replace"
    return "utf-8-sig", "surrogateescape"


def ansi(text: str) -> str:
    return text.translate(ANSI)
