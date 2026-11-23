import codecs
import cProfile

from hypothesis import assume, given
from hypothesis import strategies as st

from sift.domain.text.decoding import decoded, encoding_of, file_text

MARKS_OF_OTHER_ENCODINGS = (b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff")
BYTE_ORDER_MARK = "\N{ZERO WIDTH NO-BREAK SPACE}"


def ansi_as_sift_1(error: UnicodeError) -> tuple[str, int]:
    if not isinstance(error, UnicodeDecodeError):
        raise error
    return error.object[error.start : error.end].decode("cp1252", "replace"), error.end


codecs.register_error("tests.sift_1_ansi", ansi_as_sift_1)


def builtins_called(raw: bytes) -> set[str]:
    profile = cProfile.Profile()
    profile.runcall(decoded, raw)
    return {entry.code for entry in profile.getstats() if isinstance(entry.code, str)}


@given(st.binary(max_size=200))
def test_bytes_that_are_no_utf8_read_as_sift_1_read_them(raw: bytes) -> None:
    assume(not raw.startswith(MARKS_OF_OTHER_ENCODINGS))
    assert decoded(raw) == raw.decode("utf-8-sig", "tests.sift_1_ansi")


@given(st.text(max_size=100))
def test_utf8_reads_as_it_was_written_without_its_byte_order_mark(text: str) -> None:
    assert decoded(text.encode()) == text.removeprefix(BYTE_ORDER_MARK)


def test_a_byte_order_mark_says_the_encoding() -> None:
    assert decoded(f"{BYTE_ORDER_MARK}a".encode()) == "a"
    assert decoded(f"{BYTE_ORDER_MARK}a".encode("utf-16-le")) == "a"
    assert decoded(f"{BYTE_ORDER_MARK}a".encode("utf-16-be")) == "a"
    assert decoded(f"{BYTE_ORDER_MARK}a".encode("utf-32-le")) == "a"
    assert decoded(f"{BYTE_ORDER_MARK}a".encode("utf-32-be")) == "a"


def test_only_text_that_is_no_utf8_is_passed_through_character_by_character() -> None:
    assert not any("translate" in name for name in builtins_called("Zürich, 5 €".encode()))
    assert any("translate" in name for name in builtins_called("Zürich".encode("cp1252")))


def test_a_file_s_lines_end_in_newlines_alone() -> None:
    assert file_text(b"a\r\nb\rc\n") == "a\nb\rc\n"


def test_the_encoding_of_a_file_read_line_by_line_is_told_by_its_first_bytes() -> None:
    assert encoding_of("{".encode("utf-32")) == ("utf-32", "replace")
    assert encoding_of("{".encode("utf-16")) == ("utf-16", "replace")
    assert encoding_of(b"{") == ("utf-8-sig", "surrogateescape")
