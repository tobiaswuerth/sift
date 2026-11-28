import json
import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import fitted, labelled, printable, printable_json, printable_text, table, width

UNSEEN = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
JOINER = "\N{ZERO WIDTH JOINER}"

WRITING = frozenset(
    {
        "\N{ZERO WIDTH NON-JOINER}",
        "\N{LEFT-TO-RIGHT MARK}",
        "\N{RIGHT-TO-LEFT MARK}",
        "\N{ARABIC LETTER MARK}",
        "\N{SOFT HYPHEN}",
    }
)


@pytest.mark.parametrize(
    "text, columns",
    [
        ("", 0),
        ("notes", 5),
        ("日本", 4),
        ("e\N{COMBINING ACUTE ACCENT}", 1),
        ("a\N{ZERO WIDTH SPACE}b", 2),
        ("\N{FULLWIDTH LATIN CAPITAL LETTER A}", 2),
        ("\N{THUMBS UP SIGN}", 2),
        ("\N{HEAVY BLACK HEART}\N{VARIATION SELECTOR-16}", 2),
        ("\N{THUMBS UP SIGN}\N{EMOJI MODIFIER FITZPATRICK TYPE-4}", 2),
        ("\N{MAN}\N{ZERO WIDTH JOINER}\N{WOMAN}\N{ZERO WIDTH JOINER}\N{GIRL}", 2),
        ("\N{HANGUL CHOSEONG KIYEOK}\N{HANGUL JUNGSEONG A}\N{HANGUL JONGSEONG KIYEOK}", 2),
        ("a\N{EMOJI MODIFIER FITZPATRICK TYPE-4}", 3),
    ],
)
def test_width_counts_the_columns_a_terminal_gives_a_text(text: str, columns: int) -> None:
    assert width(text) == columns


@given(st.text(st.characters(codec="ascii", categories=["L", "N", "P", "S", "Zs"])))
def test_printable_ascii_takes_a_column_a_character(text: str) -> None:
    assert width(text) == len(text)


@pytest.mark.parametrize(
    "text, shown",
    [
        ("plain Müller 日本", "plain Müller 日本"),
        ("a\nb\tc", "a\\nb\\tc"),
        ("\x1b[2J", "\\x1b[2J"),
        ("\x85", "\\x85"),
        ("\N{RIGHT-TO-LEFT OVERRIDE}evil", "\\u202eevil"),
        ("\N{LEFT-TO-RIGHT ISOLATE}a\N{POP DIRECTIONAL ISOLATE}", "\\u2066a\\u2069"),
        ("a\N{ZERO WIDTH SPACE}b\N{LINE SEPARATOR}c\N{PARAGRAPH SEPARATOR}", "a\\u200bb\\u2028c\\u2029"),
        ("\N{MAN}\N{ZERO WIDTH JOINER}\N{WOMAN}", "\N{MAN}\N{ZERO WIDTH JOINER}\N{WOMAN}"),
        ("\U000e0041", "\\U000e0041"),
        ("\N{RIGHT-TO-LEFT EMBEDDING}a", "\\u202ba"),
    ],
)
def test_control_characters_are_shown_as_escapes_so_no_text_can_steer_the_terminal(text: str, shown: str) -> None:
    assert printable(text) == shown


@pytest.mark.parametrize("kept", sorted(WRITING))
def test_marks_writing_needs_are_shown_as_they_are(kept: str) -> None:
    assert (printable(f"a{kept}b"), printable_text(f"a{kept}b")) == (f"a{kept}b", f"a{kept}b")


def test_half_a_surrogate_pair_which_no_stream_can_encode_is_shown_as_an_escape() -> None:
    assert printable("name" + chr(0xDCE9) + ".md") == "name\\udce9.md"


@given(st.text(st.characters(codec=None)))
def test_a_printable_text_holds_no_control_character_and_can_be_encoded(text: str) -> None:
    shown = printable(text)
    assert not any(unicodedata.category(each) in UNSEEN for each in shown if each != JOINER and each not in WRITING)
    assert shown.encode().decode() == shown


def test_json_holds_what_could_steer_a_terminal_as_json_escapes_that_read_back_the_same() -> None:
    text = "a\x85\N{RIGHT-TO-LEFT OVERRIDE}\U000e0041 日本"
    written = json.dumps({"t": text}, ensure_ascii=False)
    shown = printable_json(written)
    assert shown == '{"t": "a\\u0085\\u202e\\udb40\\udc41 日本"}'
    assert json.loads(shown) == {"t": text}


def test_every_character_that_can_steer_a_terminal_or_hide_text_is_escaped_and_no_other() -> None:
    every = [chr(code) for code in range(0x110000)]
    steering = {character for character in every if unicodedata.category(character) in UNSEEN} - {JOINER, *WRITING}
    assert {character for character in every if printable(character) != character} == steering


@pytest.mark.parametrize(
    "text, shown",
    [
        ("one\r\n\ttwo\n", "one\r\n\ttwo\n"),
        ("\x1b]52;c;aGk=\x07 and \x1b[2J\r", "\\x1b]52;c;aGk=\\x07 and \\x1b[2J\\r"),
        ("\N{RIGHT-TO-LEFT OVERRIDE}", "\\u202e"),
    ],
)
def test_a_text_shown_whole_keeps_its_lines_and_tabs_and_nothing_else_that_steers_the_terminal(
    text: str, shown: str
) -> None:
    assert printable_text(text) == shown


def test_a_table_aligns_its_columns_by_the_width_a_terminal_shows() -> None:
    assert table((("name", "count"), ("日本", "7"), ("notes", "12,431")), right=frozenset({1})) == (
        "name    count",
        "日本        7",
        "notes  12,431",
    )


def test_a_table_shows_its_cells_as_printable_text() -> None:
    assert table((("a\nb", "c"),), right=frozenset()) == ("a\\nb  c",)


@given(st.lists(st.tuples(st.text(), st.text()), min_size=1))
def test_the_rows_of_a_table_end_in_one_column(rows: list[tuple[str, str]]) -> None:
    lines = table(tuple(rows), right=frozenset({1}))
    assert len(lines) == len(rows)
    assert len({width(line) for line in lines}) == 1


def test_a_labelled_line_aligns_its_label_to_the_right() -> None:
    assert labelled("fix", "sift index notes") == "  fix  sift index notes"
    assert labelled("error", "a\nb") == "error  a\\nb"


def test_a_line_is_fitted_to_the_columns_a_terminal_has() -> None:
    assert fitted("notes 日本 x", 7) == "notes "
    assert fitted("notes 日本 x", 8) == "notes 日"
    assert fitted("short", 80) == "short"


@given(st.text(), st.integers(0, 40))
def test_a_fitted_line_never_takes_more_columns_than_there_are(text: str, columns: int) -> None:
    shown = fitted(text, columns)
    assert width(shown) <= columns
    assert text.startswith(shown)


def test_a_label_takes_its_tone_where_colour_is_on() -> None:
    assert labelled("fix", "x", Tone.SECONDARY, Style(colour=True)) == "  \x1b[2mfix\x1b[0m  x"
    assert labelled("fix", "x", Tone.SECONDARY) == "  fix  x"


def test_a_label_wider_than_the_label_columns_is_refused() -> None:
    assert labelled("error", "x") == "error  x"
    with pytest.raises(ValueError, match="'nothing' is wider than the 5 columns labels take"):
        labelled("nothing", "x")
