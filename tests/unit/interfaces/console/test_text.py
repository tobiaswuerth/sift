import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import fitted, labelled, printable, table, width


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
    ],
)
def test_width_counts_the_columns_a_terminal_gives_a_text(text: str, columns: int) -> None:
    assert width(text) == columns


@given(st.text(st.characters(codec="ascii", categories=["L", "N", "P", "S", "Zs"])))
def test_printable_ascii_takes_a_column_a_character(text: str) -> None:
    assert width(text) == len(text)


@pytest.mark.parametrize(
    "text, shown",
    [("plain Müller 日本", "plain Müller 日本"), ("a\nb\tc", "a\\nb\\tc"), ("\x1b[2J", "\\x1b[2J"), ("\x85", "\\x85")],
)
def test_control_characters_are_shown_as_escapes_so_no_text_can_steer_the_terminal(text: str, shown: str) -> None:
    assert printable(text) == shown


@given(st.text())
def test_a_printable_text_holds_no_control_character(text: str) -> None:
    assert not any(unicodedata.category(character) == "Cc" for character in printable(text))


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
