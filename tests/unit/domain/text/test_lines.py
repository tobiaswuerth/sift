import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.api.errors import InvalidRequestError
from sift.domain.text.lines import line_range, lines_of

TEXT = "# Reward shaping\n\nReward shaping speeds up learning.\n"


@pytest.mark.parametrize("written, span", [("1", (1, 1)), ("2-3", (2, 3)), ("17-26", (17, 26)), ("40", (40, 40))])
def test_a_line_range_is_a_line_or_two_joined_by_a_dash(written: str, span: tuple[int, int]) -> None:
    assert line_range(written) == span


@pytest.mark.parametrize("written", ["0", "3-2", "a", "1-", "-3", " 1", "1 - 2", "1.5", "\N{SUPERSCRIPT TWO}"])
def test_anything_else_is_refused_with_what_a_range_looks_like(written: str) -> None:
    with pytest.raises(InvalidRequestError, match="lines are a line number or a range such as 17-26"):
        line_range(written)


@pytest.mark.parametrize(
    "span, text",
    [((1, 1), "# Reward shaping\n"), ((2, 3), "\nReward shaping speeds up learning.\n"), ((3, 99), TEXT[18:])],
)
def test_lines_are_cut_as_written_with_their_endings(span: tuple[int, int], text: str) -> None:
    assert lines_of(TEXT, *span, "notes/rl.md") == text


def test_a_last_line_without_an_ending_is_a_line() -> None:
    assert lines_of("a\nb", 2, 2, "x") == "b"


def test_lines_past_the_end_are_refused_with_how_many_there_are() -> None:
    with pytest.raises(InvalidRequestError, match=r"notes/rl\.md has 3 lines; line 9 is past its end"):
        lines_of(TEXT, 9, 12, "notes/rl.md")


@given(st.text(alphabet="ab\n\r\N{LINE SEPARATOR}"))
def test_every_line_cut_one_by_one_gives_back_the_text(text: str) -> None:
    count = text.count("\n") + (not text.endswith("\n") and text != "")
    assert "".join(lines_of(text, n, n, "x") for n in range(1, count + 1)) == text
