import pytest

from sift.api.errors import InvalidRequestError
from sift.domain.text.lines import line_range, refuse_past_end


@pytest.mark.parametrize("written, span", [("1", (1, 1)), ("2-3", (2, 3)), ("17-26", (17, 26)), ("40", (40, 40))])
def test_a_line_range_is_a_line_or_two_joined_by_a_dash(written: str, span: tuple[int, int]) -> None:
    assert line_range(written) == span


@pytest.mark.parametrize("written", ["0", "3-2", "a", "1-", "-3", " 1", "1 - 2", "1.5", "\N{SUPERSCRIPT TWO}"])
def test_anything_else_is_refused_with_what_a_range_looks_like(written: str) -> None:
    with pytest.raises(InvalidRequestError, match="lines are a line number or a range such as 17-26"):
        line_range(written)


def test_lines_past_the_end_are_refused_with_how_many_there_are() -> None:
    refuse_past_end("notes/rl.md", 3, first=3)
    with pytest.raises(InvalidRequestError, match=r"notes/rl\.md has 3 lines; line 9 is past its end"):
        refuse_past_end("notes/rl.md", 3, first=9)
