import pytest

from sift.api.errors import InvalidRequestError
from sift.interfaces.cli.reading import line_range


@pytest.mark.parametrize("written, span", [("1", (1, 1)), ("2-3", (2, 3)), ("17-26", (17, 26)), ("40", (40, 40))])
def test_a_line_range_is_a_line_or_two_joined_by_a_dash(written: str, span: tuple[int, int]) -> None:
    assert line_range(written) == span


@pytest.mark.parametrize("written", ["0", "3-2", "a", "1-", "-3", " 1", "1 - 2", "1.5", "\N{SUPERSCRIPT TWO}"])
def test_anything_else_is_refused_with_what_a_range_looks_like(written: str) -> None:
    with pytest.raises(InvalidRequestError, match="lines are a line number or a range such as 17-26"):
        line_range(written)


@pytest.mark.parametrize("digits", [19, 5_000])
def test_a_line_of_more_digits_than_a_line_number_has_is_refused_as_no_range(digits: int) -> None:
    for written in ("1" * digits, f"3-{'1' * digits}"):
        with pytest.raises(InvalidRequestError, match="lines are a line number or a range such as 17-26"):
            line_range(written)
    assert line_range("9" * 18) == (int("9" * 18), int("9" * 18))
