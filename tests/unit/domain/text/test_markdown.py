import re

from hypothesis import given
from hypothesis import strategies as st

from sift.domain.text.markdown import title_of


def test_a_title_of_several_lines_is_one_line_with_each_break_and_the_blanks_around_it_one_space() -> None:
    assert title_of(" a  \r\n\t b\n\nc ") == "a b c"
    assert title_of("a\n \n b") == "a  b"


@given(st.text(st.sampled_from("ab \t\r\n"), max_size=40))
def test_a_title_is_made_one_line_as_the_pattern_of_breaks_and_their_blanks_would(title: str) -> None:
    assert title_of(title) == re.sub(r"[ \t]*[\r\n]+[ \t]*", " ", title.strip())


def test_a_title_with_a_megabyte_of_blanks_before_a_break_is_one_line_at_once() -> None:
    assert title_of("a" + " \t" * 500_000 + "\nb") == "a b"
