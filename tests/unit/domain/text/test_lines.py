import pytest

from sift.api.errors import InvalidRequestError
from sift.domain.text.lines import refuse_past_end


def test_lines_past_the_end_are_refused_with_how_many_there_are() -> None:
    refuse_past_end("notes/rl.md", 3, first=3)
    with pytest.raises(InvalidRequestError, match=r"notes/rl\.md has 3 lines; line 9 is past its end"):
        refuse_past_end("notes/rl.md", 3, first=9)


def test_a_line_asked_of_a_document_without_text_is_refused_saying_it_holds_none() -> None:
    with pytest.raises(InvalidRequestError, match=r"notes/empty\.md holds no text; line 1 is past its end") as past:
        refuse_past_end("notes/empty.md", 0, first=1)
    assert past.value.remedy == "ask for it without lines"
