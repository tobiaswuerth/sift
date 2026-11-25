import pytest

from sift.api.errors import InvalidRequestError
from sift.domain.text.lines import refuse_past_end


def test_lines_past_the_end_are_refused_with_how_many_there_are() -> None:
    refuse_past_end("notes/rl.md", 3, first=3)
    with pytest.raises(InvalidRequestError, match=r"notes/rl\.md has 3 lines; line 9 is past its end"):
        refuse_past_end("notes/rl.md", 3, first=9)
