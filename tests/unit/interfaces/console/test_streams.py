import io
from pathlib import Path

from sift.interfaces.console.streams import DEFAULT_COLUMNS, columns, is_terminal


def test_a_stream_that_is_no_terminal_has_the_usual_width(tmp_path: Path) -> None:
    assert columns(io.StringIO()) == DEFAULT_COLUMNS
    with (tmp_path / "out.txt").open("w", encoding="utf-8") as file:
        assert columns(file) == DEFAULT_COLUMNS
        assert not is_terminal(file)


def test_a_closed_stream_or_none_is_no_terminal() -> None:
    closed = io.StringIO()
    closed.close()
    assert not is_terminal(closed)
    assert not is_terminal(None)
