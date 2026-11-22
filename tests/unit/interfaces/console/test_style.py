import pytest

from sift.interfaces.console.style import PLAIN, Asked, Colour, Style, Terminal, Tone, paint, style_of

TERMINAL = Terminal(terminal=True, escapes=True, encoding="utf-8")
AUTO = Asked(Colour.AUTO, no_color=False, dumb=False)


def test_paint_colours_a_text_only_where_colour_is_on() -> None:
    assert paint("done", Tone.DONE, Style(colour=True)) == "\x1b[32mdone\x1b[0m"
    assert paint("done", Tone.DONE, PLAIN) == "done"
    assert paint("", Tone.DONE, Style(colour=True)) == ""


def test_a_terminal_that_takes_escapes_gets_colour() -> None:
    assert style_of(TERMINAL, AUTO) == Style(colour=True, unicode=True)


@pytest.mark.parametrize(
    "stream, asked",
    [
        (Terminal(terminal=False, escapes=True, encoding="utf-8"), AUTO),
        (Terminal(terminal=True, escapes=False, encoding="utf-8"), AUTO),
        (TERMINAL, Asked(Colour.AUTO, no_color=True, dumb=False)),
        (TERMINAL, Asked(Colour.AUTO, no_color=False, dumb=True)),
        (TERMINAL, Asked(Colour.NEVER, no_color=False, dumb=False)),
    ],
)
def test_colour_is_off_in_a_pipe_without_escapes_with_no_color_a_dumb_terminal_or_when_asked(
    stream: Terminal, asked: Asked
) -> None:
    assert not style_of(stream, asked).colour


def test_colour_asked_for_is_given_anywhere() -> None:
    piped = Terminal(terminal=False, escapes=False, encoding="utf-8")
    assert style_of(piped, Asked(Colour.ALWAYS, no_color=True, dumb=True)).colour


@pytest.mark.parametrize(
    "encoding, unicode",
    [("utf-8", True), ("UTF8", True), (None, True), ("ascii", False), ("cp1252", False), ("x", False)],
)
def test_glyphs_are_drawn_where_the_stream_can_encode_them(encoding: str | None, *, unicode: bool) -> None:
    stream = Terminal(terminal=True, escapes=True, encoding=encoding)
    assert style_of(stream, AUTO).unicode is unicode
