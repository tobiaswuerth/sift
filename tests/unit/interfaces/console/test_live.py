import io

from sift.api.events import Advanced, CollectionStarted, Event, Phase, PhaseDone, PhaseStarted
from sift.interfaces.console.live import Live
from sift.interfaces.console.progress import Looks
from sift.interfaces.console.style import PLAIN

LOOKS = Looks(PLAIN, "linux")
EVENTS: tuple[Event, ...] = (
    CollectionStarted("a", "/a"),
    PhaseStarted(Phase.READ, 2),
    Advanced(1),
    Advanced(2),
    PhaseDone(2),
)
TIMES = (0.0, 0.0, 0.05, 0.2, 0.3)
ROW = "  read           2 files       6.7/s     0.3 s"


def written(*, redraw: bool) -> str:
    out = io.StringIO()
    live = Live(out, LOOKS, redraw=redraw, columns=lambda: 100, clock=iter(TIMES).__next__)
    for event in EVENTS:
        live(event)
    return out.getvalue()


def test_in_a_terminal_the_running_phase_is_drawn_again_in_place() -> None:
    shown = written(redraw=True)
    assert shown.startswith("a  /a\n\r  read           0 files   ")
    assert "\r  read           2 files   " in shown
    assert shown.endswith("\r" + ROW + "\n")


def test_progress_is_drawn_at_most_ten_times_a_second() -> None:
    assert "  read           1 files" not in written(redraw=True)


def test_a_finished_line_takes_the_place_of_the_running_one() -> None:
    shown = written(redraw=True)
    drawn = shown.split("\r")[-3]
    assert shown.split("\r")[-2] == " " * len(drawn)


def test_elsewhere_only_finished_lines_are_written() -> None:
    assert written(redraw=False) == f"a  /a\n{ROW}\n"


def test_closing_clears_a_line_still_drawn() -> None:
    out = io.StringIO()
    live = Live(out, LOOKS, redraw=True, columns=lambda: 100, clock=iter((0.0, 0.0)).__next__)
    live(PhaseStarted(Phase.SCAN, None))
    drawn = out.getvalue()
    live.close()
    live.close()
    assert out.getvalue() == drawn + "\r" + " " * (len(drawn) - 1) + "\r"


def test_without_a_stream_nothing_is_written_and_nothing_fails() -> None:
    live = Live(None, LOOKS, redraw=True, columns=lambda: 100, clock=iter(TIMES).__next__)
    for event in EVENTS:
        live(event)
    assert live.drawn == 0
