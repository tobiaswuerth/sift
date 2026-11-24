import io
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sift.api.events import CollectionStarted, Phase, PhaseDone, PhaseStarted
from sift.interfaces.cli.logs import Log, LogLimits, heard_too, kept
from sift.interfaces.console.live import Live
from sift.interfaces.console.progress import Looks
from sift.interfaces.console.streams import say
from sift.interfaces.console.style import PLAIN, Style

NOW = datetime(2026, 10, 23, 21, 5, 9, tzinfo=UTC)
TIMED = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} ")
LOOKS = Looks(PLAIN, "linux")


def lines(path: Path) -> list[str]:
    return [TIMED.sub("", line, count=1) for line in path.read_text(encoding="utf-8").split("\n")[:-1]]


def test_a_log_holds_what_sift_said_and_what_the_libraries_logged_each_line_with_its_time(tmp_path: Path) -> None:
    with kept(tmp_path / "logs", "index", NOW, "sift 2.0.0, index") as log:
        say(log, ("one", "\x1b[1mtwo\x1b[0m"))
        logging.getLogger("a.library").warning("a library's record")
        logging.getLogger("a.library").info("what it did")
    logging.getLogger("a.library").warning("after the command")
    assert log.path == tmp_path / "logs" / "index-2026-10-23_210509.log"
    assert all(TIMED.match(line) for line in log.path.read_text(encoding="utf-8").split("\n")[:-1])
    assert lines(log.path) == [
        "INFO sift: sift 2.0.0, index",
        "INFO sift: one",
        "INFO sift: two",
        "WARNING a.library: a library's record",
        "INFO a.library: what it did",
    ]


def test_what_sift_says_is_logged_once_even_where_the_root_logger_prints(tmp_path: Path) -> None:
    printed = io.StringIO()
    printing = logging.StreamHandler(printed)
    logging.getLogger().addHandler(printing)
    try:
        with kept(tmp_path, "index", NOW, "sift") as log:
            say(log, ("3 added",))
    finally:
        logging.getLogger().removeHandler(printing)
    assert ("3 added" in printed.getvalue(), lines(log.path)[-1]) == (False, "INFO sift: 3 added")


ENDINGS: tuple[tuple[Callable[[Log], None], tuple[str, ...]], ...] = (
    (lambda log: log.ended(0), ("INFO sift: ended with exit code 0",)),
    (lambda log: log.ended(2), ("ERROR sift: ended with exit code 2",)),
    (Log.stopped, ("ERROR sift: stopped",)),
)


@pytest.mark.parametrize("end, said", ENDINGS)
def test_a_log_ends_saying_how_its_command_ended(
    tmp_path: Path, end: Callable[[Log], None], said: tuple[str, ...]
) -> None:
    with kept(tmp_path, "embed", NOW, "sift") as log:
        end(log)
    assert tuple(lines(log.path)[1:]) == said


def broke() -> None:
    raise ValueError("bad")


def test_a_command_that_fails_unforeseen_logs_why_and_where(tmp_path: Path) -> None:
    with kept(tmp_path, "embed", NOW, "sift") as log:
        try:
            broke()
        except ValueError:
            log.failed()
    found = lines(log.path)
    assert (found[1], found[-1]) == ("ERROR sift: failed", "ValueError: bad")


def test_two_commands_started_in_one_second_get_a_file_each_and_one_there_is_left_as_it_is(tmp_path: Path) -> None:
    taken = tmp_path / "index-2026-10-23_210509.log"
    taken.write_bytes(b"another command's\n")
    with kept(tmp_path, "index", NOW, "sift") as first, kept(tmp_path, "index", NOW, "sift") as second:
        pass
    assert (first.path.name, second.path.name) == ("index-2026-10-23_210509-2.log", "index-2026-10-23_210509-3.log")
    assert taken.read_bytes() == b"another command's\n"


def test_the_latest_logs_of_each_command_are_kept_with_their_parts(tmp_path: Path) -> None:
    older = [f"index-2026-10-0{day}_120000.log" for day in range(1, 6)]
    for name in [*older, "index-2026-10-01_120000.log.1", "embed-2026-10-01_120000.log"]:
        (tmp_path / name).write_bytes(b"x")
    with kept(tmp_path, "index", NOW, "sift", LogLimits(kept=3)) as log:
        pass
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "embed-2026-10-01_120000.log",
        *older[3:],
        log.path.name,
    ]


def test_a_log_another_sift_still_writes_is_deleted_once_it_can_be(tmp_path: Path) -> None:
    old = tmp_path / "index-2026-10-01_120000.log"
    with old.open("w", encoding="utf-8") as held:
        held.write("a server's log\n")
        with kept(tmp_path, "index", NOW, "sift", LogLimits(kept=1)):
            pass
    with kept(tmp_path, "index", NOW, "sift", LogLimits(kept=1)):
        pass
    assert not old.exists()


def test_a_long_log_goes_on_in_parts_the_oldest_deleted(tmp_path: Path) -> None:
    with kept(tmp_path, "index", NOW, "sift", LogLimits(most_bytes=200)) as log:
        for number in range(40):
            say(log, (f"line {number:02} " + "x" * 50,))
    parts = sorted(path.name for path in tmp_path.iterdir())
    assert parts == [log.path.name, *(f"{log.path.name}.{number}" for number in range(1, 4))]
    assert lines(log.path)[-1].endswith("line 39 " + "x" * 50)
    assert all(path.stat().st_size <= 200 for path in tmp_path.iterdir())


def embedding(tmp_path: Path, times: tuple[float, ...]) -> tuple[str, list[str]]:
    screen, clock = io.StringIO(), iter(times).__next__
    with kept(tmp_path, "embed", NOW, "sift") as log:
        drawn = Live(screen, Looks(Style(colour=True), "linux"), redraw=True, columns=lambda: 100, clock=clock)
        both = heard_too(drawn, log, clock)
        both(CollectionStarted("a", "/a"))
        both(PhaseStarted(Phase.EMBED, 3))
        both.tick()
        both.tick()
        both(PhaseDone(3))
    return screen.getvalue(), [line for line in lines(log.path) if line.startswith("INFO sift:   embed")]


def test_a_long_step_goes_into_the_log_every_minute_while_the_screen_draws_it_in_place(tmp_path: Path) -> None:
    screen, embeds = embedding(tmp_path, (0.0, 0.0, 0.0, 0.0, 61.0, 61.0, 62.0, 62.0, 62.0, 62.0))
    assert (len(embeds), embeds[-1].endswith(" 1 m 2 s"), "\r" in screen, "\x1b[" in screen) == (2, True, True, True)
    assert "\x1b[" not in "".join(embeds)


def test_a_step_shorter_than_a_minute_goes_into_the_log_once_where_it_ended(tmp_path: Path) -> None:
    _, embeds = embedding(tmp_path, (0.0, 0.0, 0.0, 0.0, 30.0, 30.0, 31.0, 31.0, 31.0, 31.0))
    assert (len(embeds), embeds[0].endswith(" 31 s"), "3 chunks" in embeds[0]) == (1, True, True)
