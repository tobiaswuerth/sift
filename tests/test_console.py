"""What sift tells the user while it works, on stderr: messages, plurals, durations, and one kind of progress line."""

import gc
import io
import sys
import threading

import pytest

from sift import console


def test_say_writes_to_stderr(capsys):
    console.say("hello", 42)
    assert capsys.readouterr() == ("", "hello 42\n")  # stdout stays for results


@pytest.mark.parametrize("n, text", [(0, "0 chunks"), (1, "1 chunk"), (2, "2 chunks"), (12345, "12,345 chunks")])
def test_count(n, text):
    assert console.count(n, "chunk") == text


@pytest.mark.parametrize(
    "seconds, text", [(0, "0:00:00"), (59.9, "0:00:59"), (3725, "1:02:05"), (40 * 3600, "40:00:00")]
)
def test_duration(seconds, text):
    assert console.duration(seconds) == text


@pytest.mark.parametrize(
    "platform, path, text",
    [
        ("win32", "C:\\My Notes", 'sift add "C:\\My Notes"'),  # as cmd and PowerShell take it
        ("win32", "C:\\notes", "sift add C:\\notes"),
        ("linux", "/home/me/My Notes", "sift add '/home/me/My Notes'"),
        ("darwin", "/notes", "sift add /notes"),
    ],
)
def test_a_command_to_paste_is_quoted_for_the_shells_there(platform, path, text):
    assert console.command("sift", "add", path, platform=platform) == text


def test_a_line_blanks_out_what_a_longer_one_left_behind(capsys):
    line = console.Line()
    line.show("a longer line")
    line.show("short")
    line.end()
    assert capsys.readouterr().err == "\ra longer line\rshort        \n"


# --- progress ------------------------------------------------------------------------


@pytest.fixture
def progress(clock):
    return lambda total, **kw: console.Progress(total, "chunks", **kw)


def test_progress_shows_a_bar_the_speed_and_the_time_taken_and_left(progress, clock, capsys):
    with progress(3) as bar:
        clock.advance(10)
        bar.update(1)
        clock.advance(20)
        bar.update(3)
    err = capsys.readouterr().err
    assert "[                    ]   0% 0/3 chunks, 0:00:00 elapsed" in err  # at once, before anything is done
    assert "[#######             ]  33% 1/3 chunks, 0.1/s, 0:00:10 elapsed, 0:00:20 left" in err
    assert "[####################] 100% 3/3 chunks, 0.1/s, 0:00:30 elapsed, 0:00:00 left" in err
    assert err.endswith("\n")  # what comes next starts on a line of its own


def test_progress_speeds_of_ten_and_more_are_whole_numbers(progress, clock, capsys):
    with progress(20_000) as bar:
        clock.advance(10)
        bar.update(12_345)
    assert "12,345/20,000 chunks, 1,234/s" in capsys.readouterr().err


def test_progress_survives_a_clock_that_stands_still(progress, capsys):
    # Coarse timers can return the same reading twice; the speed must not divide by zero.
    with progress(3) as bar:
        bar.update(3)
    assert "] 100% 3/3 chunks, 0:00:00 elapsed" in capsys.readouterr().err


def test_progress_carries_a_prefix(clock, capsys):
    with console.Progress(2, "files", prefix="notes: ") as bar:
        bar.update(1)
    assert "\rnotes: [##########          ]  50% 1/2 files" in capsys.readouterr().err


def test_a_note_takes_the_progress_line_and_the_bar_continues_below(progress, capsys):
    with progress(2) as bar:
        bar.update(1)
        bar.note("r.jsonl: 1 line skipped")
        bar.update(2)
    lines = capsys.readouterr().err.split("\n")
    assert lines[0].rsplit("\r", 1)[-1].startswith("r.jsonl: 1 line skipped")  # on its own line
    assert "100% 2/2 chunks" in lines[1]


def test_a_message_said_during_progress_takes_a_line_of_its_own(progress, capsys):
    # A model loading in the middle of `sift index` says so: its words ran on after the progress line's.
    with progress(2) as bar:
        bar.update(1)
        console.say("loading the OCR model", "(2 GB)")
        bar.update(2)
    console.say("done")
    lines = capsys.readouterr().err.split("\n")
    assert lines[0].rsplit("\r", 1)[-1].startswith("loading the OCR model (2 GB)")
    assert "100% 2/2 chunks" in lines[1]
    assert lines[2] == "done"  # once the progress line is done, messages are as ever


def progress_threads():
    return [t for t in threading.enumerate() if t.name == "sift-progress"]


def test_progress_keeps_counting_while_a_step_takes_long(progress, clock, monkeypatch):
    # On a CPU one batch can take minutes; meanwhile the elapsed time must keep moving, so sift visibly works.
    monkeypatch.setattr(console, "REDRAW_SECONDS", 0.01)
    redrawn = threading.Event()

    class Terminal(io.StringIO):
        def write(self, text):
            if "0:01:00 elapsed" in text:
                redrawn.set()
            return super().write(text)

    monkeypatch.setattr(sys, "stderr", Terminal())
    with progress(3):
        clock.advance(60)  # a slow step, with no update
        assert redrawn.wait(timeout=10), "the progress line stood still"
    assert progress_threads() == []


def test_a_long_step_goes_into_the_log_every_minute(progress, clock, monkeypatch):
    # `sift index` ran for half an hour with OCR; its log said nothing between loading the model and the end.
    monkeypatch.setattr(console, "REDRAW_SECONDS", 0.01)
    logged = threading.Event()
    monkeypatch.setattr(console.logs.said, "info", lambda text: "0:01:00 elapsed" in text and logged.set())
    with progress(3) as bar:
        bar.update(1)
        clock.advance(30)
        clock.advance(30)  # a minute in: once into the log
        assert logged.wait(timeout=10), "the log heard nothing of a long step"


def test_a_step_shorter_than_a_minute_goes_into_the_log_once_where_it_ended(progress, clock, monkeypatch):
    monkeypatch.setattr(console, "REDRAW_SECONDS", 0.01)
    logged, redrawn = [], threading.Event()
    monkeypatch.setattr(console.logs.said, "info", logged.append)

    class Terminal(io.StringIO):
        def write(self, text):
            if "0:00:30 elapsed" in text:
                redrawn.set()
            return super().write(text)

    monkeypatch.setattr(sys, "stderr", Terminal())
    with progress(3):
        clock.advance(30)
        assert redrawn.wait(timeout=10), "the progress line stood still"
    assert len(logged) == 1  # where it ended, as ever
    assert "0:00:30 elapsed" in logged[0]


def test_progress_stops_redrawing_when_interrupted(progress):
    with pytest.raises(KeyboardInterrupt), progress(3):
        raise KeyboardInterrupt  # Ctrl+C
    assert progress_threads() == []
    assert gc.isenabled()


def test_garbage_is_freed_on_the_working_thread_while_a_line_redraws(progress, monkeypatch):
    # The collector ran on the redrawing thread and freed PyTorch's tensors there, which crashed it (Python 3.15).
    collected = []
    monkeypatch.setattr(gc, "collect", lambda generation=2: collected.append((generation, threading.current_thread())))
    monkeypatch.setattr(gc, "get_count", lambda: (10**6, 0, 0))  # always due
    with progress(300) as bar, progress(2) as inner:
        assert not gc.isenabled()
        inner.update(1)
        for done in range(1, 101):
            bar.update(done)
        assert not gc.isenabled()  # the inner line ended; the outer one still draws
    assert gc.isenabled()
    assert {thread for _, thread in collected} == {threading.main_thread()}
    generations = [generation for generation, _ in collected]
    assert (generations.count(0), generations.count(1), generations.count(2)) == (91, 9, 1)  # young, older, all


def test_garbage_waits_while_the_collector_is_off(progress, monkeypatch):
    collected = []
    monkeypatch.setattr(gc, "collect", lambda generation=2: collected.append(generation))
    monkeypatch.setattr(gc, "get_count", lambda: (10**6, 0, 0))
    gc.disable()
    try:
        with progress(2) as bar:
            bar.update(1)
        assert not gc.isenabled()  # as it was
    finally:
        gc.enable()
    assert collected == []
