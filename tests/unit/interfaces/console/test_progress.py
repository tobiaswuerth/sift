from hypothesis import given
from hypothesis import strategies as st

from sift.api.events import Advanced, CollectionStarted, Event, Finished, Noted, Phase, PhaseDone, PhaseStarted
from sift.interfaces.console.progress import Looks, View, running_line, shown
from sift.interfaces.console.style import PLAIN, Style
from sift.interfaces.console.text import width

LOOKS = Looks(PLAIN, "linux")
RUN: tuple[tuple[float, Event], ...] = (
    (0.0, CollectionStarted("notes", "/data/notes")),
    (0.0, PhaseStarted(Phase.SCAN, None)),
    (0.4, PhaseDone(12_431)),
    (0.4, PhaseStarted(Phase.READ, 1_208)),
    (1.0, Advanced(600)),
    (4.3, PhaseDone(1_208)),
    (4.3, Finished(1_102, 106, 14, 3)),
)


def replayed(events: tuple[tuple[float, Event], ...], looks: Looks = LOOKS) -> tuple[View, list[str]]:
    view, lines = View(), list[str]()
    for now, event in events:
        view, said = shown(view, event, now, looks)
        lines.extend(said)
    return view, lines


def test_a_run_leaves_a_line_a_phase_and_one_that_sums_it_up() -> None:
    assert replayed(RUN)[1] == [
        "notes  /data/notes",
        "  scan      12,431 files                 0.4 s",
        "  read       1,208 files       310/s     3.9 s",
        "  done  +1,102  ~106  -14  \N{MIDDLE DOT}  3 with problems  \N{MIDDLE DOT}  4.3 s",
        "        sift status notes --unreadable",
    ]


def test_the_running_phase_is_one_line_with_a_bar_its_speed_and_the_time_left() -> None:
    view, _ = replayed(RUN[:5])
    bar = "\N{BOX DRAWINGS HEAVY HORIZONTAL}" * 8 + "\N{BOX DRAWINGS LIGHT HORIZONTAL}" * 8
    assert running_line(view, 1.0, 100, PLAIN) == f"  read         600 files   {bar}  50%  1,000/s  0.6 s left"
    assert running_line(replayed(RUN[:2])[0], 0.25, 100, PLAIN) == "  scan" + " " * 35 + "0.2 s"
    assert running_line(replayed(RUN[:3])[0], 0.5, 100, PLAIN) is None


def test_more_done_than_counted_at_the_start_fills_the_bar_and_leaves_no_time() -> None:
    view, _ = replayed((*RUN[:4], (1.0, Advanced(1_500))))
    bar = "\N{BOX DRAWINGS HEAVY HORIZONTAL}" * 16
    assert running_line(view, 1.0, 100, PLAIN) == f"  read       1,500 files   {bar} 100%  2,500/s"


def test_a_note_while_a_phase_runs_is_said_and_the_phase_still_ends_in_its_row() -> None:
    noted = (2.0, Noted("the store is damaged", "remove the collection"))
    view, lines = replayed((*RUN[:5], noted, (4.3, PhaseDone(1_208))))
    assert view.running is None
    assert lines[-3:-1] == ["  note  the store is damaged", "  fix   remove the collection"]
    assert lines[-1].startswith("  read       1,208 files")


def test_without_unicode_the_glyphs_are_ascii() -> None:
    ascii_only = Style(unicode=False)
    view, lines = replayed(RUN, Looks(ascii_only, "linux"))
    assert " | 3 with problems | " in lines[3].replace("  ", " ")
    running, _ = replayed(RUN[:5], Looks(ascii_only, "linux"))
    assert "########--------  50%" in (running_line(running, 1.0, 100, ascii_only) or "")
    assert view.running is None


def test_with_colour_labels_and_secondary_parts_are_painted() -> None:
    lines = replayed(RUN, Looks(Style(colour=True), "linux"))[1]
    assert lines[0] == "\x1b[1mnotes\x1b[0m  \x1b[2m/data/notes\x1b[0m"
    assert lines[3].startswith("  \x1b[32mdone\x1b[0m  ")
    assert "\x1b[33m3 with problems\x1b[0m" in lines[3]


def test_a_note_says_what_happened_and_how_to_fix_it() -> None:
    noted = replayed(((0.0, Noted("robots/ can't be read\n", "connect the drive")),))[1]
    assert noted == ["  note  robots/ can't be read\\n", "  fix   connect the drive"]
    assert replayed(((0.0, Noted("plain")),))[1] == ["  note  plain"]


def test_collections_after_the_first_are_set_apart_by_a_blank_line() -> None:
    lines = replayed(((0.0, CollectionStarted("a", "/a")), (1.0, CollectionStarted("b", "/b"))))[1]
    assert lines == ["a  /a", "", "b  /b"]


def test_progress_without_a_running_phase_changes_nothing() -> None:
    view, lines = replayed(((0.0, Advanced(3)), (0.0, PhaseDone(3))))
    assert (view, lines) == (View(), [])


def test_a_run_without_changes_or_problems_says_so_in_one_line() -> None:
    lines = replayed(((0.0, CollectionStarted("a", "/a")), (2.0, Finished(0, 0, 0, 0))))[1]
    assert lines[1] == "  done  +0  ~0  -0  \N{MIDDLE DOT}  2.0 s"


def test_an_empty_read_phase_is_full() -> None:
    view, _ = replayed(((0.0, PhaseStarted(Phase.READ, 0)),))
    assert "100%" in (running_line(view, 0.0, 100, PLAIN) or "")


@given(st.integers(0, 10**7), st.integers(1, 10**7), st.floats(0, 10**5), st.integers(1, 200))
def test_a_running_line_never_takes_the_last_column(done: int, total: int, seconds: float, columns: int) -> None:
    view, _ = replayed(((0.0, PhaseStarted(Phase.READ, total)), (seconds, Advanced(min(done, total)))))
    line = running_line(view, seconds, columns, PLAIN) or ""
    assert width(line) <= max(columns - 1, 0)


def test_one_file_is_a_file() -> None:
    lines = replayed(((0.0, PhaseStarted(Phase.READ, 1)), (0.5, PhaseDone(1))))[1]
    assert lines == ["  read           1 file        2.0/s     0.5 s"]
    view, _ = replayed(((0.0, PhaseStarted(Phase.READ, 2)), (0.5, Advanced(1))))
    assert (running_line(view, 0.5, 100, PLAIN) or "").startswith("  read           1 file    ")


def test_a_remake_counts_chunks_and_their_speed() -> None:
    lines = replayed(((0.0, PhaseStarted(Phase.REMAKE, 5_000)), (2.0, PhaseDone(5_000))))[1]
    assert lines == ["  remake     5,000 chunks    2,500/s     2.0 s"]


def test_a_phase_done_quicker_than_the_clock_ticks_shows_no_speed() -> None:
    lines = replayed(((0.0, PhaseStarted(Phase.READ, 2)), (0.0, PhaseDone(2))))[1]
    assert lines == ["  read           2 files                 0.0 s"]


def test_a_running_scan_says_how_many_files_it_has_found_so_far() -> None:
    view, _ = replayed(((0.0, PhaseStarted(Phase.SCAN, None)), (2.0, Advanced(12_431))))
    assert running_line(view, 2.5, 100, PLAIN) == "  scan      12,431 files                 2.5 s"
