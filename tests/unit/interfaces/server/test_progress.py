import threading
import tracemalloc
from pathlib import Path
from typing import cast

from sift.api.events import Noted
from sift.api.serving import Served
from sift.interfaces.acts.acting import Acting
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.serving import Serve
from sift.settings import Settings
from tests.files import write
from tests.serving import Line, asked, running, streaming
from tests.waits import until, waited

FILES = 2000
LEFT_LINES = 2000
LEFT_LINE_BYTES = 10_000
MOST_HELD_BYTES = 5_000_000


def phases_of(lines: list[Line]) -> list[tuple[str, int, int, int]]:
    phases = list[tuple[str, int, int, int]]()
    phase, last, run, longest = "", 0, 0, 0
    for line in lines:
        data, event = cast("Line", line.get("data")), line.get("event")
        run = run + 1 if event == "Advanced" else 0
        longest = max(longest, run)
        match event:
            case "PhaseStarted":
                phase, longest = cast("str", data["phase"]), 0
            case "Advanced":
                last = cast("int", data["done"])
            case "PhaseDone":
                phases.append((phase, last, cast("int", data["count"]), longest))
            case _:
                pass
    return phases


def test_indexing_many_files_through_the_home_process_sends_progress_at_most_twice_between_other_lines(
    settings: Settings, tmp_path: Path
) -> None:
    for number in range(FILES):
        write(tmp_path / "notes" / f"{number}.md", f"# Note {number}\n")
    with running(settings, how=Serve(clock=lambda: 0.0)) as server:
        asked(server, "add", {"folder": str(tmp_path / "notes")})
        _, indexed = asked(server, "index", {})
    phases = phases_of(indexed)
    shown = [(phase, last, count) for phase, last, count, _ in phases]
    assert shown == [("scan", FILES, FILES), ("read", FILES, FILES)]
    assert [longest <= 2 for *_, longest in phases] == [True, True]


def test_what_a_job_says_once_its_client_is_gone_is_not_held_for_it(settings: Settings) -> None:
    left, held = threading.Event(), list[int]()

    def saying(acting: Acting, _data: object) -> object:
        acting.report(Noted("begun"))
        waited(left, 30)
        before = tracemalloc.get_traced_memory()[0]
        for number in range(LEFT_LINES):
            acting.report(Noted(f"{number} " + "x" * LEFT_LINE_BYTES))
        held.append(tracemalloc.get_traced_memory()[0] - before)
        return Served("said")

    tracemalloc.start()
    try:
        with running(settings, {**HANDLED, "add": Handled(saying, writes=True)}) as server:
            with streaming(server, "add") as response:
                _ = response.readline()
            assert until(lambda: not server.serving.jobs.listed()[0].watched)
            left.set()
            assert until(lambda: not server.serving.jobs.listed())
    finally:
        tracemalloc.stop()
    assert held[0] < MOST_HELD_BYTES
