import logging
import multiprocessing
import os
import shutil
import signal
import socket
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from sift.api.collections import TitleFrom
from sift.domain.text.documents import Refused, stopped
from sift.ingest.readers import ToRead, read_file
from sift.ingest.sandbox import (
    ENDED,
    FileReading,
    Patience,
    ReaderFailedError,
    came_to_no_end,
    out_of_memory,
    reader_processes,
    sandboxed,
    usable_cpus,
)
from tests.later import in_process
from tests.mails import mail_bytes, message, sample
from tests.sandboxed import (
    LARGE,
    EndsWhenUnpickled,
    IdentityEndingWhenUnpickled,
    StuckWhenUnpickled,
    echoed,
    echoed_after,
    misbehaving,
    pid_or_hang,
)

ROOT = Path(__file__).parents[3]
WAITED_SECONDS = 60.0

PATIENT = Patience(base_seconds=120.0, bytes_a_second=1e12)
HASTY = Patience(base_seconds=2.0, bytes_a_second=1e12)


def asked(*paths: Path | str) -> list[ToRead]:
    return [ToRead(Path(path), 10, 1 << 20, None, TitleFrom.HEADING) for path in paths]


def read(*names: str) -> Refused:
    return Refused(names[0], read_again=False)


def test_files_read_in_reader_processes_come_back_each_with_its_place() -> None:
    names = [f"f{n}.pdf" for n in range(12)]
    with sandboxed(echoed, PATIENT, workers=3) as reading:
        found = dict(reading(asked(*names)))
    assert found == {at: read(name) for at, name in enumerate(names)}
    assert multiprocessing.active_children() == []


def test_a_sandbox_asked_for_no_file_starts_no_reader_process() -> None:
    with sandboxed(echoed, PATIENT, workers=3) as reading:
        assert (list(reading([])), multiprocessing.active_children()) == ([], [])


def test_a_reader_that_ends_its_process_costs_its_file_alone_and_its_process_is_replaced() -> None:
    with sandboxed(misbehaving, PATIENT, workers=2) as reading:
        found = dict(reading(asked("a.pdf", "crash.pdf", "b.pdf", "crash.pdf", "c.pdf")))
        again = dict(reading(asked("d.pdf")))
    assert found == {
        0: read("a.pdf"),
        1: stopped(ENDED),
        2: read("b.pdf"),
        3: found[1],
        4: read("c.pdf"),
    }
    assert again == {0: read("d.pdf")}
    assert multiprocessing.active_children() == []


def test_a_reader_that_comes_to_no_end_is_stopped_when_its_time_is_up() -> None:
    with sandboxed(misbehaving, HASTY, workers=1) as reading:
        found = dict(reading(asked("hang.pdf", "a.pdf", "b.pdf")))
    assert found == {0: came_to_no_end(2.0), 1: read("a.pdf"), 2: read("b.pdf")}
    assert came_to_no_end(2.0).problem == "its reader came to no end in 2 seconds"
    assert multiprocessing.active_children() == []


def test_a_reader_is_given_more_time_the_larger_its_file_is() -> None:
    by_size = Patience(base_seconds=1.0, bytes_a_second=10.0)
    with sandboxed(misbehaving, by_size, workers=1) as reading:
        found = dict(reading(asked("hang.pdf")))
    assert found == {0: came_to_no_end(2.0)}
    assert multiprocessing.active_children() == []


def test_a_file_its_reader_has_no_memory_for_is_refused_alone_and_the_reader_reads_on() -> None:
    with sandboxed(misbehaving, PATIENT, workers=1) as reading:
        found = dict(reading(asked("a.pdf", "balloon.pdf", "b.pdf")))
    assert found == {0: read("a.pdf"), 1: out_of_memory(), 2: read("b.pdf")}
    assert multiprocessing.active_children() == []


def test_a_file_whose_reader_raises_is_stopped_alone_its_traceback_logged_and_the_reading_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR), sandboxed(misbehaving, PATIENT, workers=1) as reading:
        found = dict(reading(asked("a.pdf", "raise.pdf", "b.pdf")))
    failed = stopped("sift failed reading it (ValueError: a reader's bug); the log holds the details")
    assert found == {0: read("a.pdf"), 1: failed, 2: read("b.pdf")}
    assert multiprocessing.active_children() == []
    (logged,) = caplog.records
    assert logged.getMessage().startswith("reading raise.pdf raised an error:\nTraceback (most recent call last):")


def test_a_reader_that_raised_with_a_thread_left_running_ends_with_its_file() -> None:
    brief = Patience(base_seconds=120.0, bytes_a_second=1e12, stop_seconds=1.0)
    with sandboxed(misbehaving, brief) as reading:
        ((_, refused),) = list(reading(asked("linger.pdf")))
    assert isinstance(refused, Refused)
    assert (refused.stopped, "its thread left running" in refused.problem) == (True, True)
    assert multiprocessing.active_children() == []


def test_a_process_ended_with_any_exit_code_costs_its_file_alone() -> None:
    with sandboxed(misbehaving, PATIENT, workers=1) as reading:
        found = dict(reading(asked("exit1.pdf", "a.pdf")))
    assert found == {0: stopped(ENDED), 1: read("a.pdf")}


@pytest.mark.skipif(sys.platform == "win32", reason="a Windows pipe takes a message whole")
def test_a_reader_that_ends_its_process_while_it_answers_costs_its_file_at_once() -> None:
    with sandboxed(misbehaving, HASTY, workers=1) as reading:
        found = dict(reading(asked("cut.pdf", "a.pdf")))
    assert found == {0: stopped(ENDED), 1: read("a.pdf")}


def test_an_answer_larger_than_sift_takes_is_refused_as_the_file_s_problem() -> None:
    with sandboxed(misbehaving, PATIENT, workers=1, most_answer_bytes=LARGE // 2) as reading:
        ((at, found),) = list(reading(asked("large.pdf")))
    assert isinstance(found, Refused)
    assert (at, found.problem.endswith("bytes, more than the 5,000 sift takes from one file")) == (0, True)


def _ended(pid: int) -> None:
    (child,) = [each for each in multiprocessing.active_children() if each.pid == pid]
    child.kill()
    child.join(WAITED_SECONDS)


def test_a_reader_process_that_ends_while_it_waits_is_replaced() -> None:
    with sandboxed(pid_or_hang, HASTY, workers=2) as reading:
        answers = reading(asked("hang.pdf", "a.pdf"))
        at, first = next(answers)
        assert isinstance(first, Refused)
        _ended(int(first.problem))
        rest = dict(answers)
        again = dict(reading(asked("b.pdf", "c.pdf")))
    assert (at, rest, sorted(again)) == (1, {0: came_to_no_end(2.0)}, [0, 1])
    assert multiprocessing.active_children() == []


def test_files_given_to_reader_processes_that_ended_meanwhile_are_given_to_others() -> None:
    with sandboxed(echoed, PATIENT, workers=2) as reading:
        first = dict(reading(asked("a.pdf", "b.pdf")))
        for child in multiprocessing.active_children():
            _ended(child.pid or 0)
        again = dict(reading(asked("c.pdf", "d.pdf")))
    assert (first, again) == ({0: read("a.pdf"), 1: read("b.pdf")}, {0: read("c.pdf"), 1: read("d.pdf")})


def test_a_file_whose_reader_processes_end_twice_before_they_begin_it_ends_the_reading() -> None:
    doomed = ToRead(Path("a.pdf"), 10, 1 << 20, IdentityEndingWhenUnpickled(1, 1), TitleFrom.HEADING)
    with (
        pytest.raises(ReaderFailedError, match=r"reader processes ended twice before they began reading a\.pdf"),
        sandboxed(echoed, PATIENT) as reading,
    ):
        _ = list(reading([doomed]))
    assert multiprocessing.active_children() == []


class InterruptError(Exception):
    pass


def interrupting(calls: int) -> Callable[[], float]:
    made: list[int] = []

    def clock() -> float:
        made.append(1)
        if len(made) == calls:
            raise InterruptError
        return 0.0

    return clock


def test_waiting_for_reader_processes_to_start_returns_to_python_every_second() -> None:
    stuck = partial(echoed_after, StuckWhenUnpickled(multiprocessing.get_context("spawn").Event().wait))
    late = Patience(start_seconds=30.0)
    with pytest.raises(InterruptError), sandboxed(stuck, late, clock=interrupting(5)) as reading:
        _ = list(reading(asked("a.pdf")))
    assert multiprocessing.active_children() == []


def _left_unsaid(said: BinaryIO) -> bytes:
    try:
        return said.read()
    except ConnectionResetError:
        return b""


def test_reader_processes_end_with_the_process_that_started_them() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        server.settimeout(WAITED_SECONDS)
        address = cast("tuple[str, int]", server.getsockname())
        started = subprocess.Popen([sys.executable, "-m", "tests.orphaning", str(address[1])], cwd=ROOT)
        connection = server.accept()[0]
        with connection, connection.makefile("rb") as said:
            connection.settimeout(WAITED_SECONDS)
            pid = int(said.readline())
            started.kill()
            _ = started.wait(WAITED_SECONDS)
            try:
                left = _left_unsaid(said)
            except TimeoutError:
                os.kill(pid, signal.SIGTERM)
                raise
    assert left == b""


@pytest.mark.parametrize("cpus, made", [(1, 1), (2, 1), (8, 7), (64, 30), (None, 1)])
def test_reader_processes_are_one_less_than_the_cpus_this_process_may_use(cpus: int | None, made: int) -> None:
    assert reader_processes(30, cpus) == made
    assert 1 <= usable_cpus() <= (os.cpu_count() or 1)


def _left_early(reading: FileReading) -> None:
    for _ in reading(asked("hang.pdf", "a.pdf")):
        raise KeyError


def test_leaving_the_reading_early_stops_its_processes_at_once() -> None:
    with pytest.raises(KeyError), sandboxed(misbehaving, PATIENT, workers=2) as reading:
        _left_early(reading)
    assert multiprocessing.active_children() == []


def test_a_reader_process_that_ends_as_it_starts_ends_the_reading() -> None:
    ending = partial(echoed_after, EndsWhenUnpickled())
    with (
        pytest.raises(ReaderFailedError, match="ended as it started, exit code 3"),
        sandboxed(ending, PATIENT) as reading,
    ):
        _ = list(reading(asked("a.pdf")))
    assert multiprocessing.active_children() == []


def test_a_reader_process_that_does_not_start_in_its_time_ends_the_reading() -> None:
    stuck = partial(echoed_after, StuckWhenUnpickled(multiprocessing.get_context("spawn").Event().wait))
    late = Patience(start_seconds=1.0)
    with pytest.raises(ReaderFailedError, match="did not start in 1 seconds"), sandboxed(stuck, late) as reading:
        _ = list(reading(asked("a.pdf")))
    assert multiprocessing.active_children() == []


def test_a_sandbox_of_no_reader_process_is_refused() -> None:
    with pytest.raises(ValueError, match="one at least"), sandboxed(echoed, PATIENT, workers=0):
        pass


def test_documents_and_mail_read_in_reader_processes_are_what_they_are_read_as_here(tmp_path: Path) -> None:
    for extension in (".pdf", ".docx"):
        _ = shutil.copy(sample(extension), tmp_path / f"pigments{extension}")
    attached = [("contract.pdf", sample(".pdf").read_bytes(), "application/pdf")]
    (tmp_path / "offer.eml").write_bytes(mail_bytes(message(headers={"Subject": "Offer"}, attachments=attached)))
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 the rest is missing")
    files = asked(*sorted(tmp_path.iterdir()), tmp_path / "gone.pdf")
    with in_process() as reading:
        here = dict(reading(files))
    with sandboxed(read_file, PATIENT, workers=2) as reading:
        there = dict(reading(files))
    assert there == here
    assert [type(here[at]).__name__ for at in sorted(here)] == [
        "Refused",
        "FileRead",
        "FileRead",
        "FileRead",
        "Refused",
    ]
