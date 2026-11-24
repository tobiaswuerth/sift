import multiprocessing
import shutil
from functools import partial
from pathlib import Path

import pytest

from sift.api.collections import TitleFrom
from sift.domain.text.documents import Refused
from sift.ingest.readers import ToRead, read_file
from sift.ingest.sandbox import ENDED, FileReading, Patience, ReaderFailedError, came_to_no_end, sandboxed
from tests.later import in_process
from tests.mails import mail_bytes, message, sample
from tests.sandboxed import EndsWhenUnpickled, StuckWhenUnpickled, echoed, echoed_after, misbehaving

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


def test_a_reader_that_ends_its_process_costs_its_file_alone_and_its_process_is_replaced() -> None:
    with sandboxed(misbehaving, PATIENT, workers=2) as reading:
        found = dict(reading(asked("a.pdf", "crash.pdf", "b.pdf", "crash.pdf", "c.pdf")))
        again = dict(reading(asked("d.pdf")))
    assert found == {
        0: read("a.pdf"),
        1: Refused(ENDED, read_again=False),
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


def test_a_reader_that_raises_ends_the_reading_saying_which_file() -> None:
    with (
        pytest.raises(ReaderFailedError, match=r"raise\.pdf raised an error"),
        sandboxed(misbehaving, PATIENT) as reading,
    ):
        _ = list(reading(asked("a.pdf", "raise.pdf")))
    assert multiprocessing.active_children() == []


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
