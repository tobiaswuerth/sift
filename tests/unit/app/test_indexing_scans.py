import contextlib
import sqlite3
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
import torch

from sift.api.collections import AddCollection
from sift.api.documents import GetDocuments
from sift.api.errors import ModelUnavailableError
from sift.api.events import Event, Noted
from sift.api.indexing import IndexCollections, Indexed
from sift.app.collections import add
from sift.app.documents import documents
from sift.app.filing import Readers
from sift.app.indexing import index
from sift.app.scans import BUSY, FULL, NO_GPU, NO_TEXT_FOUND, NOT_LOADED, OFF, Scanning
from sift.domain.text.documents import Conversion, Refused
from sift.domain.text.scans import OCR_FIELD, PAGES, OcrSetting
from sift.inference.accelerator import CPU, Device
from sift.inference.hub import OCR_MODEL
from sift.inference.ocr import Generating, Ocr, Processing
from sift.ingest.documents import convert
from sift.ingest.pictures import Pictured, ToPicture, Unpictured
from sift.ingest.sandbox import ENDED, Picturing
from sift.storage.catalog import known_collections
from tests.files import write
from tests.homes import unreadable_files
from tests.later import later, pictured_here, reading
from tests.mails import mail_bytes, message
from tests.scans import scan

GPU = Device("cuda", torch.bfloat16)
BUSY_GPU = Device("cpu", torch.float32, busy=True)
GIB_FULL = "the model ran out of memory on the cuda: CUDA out of memory"
UNLOADED = f"the model {OCR_MODEL} could not be loaded: 401"
READ = ("scan", "read page 1\n\nread page 2", OCR_MODEL)


@dataclass(slots=True)
class FakeOcr:
    device: Device = GPU
    full: bool = False
    unloadable: bool = False
    words: Callable[[int], str] = lambda at: f"read page {at + 1}"
    read: list[int] = field(default_factory=list[int])
    loads: int = 0
    stores: list[Path] = field(default_factory=list[Path])

    def ocr_device(self) -> Device:
        return self.device

    def loaded_ocr(self, device: Device) -> Ocr:
        self.loads += 1
        if self.unloadable:
            raise ModelUnavailableError(UNLOADED, "log in")
        return Ocr(cast("Generating", object()), cast("Processing", object()), device)

    def pages_text(self, ocr: Ocr, pictures: Sequence[bytes]) -> list[str]:
        assert ocr.device == self.device
        self.read.append(len(pictures))
        for store in self.stores:
            with contextlib.closing(sqlite3.connect(store, timeout=0)) as other:
                _ = other.execute("BEGIN IMMEDIATE")
                other.rollback()
        if self.full:
            raise ModelUnavailableError(GIB_FULL, "free the GPU's memory")
        return [self.words(at) for at in range(len(pictures))]


@dataclass(slots=True)
class Converting:
    names: list[str] = field(default_factory=list[str])

    def __call__(self, raw: bytes, name: str) -> Conversion:
        self.names.append(name)
        return convert(raw, name)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def scans(tmp_path: Path, home: Path) -> Path:
    root = tmp_path / "scans"
    write(root / "note.md", "# Note\n\nwords\n")
    (root / "scan.pdf").write_bytes(scan(pages=2))
    add(home, AddCollection(str(root)))
    return root


def scanning(ocr: FakeOcr, setting: OcrSetting = OcrSetting.AUTO, pages: int = PAGES) -> Scanning:
    return Scanning(setting, lambda: ocr, pictured_here, pages)


def indexed(
    home: Path, ocr: Scanning, said: list[Event] | None = None, converting: Converting | None = None
) -> Indexed:
    heard: list[Event] = [] if said is None else said
    readers = Readers(clock=later, files=reading(converting=converting or Converting()), scanning=ocr)
    (one,) = index(home, IndexCollections(), heard.append, readers=readers).collections
    assert isinstance(one.outcome, Indexed)
    return one.outcome


def text_of(home: Path, id_: str) -> tuple[str, str, object] | None:
    found = documents(home, GetDocuments((id_,), fields=(OCR_FIELD,))).found
    return (found[0].title, found[0].text, (found[0].fields or {}).get(OCR_FIELD)) if found else None


def notes(said: list[Event]) -> list[str]:
    return [each.text for each in said if isinstance(each, Noted)]


@pytest.mark.usefixtures("scans")
def test_a_scan_is_read_by_ocr_when_indexed_saying_so(home: Path) -> None:
    ocr, said = FakeOcr(), list[Event]()
    assert indexed(home, scanning(ocr), said) == Indexed(files=2, added=2, changed=0, removed=0, unreadable=0)
    assert (text_of(home, "scans/scan.pdf"), ocr.read) == (READ, [2])
    assert "scan.pdf: a scan, read with OCR (2 pages)" in notes(said)


def test_a_scan_is_read_once_though_its_file_is_read_again(home: Path, scans: Path) -> None:
    ocr = FakeOcr()
    indexed(home, scanning(ocr))
    (scans / "scan.pdf").write_bytes((scans / "scan.pdf").read_bytes())
    converting = Converting()
    assert indexed(home, scanning(ocr), converting=converting).unreadable == 0
    assert (converting.names, ocr.read, text_of(home, "scans/scan.pdf")) == (["scan.pdf"], [2], READ)


def test_a_scan_attached_to_a_mail_is_read_too(tmp_path: Path, home: Path) -> None:
    attached = [("delivery.pdf", scan(), "application/pdf")]
    write(tmp_path / "post" / "offer.eml", "")
    (tmp_path / "post" / "offer.eml").write_bytes(mail_bytes(message(attachments=attached)))
    add(home, AddCollection(str(tmp_path / "post")))
    said: list[Event] = []
    indexed(home, scanning(FakeOcr(words=lambda _: "# Lieferschein")), said)
    assert text_of(home, "post/offer.eml#delivery.pdf") == ("Lieferschein", "# Lieferschein", OCR_MODEL)
    assert "offer.eml#delivery.pdf: a scan, read with OCR (1 page)" in notes(said)


def test_a_long_scan_is_read_up_to_its_first_pages_saying_so(home: Path, scans: Path) -> None:
    (scans / "scan.pdf").write_bytes(scan(pages=3))
    ocr, said = FakeOcr(), list[Event]()
    indexed(home, scanning(ocr, pages=2), said)
    assert (ocr.read, "scan.pdf: a scan, read with OCR (its first 2 of 3 pages)" in notes(said)) == ([2], True)


@pytest.mark.usefixtures("scans")
def test_scans_are_read_outside_any_write(home: Path) -> None:
    (one,) = known_collections(home)
    ocr = FakeOcr(stores=[one.store])
    indexed(home, scanning(ocr))
    assert ocr.read == [2]


@pytest.mark.parametrize(
    "ocr, problem, again",
    [
        (scanning(FakeOcr(), OcrSetting.OFF), OFF, False),
        (scanning(FakeOcr(CPU)), NO_GPU, False),
        (scanning(FakeOcr(BUSY_GPU)), BUSY, True),
        (scanning(FakeOcr(full=True)), FULL.format(GIB_FULL), True),
        (scanning(FakeOcr(unloadable=True)), NOT_LOADED.format(UNLOADED), True),
        (scanning(FakeOcr(words=lambda _: " ")), NO_TEXT_FOUND, False),
    ],
)
@pytest.mark.usefixtures("scans")
def test_a_scan_that_cannot_be_read_is_left_out_said_and_read_again_only_if_another_try_may_read_it(
    home: Path, ocr: Scanning, problem: str, *, again: bool
) -> None:
    assert indexed(home, ocr).unreadable == 1
    assert (text_of(home, "scans/scan.pdf"), [(each.path, each.problem) for each in unreadable_files(home)]) == (
        None,
        [("scan.pdf", problem)],
    )
    converting = Converting()
    indexed(home, ocr, converting=converting)
    assert converting.names == ["scan.pdf"] * again


@pytest.mark.usefixtures("scans")
def test_a_scan_the_gpu_had_no_room_for_is_read_by_the_next_index_once_it_has(home: Path) -> None:
    assert indexed(home, scanning(FakeOcr(BUSY_GPU))).unreadable == 1
    assert indexed(home, scanning(FakeOcr())).unreadable == 0
    assert text_of(home, "scans/scan.pdf") == READ


@pytest.mark.parametrize(
    "before, after, device", [(OcrSetting.OFF, OcrSetting.AUTO, GPU), (OcrSetting.AUTO, OcrSetting.ON, CPU)]
)
@pytest.mark.usefixtures("scans")
def test_scans_skipped_under_a_setting_are_read_once_it_reads_more(
    home: Path, before: OcrSetting, after: OcrSetting, device: Device
) -> None:
    skipping = FakeOcr(CPU)
    assert [indexed(home, scanning(skipping, before)).unreadable for _ in range(2)] == [1, 1]
    reading_ocr, converting = FakeOcr(device), Converting()
    assert indexed(home, scanning(reading_ocr, after), converting=converting).unreadable == 0
    assert indexed(home, scanning(reading_ocr, after), converting=converting).unreadable == 0
    assert (skipping.read, reading_ocr.read, converting.names) == ([], [2], ["scan.pdf"])
    assert text_of(home, "scans/scan.pdf") == READ


@pytest.mark.usefixtures("scans")
def test_scans_are_read_on_the_cpu_when_told(home: Path) -> None:
    ocr = FakeOcr(CPU)
    assert indexed(home, scanning(ocr, OcrSetting.ON)).unreadable == 0
    assert ocr.read == [2]


def test_the_model_loads_once_however_many_scans_an_index_reads(home: Path, scans: Path) -> None:
    (scans / "second.pdf").write_bytes(scan())
    ocr = FakeOcr()
    assert indexed(home, scanning(ocr)).unreadable == 0
    assert (ocr.loads, sorted(ocr.read)) == (1, [1, 2])


def _painted(found: Pictured | Refused) -> Callable[[], contextlib.AbstractContextManager[Picturing]]:
    def answered(asked: Sequence[ToPicture]) -> Iterator[tuple[int, Pictured | Refused]]:
        return iter([(0, found)] * len(asked))

    @contextmanager
    def opened() -> Generator[Picturing]:
        yield answered

    return opened


@pytest.mark.parametrize("found", [Unpictured("Failed to load document"), Refused(ENDED, read_again=False)])
@pytest.mark.usefixtures("scans")
def test_a_scan_whose_pages_cannot_be_pictured_is_left_out_saying_why(home: Path, found: Unpictured | Refused) -> None:
    ocr = FakeOcr()
    painting = Scanning(OcrSetting.AUTO, lambda: ocr, _painted(found))
    assert indexed(home, painting).unreadable == 1
    (unread,) = unreadable_files(home)
    assert (unread.problem, ocr.read) == (f"a scan whose pages could not be pictured: {found.problem}", [])
