from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from sift.api.errors import ModelUnavailableError
from sift.api.events import Noted, Reporter
from sift.domain.text.documents import FileRead, Refused, Scan
from sift.domain.text.notes import Document
from sift.domain.text.scans import PAGES, OcrSetting, read_scan
from sift.inference.hub import OCR_MODEL
from sift.inference.loading import OcrModule, ocr_module
from sift.ingest.pictured import Pages, ToPicture
from sift.ingest.sandbox import OpenPicturing, Picturing, picturing

if TYPE_CHECKING:
    from sift.inference.ocr import Ocr

OFF = "a scan; SIFT_OCR is off"
NO_GPU = "a scan; sift reads scans on a GPU, and this machine has none; SIFT_OCR=on reads them on the CPU, slowly"
GPU_UNUSED = "a scan; sift reads scans on a GPU: {}; SIFT_OCR=on reads them on the CPU, slowly"
BUSY = "a scan; too little of the GPU's memory is free now; the next index tries again"
NOT_LOADED = "a scan, and its OCR model could not be loaded ({}); the next index tries again"
FULL = "a scan, and OCR ran out of memory ({}); the next index tries again"
UNPICTURED = "a scan whose pages could not be pictured: {}"
NO_TEXT_FOUND = "a scan, and OCR found no text in it"
ON_THE_CPU = "cpu"


@dataclass(frozen=True, slots=True)
class Scanning:
    setting: OcrSetting = OcrSetting.AUTO
    module: Callable[[], OcrModule] = ocr_module
    pictures: OpenPicturing = picturing
    most_pages: int = PAGES
    gpus: str | None = None


SCANNING = Scanning()


class ScanAgain(Enum):
    NEVER = "once the file changes"
    NEXT_INDEX = "at the next index"
    PASSING = "once the GPU has room, within this index too"
    SETTING = "once SIFT_OCR or the GPUs change"


@dataclass(frozen=True, slots=True)
class ScanUnread:
    problem: str
    again: ScanAgain


@dataclass(frozen=True, slots=True)
class _Reader:
    module: OcrModule
    ocr: "Ocr"


def _available(scanning: Scanning) -> _Reader | ScanUnread:
    if scanning.setting is OcrSetting.OFF:
        return ScanUnread(OFF, ScanAgain.SETTING)
    module = scanning.module()
    device = module.ocr_device(scanning.gpus)
    if device.name == ON_THE_CPU and scanning.setting is OcrSetting.AUTO:
        if device.busy:
            return ScanUnread(BUSY, ScanAgain.PASSING)
        reason = NO_GPU if device.notice is None else GPU_UNUSED.format(device.notice)
        return ScanUnread(reason, ScanAgain.SETTING)
    try:
        return _Reader(module, module.loaded_ocr(device))
    except ModelUnavailableError as error:
        return ScanUnread(NOT_LOADED.format(error.message), ScanAgain.NEXT_INDEX)


@dataclass(slots=True)
class ScanModel:
    scanning: Scanning
    held: _Reader | ScanUnread | None = None

    def reader(self) -> _Reader | ScanUnread:
        if self.held is None or (isinstance(self.held, ScanUnread) and self.held.again is ScanAgain.PASSING):
            self.held = _available(self.scanning)
        return self.held

    def loaded(self) -> bool:
        return isinstance(self.held, _Reader)

    def let_go(self) -> None:
        held, self.held = self.held, None
        if isinstance(held, _Reader):
            module, device = held.module, held.ocr.device
            del held
            module.let_go(device)


@dataclass(frozen=True, slots=True)
class ScanRun:
    model: ScanModel
    picturing: Picturing
    report: Reporter


@dataclass(frozen=True, slots=True)
class ScansRead:
    read: FileRead
    model_ready: bool


def _pages_said(found: Pages) -> str:
    read = len(found.pictures)
    return (
        f"{read} page{'' if read == 1 else 's'}" if read == found.count else f"its first {read} of {found.count} pages"
    )


def _named(path: str, key: str) -> str:
    return f"{path}#{key}" if key else path


def _pictured(run: ScanRun, path: str, scan: Scan) -> Pages | ScanUnread:
    asked = ToPicture(Path(path), len(scan.raw), scan.raw, run.model.scanning.most_pages)
    ((_, found),) = list(run.picturing([asked]))
    match found:
        case Pages():
            return found
        case Refused(problem=problem):
            return ScanUnread(UNPICTURED.format(problem), ScanAgain.NEVER)
        case unpictured:
            return ScanUnread(UNPICTURED.format(unpictured.problem), ScanAgain.NEVER)


def _scan_read(run: ScanRun, path: str, scan: Scan) -> Document | ScanUnread:
    reader = run.model.reader()
    if isinstance(reader, ScanUnread):
        return reader
    pictured = _pictured(run, path, scan)
    if isinstance(pictured, ScanUnread):
        return pictured
    try:
        texts = reader.module.pages_text(reader.ocr, pictured.pictures)
    except ModelUnavailableError as error:
        return ScanUnread(FULL.format(error.message), ScanAgain.NEXT_INDEX)
    found = read_scan(scan.document, texts, OCR_MODEL)
    if found is None:
        return ScanUnread(NO_TEXT_FOUND, ScanAgain.NEVER)
    run.report(Noted(f"{_named(path, scan.document.key)}: a scan, read with OCR ({_pages_said(pictured)})"))
    return found


def scans_read(run: ScanRun, path: str, read: FileRead, held: Mapping[str, str]) -> ScansRead:
    documents, problems, again = list(read.documents), list(read.problems), read.read_again
    skipped = tried = False
    for scan in read.scans:
        key = scan.document.key
        if held.get(key) == scan.document.digest:
            documents.append(scan.document)
            continue
        tried = True
        match _scan_read(run, path, scan):
            case ScanUnread() as unread:
                problems.append(f"{key}: {unread.problem}" if key else unread.problem)
                again |= unread.again in {ScanAgain.NEXT_INDEX, ScanAgain.PASSING}
                skipped |= unread.again is ScanAgain.SETTING
            case found:
                documents.append(found)
    under = run.model.scanning.setting if skipped else None
    return ScansRead(FileRead(tuple(documents), tuple(problems), (), again, under), tried and run.model.loaded())
