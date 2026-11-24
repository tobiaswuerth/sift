from collections.abc import Sequence
from dataclasses import replace
from enum import Enum

from sift.domain.text.chunks import chunks
from sift.domain.text.markdown import heading
from sift.domain.text.notes import Document, storable, title_field

PAGES = 50
DPI, LONGEST = 200, 1540
POINTS_AN_INCH = 72
LOOP_LINES = 10
LOOP_CHARACTERS, LOOP_PERIOD = 400, 60
PAGES_APART = "\n\n"
OCR_FIELD = "ocr"


class OcrSetting(Enum):
    OFF = "off"
    AUTO = "auto"
    ON = "on"


READ_IN_ORDER = (OcrSetting.OFF, OcrSetting.AUTO, OcrSetting.ON)


def reads_more(now: OcrSetting, before: OcrSetting) -> bool:
    return READ_IN_ORDER.index(now) > READ_IN_ORDER.index(before)


def page_scale(width: float, height: float) -> float:
    return min(DPI / POINTS_AN_INCH, LONGEST / max(width, height))


def _repeated_tail(text: str) -> tuple[int, int] | None:
    tail = text[-LOOP_CHARACTERS:]
    if len(tail) < LOOP_CHARACTERS:
        return None
    period = next((each for each in range(1, LOOP_PERIOD + 1) if tail[each:] == tail[:-each]), None)
    if period is None:
        return None
    start = len(text) - LOOP_CHARACTERS
    while start > 0 and text[start - 1] == text[start - 1 + period]:
        start -= 1
    return start, period


def _last_lines(text: str, count: int) -> list[str]:
    found: list[str] = []
    end = len(text)
    while end >= 0 and len(found) < count:
        start = text.rfind("\n", 0, end) + 1
        if line := text[start:end].strip():
            found.append(line)
        end = start - 1
    return found


def looping(text: str) -> bool:
    last = _last_lines(text, LOOP_LINES)
    return (len(last) == LOOP_LINES and len(set(last)) == 1) or _repeated_tail(text) is not None


def _runs(lines: Sequence[str]) -> list[list[int]]:
    filled = [at for at, line in enumerate(lines) if line.strip()]
    runs: list[list[int]] = []
    for at in filled:
        if runs and lines[runs[-1][0]].strip() == lines[at].strip():
            runs[-1].append(at)
        else:
            runs.append([at])
    return runs


def without_loop(text: str) -> str:
    if found := _repeated_tail(text):
        start, period = found
        text = text[: start + period]
    lines = text.rstrip().split("\n")
    dropped = {at for run in _runs(lines) if len(run) >= LOOP_LINES for at in range(run[0] + 1, run[-1] + 1)}
    return "\n".join(line for at, line in enumerate(lines) if at not in dropped)


def read_scan(scan: Document, pages: Sequence[str], model: str) -> Document | None:
    text = storable(PAGES_APART.join(page.strip() for page in pages if page.strip()))
    if not text:
        return None
    title = title_field(scan.fields) or heading(text) or scan.title
    fields = {**scan.fields, OCR_FIELD: model}
    return replace(scan, title=storable(title), fields=fields, text=text, chunks=chunks(text))


def setting_of(written: str) -> OcrSetting:
    return next((each for each in OcrSetting if each.value == written), OcrSetting.OFF)
