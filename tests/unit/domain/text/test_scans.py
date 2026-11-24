from dataclasses import replace
from types import MappingProxyType

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.text.notes import note
from sift.domain.text.scans import (
    LOOP_LINES,
    OCR_FIELD,
    OcrSetting,
    looping,
    page_scale,
    read_scan,
    reads_more,
    without_loop,
)

A4 = (595.2, 841.92)


def test_pages_are_pictures_as_the_model_was_trained_on() -> None:
    width, height = A4
    scale = page_scale(width, height)
    assert (round(width * scale), round(height * scale)) == (1089, 1540)
    assert (round(144 * page_scale(144, 96)), round(96 * page_scale(144, 96))) == (400, 267)


@pytest.mark.parametrize(
    "text, loops",
    [
        ("# Form\n\n" + "Datum:\n" * 9, False),
        ("# Form\n\n" + "Datum:\n" * 10, True),
        ("# Form\n\n" + "Datum:\n\n" * 10, True),
        ("x" * 399, False),
        ("Text. " + "ab" * 200, True),
        ("Text. " + "Lorem ipsum dolor sit amet. " * 15, True),
        ("The real page: no loop at all, as most pages read.\n" * 3, False),
        (" ".join(str(n) for n in range(200)), False),
        ("", False),
    ],
)
def test_a_loop_is_told_from_text(text: str, *, loops: bool) -> None:
    assert looping(text) is loops


def test_a_loop_is_cut_to_one_of_its_lines() -> None:
    text = "# Lieferschein\n\n| Pos | Menge |\n" + "| 1 | 2 |\n" * 12
    assert without_loop(text) == "# Lieferschein\n\n| Pos | Menge |\n| 1 | 2 |"
    assert without_loop("Line\nLine\nOther") == "Line\nLine\nOther"
    assert without_loop("Text. " + "ab" * 300) == "Text. ab"


def _split_looping(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return len(lines) >= LOOP_LINES and len(set(lines[-LOOP_LINES:])) == 1


@given(st.lists(st.sampled_from(["a", "b", "", " a", "a "]), max_size=30).map("\n".join))
def test_the_lines_of_a_loop_are_told_from_the_end_as_from_the_whole_page(text: str) -> None:
    assert looping(text) is _split_looping(text)


def test_the_setting_reads_more_from_off_to_auto_to_on() -> None:
    assert [reads_more(OcrSetting.AUTO, OcrSetting.OFF), reads_more(OcrSetting.ON, OcrSetting.AUTO)] == [True, True]
    assert [reads_more(OcrSetting.AUTO, OcrSetting.ON), reads_more(OcrSetting.AUTO, OcrSetting.AUTO)] == [False] * 2


def test_a_scan_read_takes_its_text_its_title_from_its_property_or_heading_and_names_the_model() -> None:
    scan = replace(note(b"", "scan.pdf"), title="scan", text="", chunks=())
    found = read_scan(scan, ["# Lieferschein\n\nPos 1", "", "  Seite 2 "], "owner/model")
    assert found is not None
    assert (found.title, found.text, found.fields[OCR_FIELD]) == (
        "Lieferschein",
        "# Lieferschein\n\nPos 1\n\nSeite 2",
        "owner/model",
    )
    assert [chunk.start for chunk in found.chunks][:1] == [0]
    titled = replace(scan, fields=MappingProxyType({"title": "Delivery note"}))
    assert (read_scan(titled, ["# Lieferschein"], "m") or scan).title == "Delivery note"
    assert (read_scan(scan, ["plain words"], "m") or titled).title == "scan"
    assert read_scan(scan, ["", "  \n"], "m") is None
