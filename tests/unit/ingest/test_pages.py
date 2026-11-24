import io

from PIL import Image

from sift.ingest.pages import pages
from sift.ingest.pictures import Pages, Unpictured
from tests.scans import scan


def sizes(found: Pages | Unpictured) -> tuple[int, list[tuple[int, int]]]:
    assert isinstance(found, Pages)
    return found.count, [Image.open(io.BytesIO(each)).size for each in found.pictures]


def test_a_scan_s_first_pages_are_pictures_as_the_model_was_trained_on() -> None:
    assert sizes(pages(scan(pages=3), most=2)) == (3, [(1089, 1540), (1089, 1540)])
    assert sizes(pages(scan(size=(300, 200)))) == (1, [(400, 267)])


def test_a_pdf_that_cannot_be_opened_is_unpictured_saying_why() -> None:
    assert pages(b"%PDF-1.4 the rest is missing") == Unpictured("Failed to load document (PDFium: Data format error).")
