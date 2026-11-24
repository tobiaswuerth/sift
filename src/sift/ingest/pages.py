import io

import pypdfium2

from sift.domain.text.scans import PAGES, page_scale
from sift.ingest.pictures import Pages, Pictured, Unpictured

PICTURED_AS = "PNG"
COLOURS = "RGB"


def _pictured(page: pypdfium2.PdfPage) -> bytes:
    width, height = page.get_size()
    held = io.BytesIO()
    page.render(scale=page_scale(width, height)).to_pil().convert(COLOURS).save(held, PICTURED_AS)
    return held.getvalue()


def _each_page(pdf: pypdfium2.PdfDocument, at: int) -> bytes:
    page = pdf[at]
    try:
        return _pictured(page)
    finally:
        page.close()


def pages(raw: bytes, most: int = PAGES) -> Pictured:
    try:
        pdf = pypdfium2.PdfDocument(raw)
    except pypdfium2.PdfiumError as error:
        return Unpictured(str(error))
    try:
        return Pages(tuple(_each_page(pdf, at) for at in range(min(len(pdf), most))), len(pdf))
    finally:
        pdf.close()
