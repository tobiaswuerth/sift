from typing import TYPE_CHECKING, Protocol

from sift.ingest.pictured import Pictured, ToPicture
from sift.late import late

PAGES = "sift.ingest.pages"


class Painter(Protocol):
    def pages(self, raw: bytes, most: int) -> Pictured: ...


def picture(asked: ToPicture) -> Pictured:
    return late(PAGES, Painter).pages(asked.raw, asked.most)


if TYPE_CHECKING:
    from sift.ingest import pages as _pages

    _PAGES: Painter = _pages
