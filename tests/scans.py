import io
from collections.abc import Sequence

from PIL import Image

A4_AT_150_DPI = (1240, 1754)
A4_IN_POINTS = (595, 842)


def _inked(size: tuple[int, int]) -> Image.Image:
    picture = Image.new("RGB", size, "white")
    width, height = size
    picture.paste("black", (width // 8, height // 8, width // 2, height // 8 + max(1, height // 50)))
    return picture


def scan(pages: int = 1, size: tuple[int, int] = A4_AT_150_DPI) -> bytes:
    pictures = [_inked(size) for _ in range(pages)]
    held = io.BytesIO()
    pictures[0].save(held, "PDF", resolution=150, save_all=True, append_images=pictures[1:])
    return held.getvalue()


def blank_pdf(boxes: Sequence[tuple[float, float]] = (A4_IN_POINTS,), missing: int = 0) -> bytes:
    pages = [f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] >>" for width, height in boxes]
    kids = " ".join(f"{at} 0 R" for at in range(3, 3 + len(pages) + missing))
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {len(pages) + missing} >>"]
    held, offsets = "%PDF-1.4\n", list[int]()
    for number, each in enumerate([*objects, *pages], 1):
        offsets.append(len(held))
        held += f"{number} 0 obj\n{each}\nendobj\n"
    table = "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    count = len(offsets) + 1
    trailer = f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{len(held)}\n%%EOF\n"
    return (held + f"xref\n0 {count}\n0000000000 65535 f \n{table}{trailer}").encode()
