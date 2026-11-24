import io

from PIL import Image

A4_AT_150_DPI = (1240, 1754)


def scan(pages: int = 1, size: tuple[int, int] = A4_AT_150_DPI) -> bytes:
    pictures = [Image.new("RGB", size, "white") for _ in range(pages)]
    held = io.BytesIO()
    pictures[0].save(held, "PDF", resolution=150, save_all=True, append_images=pictures[1:])
    return held.getvalue()
