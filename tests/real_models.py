import gc
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sift.api.collections import AddCollection
from sift.api.documents import GetDocuments, WantedDocument
from sift.api.embedding import EmbedCollections
from sift.api.events import Event
from sift.api.indexing import IndexCollections, Indexed
from sift.api.links import FindSimilar
from sift.api.search import SearchHybrid, SearchResult, SearchVectors
from sift.app.collections import add
from sift.app.documents import documents
from sift.app.embedding import embed
from sift.app.filing import Readers
from sift.app.indexing import index
from sift.app.query import query
from sift.app.scans import Scanning
from sift.app.similar import similar
from sift.app.vsearch import vsearch
from sift.domain.text.scans import OcrSetting
from sift.inference.held import OnDevice
from sift.inference.hub import EMBED_MODEL, OCR_MODEL, RERANK_MODEL, pinned
from sift.inference.local import IDLE_SECONDS, Local, LocalHost, local

NOTES = {
    "halo.md": "# Dark matter halos\n\nGalaxies sit inside halos of dark matter that hold them together.\n",
    "brush.md": "# Brush strokes\n\nA robot arm plans the strokes of a brush on a canvas.\n",
    "bread.md": "# Bread\n\nFlour, water, salt and a day of waiting make a sourdough loaf.\n",
    "tides.md": "# Tides\n\nThe pull of the moon raises the sea twice a day.\n",
    "pruning.md": "# Pruning\n\nA search skips the ranges whose best score cannot win.\n",
}
SCANNED = "The lighthouse keeper logs every passing ship."
MOST_RESTING_BYTES = 100 << 20


def _said(line: str) -> None:
    sys.stdout.write(f"{line}\n")


def _first(result: SearchResult) -> str:
    return result.hits[0].id if result.hits else "nothing"


def _checked(what: str, *, found: object, wanted: object) -> None:
    _said(f"{'ok  ' if found == wanted else 'FAIL'}  {what}: {found!r}")
    if found != wanted:
        raise AssertionError(f"{what}: found {found!r}, wanted {wanted!r}")


def _scan(path: Path) -> None:
    picture = Image.new("L", (1600, 400), 255)
    ImageDraw.Draw(picture).text((60, 160), SCANNED, fill=0, font=ImageFont.load_default(size=48))
    picture.save(path, "PDF", resolution=150)


def _searched(home: Path, models: Local) -> None:
    halo = vsearch(home, SearchVectors("what holds galaxies together"), models)
    _checked("vsearch finds the note on halos", found=_first(halo), wanted="notes/halo.md")
    brush = query(home, SearchHybrid("how does a robot paint"), models)
    _checked("query finds the note on brushes", found=_first(brush), wanted="notes/brush.md")
    alike = similar(home, FindSimilar("notes/halo.md", k=3))
    _checked(
        "similar leaves out the note itself", found="notes/halo.md" in [hit.id for hit in alike.hits], wanted=False
    )


def _megabytes(count: int) -> str:
    return f"{count / (1 << 20):,.0f} MB"


def _rests_and_returns(model: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    base, now = torch.cuda.memory_allocated(), [0.0]
    _said(f"held    {_megabytes(base)} of GPU memory before the models load")
    models = Local(model, host=LocalHost(clock=lambda: now[0]))
    texts = ["dark matter halos", "brush strokes"]
    before = models.encode(texts, query=False)
    loaded = torch.cuda.memory_allocated()
    now[0] += IDLE_SECONDS + 1
    _checked("the models rest once idle", found=models.release_idle(), wanted=True)
    rested = torch.cuda.memory_allocated()
    _said(f"held    {_megabytes(loaded)} loaded, {_megabytes(rested)} at rest")
    _checked("their GPU memory is let go", found=rested - base < MOST_RESTING_BYTES, wanted=True)
    after = models.encode(texts, query=False)
    _checked("they embed as before", found=bool(np.allclose(before, after, atol=1e-2)), wanted=True)
    state = models.state
    assert isinstance(state, OnDevice)
    assert state.embedder is not None
    held = state.embedder.model
    assert isinstance(held, torch.nn.Module)
    parameter = next(held.parameters())
    on = (parameter.device.type, parameter.dtype)
    _checked("and are back on the GPU in bfloat16", found=on, wanted=("cuda", torch.bfloat16))


def _indexed_and_searched(model: str) -> None:
    with tempfile.TemporaryDirectory() as folder:
        home, notes = Path(folder) / "home", Path(folder) / "notes"
        notes.mkdir()
        for name, text in NOTES.items():
            (notes / name).write_text(text, encoding="utf-8")
        _scan(notes / "scan.pdf")
        add(home, AddCollection(str(notes)))
        readers = Readers(scanning=Scanning(OcrSetting.ON))
        said = list[Event]()
        (done,) = index(home, IndexCollections(), said.append, readers=readers).collections
        _checked("index reads the notes and the scan", found=done.outcome, wanted=Indexed(6, 6, 0, 0, 0))
        _checked(
            "OCR reads the scan",
            found="lighthouse keeper"
            in documents(home, GetDocuments((WantedDocument("notes/scan.pdf"),))).found[0].text,
            wanted=True,
        )
        models = local(model)
        embed(home, EmbedCollections(), said.append, models)
        _said(f"device  {cast('OnDevice', models.state).device.name}")
        _searched(home, models)


def main(model: str = EMBED_MODEL, gpu: Callable[[], bool] = torch.cuda.is_available) -> int:
    _said(f"models  {pinned(model)}, {pinned(RERANK_MODEL)}, {pinned(OCR_MODEL)}")
    _indexed_and_searched(model)
    if gpu():
        _rests_and_returns(model)
    else:
        _said("skip  no CUDA GPU: the rest and return of the models on it is not checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
