"""Reading scans: a PDF without text of its own, as scanners and phones make them, is read from its pages' pictures by
an OCR model, LightOnOCR-2 (one billion parameters, Apache-2.0), on the device the other models use. Readers only mark
such a document (`Doc.scan`): the model runs in the process that writes the index, once per scan new or changed.

SIFT_OCR chooses where: auto (the default) on a GPU, on also on the CPU, where a page takes minutes, off never.
"""

import time
from functools import cache

from . import console, settings

MODEL = "lightonai/LightOnOCR-2-1B"  # https://huggingface.co/lightonai/LightOnOCR-2-1B
SETTINGS = ("auto", "on", "off")
PAGES = 50  # of a scan, read; the rest of a longer one stays unread, and is said
DPI, LONGEST = 200, 1540  # the pictures the model was trained on: 200 dpi, at most 1540 pixels on the longer side
NEW_TOKENS = 4096  # per page: a dense page has about 2,000
LOOP_LINES = 10  # the same line this many times running: the model loops, as greedy decoding does now and then
LOOP_CHARS, LOOP_PERIOD = 400, 60  # or this many characters of a pattern of at most this many repeated, in a line
LOOP_STEP = 16  # tokens between looks for a loop while a page is read
RETRY_PENALTY = 1.1  # the repetition penalty a page that looped is read again with: it got 5 to 10 times as far
# Pages are read one at a time: on an RTX 4090, 26 s a page in 3 GB; 4 at once took 14 s a page but 7 GB, 8 at once
# 12 s and 20 GB, too much of a GPU other programs share.


def setting():
    """SIFT_OCR, checked."""
    value = settings.current().ocr
    if value not in SETTINGS:
        raise SystemExit(
            f"SIFT_OCR is {value!r}: auto (on a GPU, the default), on (also on the CPU, which takes minutes a page) "
            "or off"
        )
    return value


def pages(raw, limit=None):
    """The pictures of a PDF's first `limit` pages (PAGES by default), as the model reads them, and how many pages it
    has."""
    import pypdfium2  # only reading scans needs it

    pdf = pypdfium2.PdfDocument(raw)
    try:
        images = []
        for i in range(min(len(pdf), limit or PAGES)):
            page = pdf[i]
            width, height = page.get_size()  # in points, 72 to the inch
            images.append(page.render(scale=min(DPI / 72, LONGEST / max(width, height))).to_pil().convert("RGB"))
            page.close()
        return images, len(pdf)
    finally:
        pdf.close()


def _load(name, dtype):
    """LightOnOCR-2 and its processor, into RAM, in `dtype` (None: float32)."""
    import torch
    import transformers

    transformers.logging.disable_progress_bar()  # no "Loading weights" bar: it fills the log of `sift serve`
    transformers.logging.set_verbosity_error()  # nor its note that the checkpoint's type, mistral3, isn't lighton_ocr
    model = transformers.LightOnOcrForConditionalGeneration.from_pretrained(name, dtype=dtype or torch.float32)
    return model.eval(), transformers.LightOnOcrProcessor.from_pretrained(name)


class Ocr:
    """The OCR model of this machine: it loads on first use, into RAM in the accelerator's half precision, and moves
    to the GPU while it reads, as the other models do. `load` is how it loads; tests pass a fake."""

    def __init__(self, accelerator, load=_load):
        self.accelerator, self._load, self._loaded = accelerator, load, None

    def on_gpu(self):
        """Whether it reads on a GPU: the accelerator's, unless too little of its memory is free."""
        self.accelerator.budget()
        return self.accelerator.gpu

    def _ready(self):
        dtype = self.accelerator.half_precision
        if self._loaded is None or self._loaded[0] != dtype:
            console.say(f"loading the OCR model {MODEL} (downloaded once, 2 GB)")
            self._loaded = (dtype, *self._load(MODEL, dtype))
        _, model, processor = self._loaded
        self.accelerator.acquire(id(model), lambda: model.to(self.accelerator.device), lambda: model.to("cpu"))
        return model, processor

    def read(self, images):
        """The text of each picture, as Markdown, in order."""
        model, processor = self._ready()
        return [self._page(model, processor, image) for image in images]

    def _page(self, model, processor, image):

        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device, dtype = self.accelerator.device, self.accelerator.half_precision
        inputs = {
            key: value.to(device=device, dtype=dtype) if dtype and value.is_floating_point() else value.to(device)
            for key, value in inputs.items()
        }
        text = self._generate(model, processor, inputs)
        if looping(text):  # read once more, with a penalty that breaks loops, but would bend every page's words
            text = max(
                without_loop(text), without_loop(self._generate(model, processor, inputs, RETRY_PENALTY)), key=len
            )
        return without_loop(text)

    def _generate(self, model, processor, inputs, repetition_penalty=None):
        import torch

        prompt = inputs["input_ids"].shape[1]
        penalty = {"repetition_penalty": repetition_penalty} if repetition_penalty else {}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                stopping_criteria=_guard(processor, prompt),
                **penalty,
            )
        return processor.decode(out[0, prompt:], skip_special_tokens=True)


def _guard(processor, prompt):
    """What stops generate() once a page's text loops, looking every LOOP_STEP tokens: on two pages of the first
    twelve scans read, greedy decoding repeated a line until it ran out of tokens, two minutes a page."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Loop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            new = input_ids.shape[1] - prompt
            stop = new % LOOP_STEP == 0 and looping(processor.decode(input_ids[0, prompt:], skip_special_tokens=True))
            return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([Loop()])


def looping(text):
    """Whether a page's text ends in a loop: the same line LOOP_LINES times running, or a pattern of at most
    LOOP_PERIOD characters repeated for LOOP_CHARS."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= LOOP_LINES and len(set(lines[-LOOP_LINES:])) == 1:
        return True
    return _repeated_tail(text) is not None


def _repeated_tail(text):
    """Where the pattern that repeats for at least LOOP_CHARS at the end of `text` begins, and its length; or None."""
    tail = text[-LOOP_CHARS:]
    if len(tail) < LOOP_CHARS:
        return None
    for period in range(1, LOOP_PERIOD + 1):
        if tail[period:] == tail[:-period]:
            start = len(text) - LOOP_CHARS
            while start > 0 and text[start - 1] == text[start - 1 + period]:
                start -= 1
            return start, period
    return None


def without_loop(text):
    """A page's text with its loops cut to one turn each: a pattern repeated at its end, and a line repeated
    LOOP_LINES times or more running, blank lines between aside."""
    if found := _repeated_tail(text):
        start, period = found
        text = text[: start + period]
    lines = text.rstrip().split("\n")
    filled = [i for i, line in enumerate(lines) if line.strip()]
    dropped, first = set(), 0
    while first < len(filled):
        last = first
        while last + 1 < len(filled) and lines[filled[last + 1]].strip() == lines[filled[first]].strip():
            last += 1
        if last - first + 1 >= LOOP_LINES:
            dropped.update(range(filled[first] + 1, filled[last] + 1))
        first = last + 1
    return "\n".join(line for i, line in enumerate(lines) if i not in dropped)


class Scans:
    """What indexing reads scans with. `engine` is the OCR model (an Ocr, made on first need)."""

    def __init__(self, engine=None):
        self._engine = engine

    @property
    def engine(self):
        if self._engine is None:
            self._engine = Ocr(_accelerator())
        return self._engine

    def unavailable(self):
        """Why scans can't be read here, or None if they can."""
        chosen = setting()
        if chosen == "off":
            return "SIFT_OCR is off"
        if chosen == "auto" and not self.engine.on_gpu():
            return "sift reads scans on a GPU, and this machine has none; SIFT_OCR=on reads them on the CPU, slowly"
        return None

    def read(self, doc):
        """A scan's document with the text the OCR model reads in its pages, and what to tell the user; or None, and
        why it couldn't be read."""
        from . import readers

        if why := self.unavailable():
            return None, f"skipped (no text in it: a scan; {why})"
        start = time.perf_counter()
        try:
            images, count = pages(doc.scan)
            text = "\n\n".join(t.strip() for t in self.engine.read(images) if t.strip())
        except Exception as error:  # a broken PDF, or the GPU: one scan's trouble mustn't stop indexing the others
            return None, f"skipped (a scan, and OCR failed: {error})"
        if not text:
            return None, "skipped (a scan, and OCR found no text in it)"
        read = console.count(count, "page") if len(images) == count else f"its first {len(images)} of {count} pages"
        said = f"a scan, read with OCR ({read} in {console.duration(time.perf_counter() - start)})"
        title = doc.title if doc.meta.get("title") else readers.parse(text, doc.title)[0]  # its property, else heading
        meta = doc.meta | {"ocr": MODEL}
        return doc._replace(title=title, meta=meta, text=text, chunks=list(readers.chunk(text)), scan=None), said


def _accelerator():
    """The accelerator of this machine's models, which OCR shares, so that one memory budget covers all; a new one
    where the models are another machine's."""
    from . import models
    from .accelerator import Accelerator

    found = getattr(models.default(), "accelerator", None)
    return found if found is not None else Accelerator()


def default():
    """The scans reader of this process: one, so that the model loads once."""
    return _scans()


@cache
def _scans():
    return Scans()
