"""Reading scans: PDFs without text of their own are read by an OCR model, once each, when indexing; a scan attached
to a mail too. The model is faked here at the level of the transformers classes it is loaded with."""

import json
import sys
import types

import pytest
import torch
from PIL import Image

from sift import indexing, ocr, readers, store
from sift.accelerator import Accelerator
from sift.search import Index

from .conftest import PretendGpu, write
from .test_documents import email_file


def scan(path, pages=1, size=(1240, 1754)):
    """A PDF of `pages` pictures and no text, as a scanner makes it: A4 at 150 dpi by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.new("RGB", size, "white") for _ in range(pages)]
    images[0].save(path, "PDF", resolution=150, save_all=True, append_images=images[1:])
    return path


class FakeProcessor:
    """LightOnOcrProcessor, as far as sift uses it: a chat template that turns a picture into a prompt of tokens, and
    decoding."""

    def __init__(self):
        self.pages = 0

    def apply_chat_template(self, conversation, **options):
        (image,) = [part["image"] for part in conversation[0]["content"]]
        self.pages += 1
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.zeros(1, 1),
            "image_sizes": torch.tensor([[image.height, image.width]]),
        }

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(f"word{int(i)}" for i in ids if int(i))


class FakeModel:
    """LightOnOcrForConditionalGeneration: reads a picture as the words of its width and height."""

    out_of_memory = False

    def __init__(self):
        self.device, self.dtype = "cpu", None

    def to(self, target):
        if isinstance(target, torch.dtype):
            self.dtype = target
        else:
            self.device = str(target)
        return self

    def eval(self):
        return self

    def generate(self, input_ids, pixel_values, image_sizes, max_new_tokens, do_sample):
        assert not do_sample
        assert max_new_tokens == ocr.NEW_TOKENS
        if self.out_of_memory:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.00 GiB")
        ((height, width),) = image_sizes.tolist()
        return torch.tensor([[*input_ids[0].tolist(), width, height, 0]])


@pytest.fixture
def engine(make_local):
    """sift's OCR model, with the fakes inside, on a pretended GPU."""
    model, processor = FakeModel(), FakeProcessor()
    loads = []

    def load(name, dtype):
        loads.append((name, dtype))
        return model.to(dtype) if dtype else model, processor

    reader = ocr.Ocr(PretendGpu("cpu"), load=load)
    reader.loads, reader.model, reader.processor = loads, model, processor
    return reader


@pytest.fixture
def scans(engine, monkeypatch, configure):
    """The scans reader indexing uses, with `engine` inside; SIFT_OCR as by default."""
    configure(ocr="auto")
    reader = ocr.Scans(engine)
    monkeypatch.setattr(ocr, "_scans", lambda: reader)
    return reader


def indexed(db, root, name="docs"):
    store.add_collection(db, root, name)
    indexing.index(db)
    return {path: (title, json.loads(meta), text) for path, title, meta, text in db.execute(
        "SELECT path, title, meta, text FROM docs ORDER BY path")}  # fmt: skip


def test_pages_are_pictures_as_the_model_was_trained_on(tmp_path):
    # 200 dpi, at most 1540 pixels on the longer side: an A4 page is 1089 by 1540.
    images, count = ocr.pages(scan(tmp_path / "a.pdf", pages=3).read_bytes(), limit=2)
    assert (count, [image.size for image in images]) == (3, [(1089, 1540), (1089, 1540)])
    (small,), _ = ocr.pages(scan(tmp_path / "b.pdf", size=(300, 200)).read_bytes())
    assert small.size == (400, 267)  # 2 by 1.33 inches at 200 dpi


def test_a_scan_is_read_by_ocr_when_indexed(tmp_path, home, scans, capsys):
    scan(tmp_path / "docs" / "Lieferschein.pdf", pages=2)
    db = store.connect(home)
    docs = indexed(db, tmp_path / "docs")
    title, meta, text = docs["Lieferschein.pdf"]
    assert text == "word1089 word1540\n\nword1089 word1540"  # each page as the model read it
    assert (title, meta["ocr"]) == ("Lieferschein", ocr.MODEL)
    assert [hit["id"] for hit in Index(db).search("word1089")] == ["docs/Lieferschein.pdf"]
    assert "Lieferschein.pdf: a scan, read with OCR (2 pages" in capsys.readouterr().err
    assert scans.engine.loads == [(ocr.MODEL, None)]  # loaded once, in float32 on this pretended GPU


def test_a_scan_is_read_once(tmp_path, home, scans):
    path = scan(tmp_path / "docs" / "a.pdf")
    db = store.connect(home)
    indexed(db, tmp_path / "docs")
    path.touch()  # its time changes, not its content
    indexing.index(db)
    assert scans.engine.processor.pages == 1


def test_a_scan_attached_to_a_mail_is_read_too(tmp_path, home, scans):
    pdf = scan(tmp_path / "scan.pdf").read_bytes()
    (tmp_path / "docs").mkdir()
    email_file(tmp_path / "docs", Subject="Ausweis", attachments=[("Ausweis.pdf", pdf, "application/pdf")])
    db = store.connect(home)
    docs = indexed(db, tmp_path / "docs")
    assert docs["mail/Ausweis.pdf"][2] == "word1089 word1540"


def test_a_long_scan_is_read_up_to_its_first_pages(tmp_path, home, scans, monkeypatch, capsys):
    monkeypatch.setattr(ocr, "PAGES", 2)
    scan(tmp_path / "docs" / "book.pdf", pages=3)
    docs = indexed(store.connect(home), tmp_path / "docs")
    assert docs["book.pdf"][2].count("word1540") == 2
    assert "book.pdf: a scan, read with OCR (its first 2 of 3 pages" in capsys.readouterr().err


def test_the_model_moves_to_the_gpu_while_reading(engine, tmp_path):
    images, _ = ocr.pages(scan(tmp_path / "a.pdf").read_bytes())
    engine.read(images)
    assert engine.model.device == "cpu"  # a PretendGpu's tensors stay on the CPU, but it went "to the GPU"
    assert engine.accelerator._resident


def test_a_scan_ocr_fails_on_is_skipped_and_said(tmp_path, home, scans, capsys):
    scans.engine.model.out_of_memory = True
    scan(tmp_path / "docs" / "a.pdf")
    assert indexed(store.connect(home), tmp_path / "docs") == {}
    assert "a.pdf: skipped (a scan, and OCR failed: CUDA out of memory" in capsys.readouterr().err


def test_a_scan_ocr_finds_no_text_in_is_skipped(tmp_path, home, scans, capsys, monkeypatch):
    monkeypatch.setattr(scans.engine, "read", lambda images: ["  ", ""])
    scan(tmp_path / "docs" / "blank.pdf", pages=2)
    assert indexed(store.connect(home), tmp_path / "docs") == {}
    assert "blank.pdf: skipped (a scan, and OCR found no text in it)" in capsys.readouterr().err


@pytest.mark.parametrize(
    "setting, gpu, says",
    [
        ("off", True, "SIFT_OCR is off"),
        ("auto", False, "sift reads scans on a GPU, and this machine has none; SIFT_OCR=on reads them on the CPU"),
    ],
)
def test_without_ocr_a_scan_is_skipped_and_said(tmp_path, home, scans, configure, capsys, setting, gpu, says):
    configure(ocr=setting)
    scans.engine.accelerator = PretendGpu("cpu") if gpu else Accelerator("cpu")
    scan(tmp_path / "docs" / "a.pdf")
    assert indexed(store.connect(home), tmp_path / "docs") == {}
    assert f"a.pdf: skipped (no text in it: a scan; {says}" in capsys.readouterr().err
    assert scans.engine.loads == []  # nothing loaded for nothing


def test_on_the_cpu_when_told(tmp_path, home, scans, configure):
    configure(ocr="on")
    scans.engine.accelerator = Accelerator("cpu")
    scan(tmp_path / "docs" / "a.pdf")
    assert indexed(store.connect(home), tmp_path / "docs")["a.pdf"][2] == "word1089 word1540"


def test_an_unknown_setting_says_what_it_takes(tmp_path, home, scans, configure):
    configure(ocr="yes")
    scan(tmp_path / "docs" / "a.pdf")
    with pytest.raises(SystemExit, match=r"SIFT_OCR is 'yes': auto \(on a GPU, the default\), on .* or off"):
        indexed(store.connect(home), tmp_path / "docs")


def test_a_document_without_text_that_is_no_pdf_is_no_scan(tmp_path, monkeypatch):
    # Only PDFs are pictures of pages; an office file without text has none.
    monkeypatch.setattr(readers, "_xberg", lambda raw, name: type("Doc", (), {"content": " "})())
    (tmp_path / "empty.docx").write_bytes(b"PK")
    assert readers.load(tmp_path / "empty.docx") == ([], ["skipped (no text in it)"])


def test_the_model_loads_with_transformers(monkeypatch):
    class Loaded:
        @classmethod
        def from_pretrained(cls, name, dtype=None):
            loaded = cls()
            loaded.name, loaded.dtype, loaded.evaluated = name, dtype, False
            return loaded

        def eval(self):
            self.evaluated = True
            return self

    module = types.ModuleType("transformers")
    module.LightOnOcrForConditionalGeneration = module.LightOnOcrProcessor = Loaded
    module.logging = types.SimpleNamespace(disable_progress_bar=lambda: None)
    monkeypatch.setitem(sys.modules, "transformers", module)
    model, processor = ocr._load(ocr.MODEL, None)
    assert (model.name, model.dtype, model.evaluated) == (ocr.MODEL, torch.float32, True)  # inference, not training
    assert processor.name == ocr.MODEL
    assert ocr._load(ocr.MODEL, torch.bfloat16)[0].dtype == torch.bfloat16


def test_ocr_shares_the_accelerator_of_this_machines_models(fake_models, configure):
    ocr._scans.cache_clear()
    try:
        assert ocr.default() is ocr.default()  # one per process: the model loads once
        assert ocr.default().engine.accelerator is fake_models.accelerator  # one memory budget for all models
    finally:
        ocr._scans.cache_clear()
    configure(remote="https://t.p@workstation:8766")  # the models are another's: OCR has an accelerator of its own
    assert isinstance(ocr.Scans().engine.accelerator, Accelerator)


def test_reindex_reads_a_scan(tmp_path, home, scans):
    root = tmp_path / "docs"
    write(root / "x.md", "# x\n")
    db = store.connect(home)
    store.add_collection(db, root, "docs")
    scan(root / "a.pdf")
    ((entry, docs),) = indexing.reindex(db, ["docs/a.pdf"]).files
    assert (entry["status"], len(docs)) == ("added", 1)
    assert entry["note"].startswith("a.pdf: a scan, read with OCR (1 page")


def test_scans_an_older_sift_skipped_are_read_again_once(tmp_path, home, scans, configure):
    # sift 1.2 read no scans: their files, and mails with a PDF attached, are read again by the next index.
    root = tmp_path / "docs"
    scan(root / "a.pdf")
    email_file(root, Subject="Scan", attachments=[("b.pdf", scan(tmp_path / "b.pdf").read_bytes(), "application/pdf")])
    write(root / "note.md", "# Note\n")
    db = store.connect(home)
    store.add_collection(db, root, "docs")
    configure(ocr="off")
    indexing.index(db)  # as sift 1.2 did: no documents of the scans
    assert sorted(p for (p,) in db.execute("SELECT path FROM docs")) == ["mail.eml", "note.md"]
    db.execute("DELETE FROM meta WHERE key = 'scans_read'")  # as an index of sift 1.2 has it
    db.commit()
    db.close()
    configure(ocr="auto")
    db = store.connect(home)
    assert db.execute("SELECT count(*) FROM sources WHERE mtime IS NULL").fetchone() == (2,)  # the PDF and the mail
    indexing.index(db)
    assert sorted(p for (p,) in db.execute("SELECT path FROM docs")) == ["a.pdf", "mail.eml", "mail/b.pdf", "note.md"]
    db.close()
    assert store.connect(home).execute("SELECT count(*) FROM sources WHERE mtime IS NULL").fetchone() == (0,)


def test_scans_are_marked_to_be_read_again_once(corpus):
    # Two sifts that start at once both see an index of 1.2; the second must find the work done once it has the
    # database, and not have every scan read again a second time.
    db, _ = corpus
    db.execute("UPDATE sources SET mtime = 1")
    db.commit()
    store._read_scans_again(db)  # what the second runs, after the first has finished
    assert db.execute("SELECT count(*) FROM sources WHERE mtime IS NULL").fetchone() == (0,)


def test_an_interrupted_marking_leaves_the_index_as_it_was(tmp_path, home, monkeypatch):
    scan(tmp_path / "docs" / "a.pdf")
    db = store.connect(home)
    store.add_collection(db, tmp_path / "docs", "docs")
    db.execute("INSERT INTO sources(collection, path, mtime, size) VALUES(1, 'a.pdf', 1, 1)")
    db.execute("DELETE FROM meta WHERE key = 'scans_read'")
    db.commit()
    monkeypatch.setattr(store, "set_meta", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        store._read_scans_again(db)
    assert db.execute("SELECT mtime FROM sources").fetchone() == (1,)  # rolled back: tried again on the next start
