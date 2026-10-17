import numpy as np
import pytest

from sift import store

from .conftest import FakeSentenceTransformer, write


def vector_count(db):
    return db.execute("SELECT count(*) FROM vectors").fetchone()[0]


def test_models_load_with_limits(fake_models):
    model = fake_models.embedder()
    assert (model.name, model.device, model.max_seq_length, model.halved) == (
        fake_models.EMBED_MODEL,
        "cpu",
        fake_models.MAX_TOKENS,
        False,
    )
    assert fake_models.reranker().name == fake_models.RERANK_MODEL
    assert fake_models.embedder() is model  # loaded once


def test_models_use_fp16_on_gpu(fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    assert fake_models.embedder().halved
    assert fake_models.reranker().model.halved


def test_encode_returns_normalised_fp16(fake_models):
    vecs = fake_models.encode(["brush strokes", "reward shaping"])
    assert vecs.dtype == np.float16 and vecs.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vecs.astype(np.float32), axis=1), 1, atol=1e-2)
    assert fake_models.encode(["brush"], query=True).shape == (1, 64)


def test_rerank_scores_pairs(fake_models):
    scores = fake_models.rerank("brush strokes", ["brush strokes here", "nothing related"])
    assert scores[0] > scores[1]


def test_out_of_memory_halves_batch_size(fake_models, monkeypatch, capsys):
    monkeypatch.setattr(FakeSentenceTransformer, "max_batch", 20)
    fake_models.encode(["a", "b"])
    assert fake_models._batch_size == 16
    assert "batch size now 16" in capsys.readouterr().err


def test_out_of_memory_at_batch_size_one_raises(fake_models, monkeypatch):
    import torch

    monkeypatch.setattr(FakeSentenceTransformer, "max_batch", 0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        fake_models.encode(["a"])
    assert fake_models._batch_size == 1


def test_embed_pending(corpus, fake_models, capsys):
    db, _ = corpus
    fake_models.embed_pending(db)
    assert vector_count(db) == 3
    assert store.get_meta(db, "model") == fake_models.EMBED_MODEL
    assert store.get_meta(db, "dim") == "64"
    assert "embedded 3 chunks" in capsys.readouterr().err
    fake_models.embed_pending(db)
    assert "nothing to embed" in capsys.readouterr().err


def test_embed_pending_in_slabs_with_title_on_later_chunks(corpus, fake_models, monkeypatch):
    db, root = corpus
    write(root / "long.md", "# Long\n" + "word " * 1000 + "\n" * 2 + "more " * 1000 + "\n")
    store.index(db)
    seen = []
    real_encode = fake_models.encode
    monkeypatch.setattr(fake_models, "encode", lambda texts, query=False: seen.extend(texts) or real_encode(texts))
    monkeypatch.setattr(fake_models, "SLAB", 2)
    fake_models.embed_pending(db)
    assert vector_count(db) == db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    later = [t for t in seen if t.startswith("Long\n\n")]
    assert later and all("more" in t or "word" in t for t in later)


def test_embed_pending_survives_concurrent_writer(corpus, fake_models, monkeypatch):
    db, _ = corpus
    real_encode = fake_models.encode

    def racing_encode(texts, query=False):  # another process embeds the first chunk meanwhile
        vecs = real_encode(texts)
        db.execute("INSERT INTO vectors(chunk, vec) SELECT min(id), x'00' FROM chunks")
        return vecs

    monkeypatch.setattr(fake_models, "encode", racing_encode)
    fake_models.embed_pending(db)
    assert vector_count(db) == 3


def test_embed_refuses_mixing_models(embedded, fake_models):
    db, _ = embedded
    store.set_meta(db, "model", "another/model")
    with pytest.raises(SystemExit, match="--reset"):
        fake_models.embed_pending(db)
    fake_models.embed_pending(db, reset=True)
    assert store.get_meta(db, "model") == fake_models.EMBED_MODEL
    assert vector_count(db) == 3
