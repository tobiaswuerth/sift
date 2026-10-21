"""`sift embed`: embedding every chunk without a vector, slab by slab, with progress. The models are fakes."""

import threading

import numpy as np
import pytest

from sift import embed, indexing, store

from .conftest import FakeSentenceTransformer, fake_vector, write


def vector_count(db):
    return db.execute("SELECT count(*) FROM vectors").fetchone()[0]


def test_embed_pending(corpus, fake_models, capsys):
    db, _ = corpus
    embed.embed_pending(db)
    assert vector_count(db) == 3
    assert store.get_meta(db, "model") == fake_models.name
    assert store.get_meta(db, "dim") == "64"
    assert "embedded 3 chunks" in capsys.readouterr().err
    embed.embed_pending(db)
    assert "nothing to embed" in capsys.readouterr().err


class Ones:
    """The least a set of models needs to offer embed_pending: a name, a place, a batch size and vectors."""

    name, address, batch = "ones/v1", "nowhere", 2

    def prepare(self, longest):
        pass

    def encode(self, texts, query=False):
        return np.ones((len(texts), 4), np.float16)


def test_embed_pending_uses_the_models_it_is_given(corpus, capsys):
    db, _ = corpus
    embed.embed_pending(db, models=Ones())
    assert vector_count(db) == 3
    assert (store.get_meta(db, "model"), store.get_meta(db, "dim")) == ("ones/v1", "4")
    assert "embedding 3 chunks with ones/v1 on nowhere" in capsys.readouterr().err


NAN = np.full(64, np.nan, np.float16).tobytes()


def test_vectors_the_model_broke_are_not_stored(corpus, fake_models, monkeypatch, capsys):
    real = fake_models.encode

    def breaking(texts, query=False):
        vectors = real(texts, query).copy()
        vectors[[i for i, t in enumerate(texts) if "Reward" in t]] = np.nan  # an overflow inside the model
        return vectors

    monkeypatch.setattr(fake_models, "encode", breaking)
    embed.embed_pending(corpus[0])
    assert vector_count(corpus[0]) == 2  # the third stays pending, for another try
    assert "1 chunk could not be embedded" in capsys.readouterr().err


def test_broken_vectors_from_older_versions_are_embedded_again(embedded, capsys):
    # sift 0.5 and older embedded in fp16, where EmbeddingGemma overflows now and then into NaN vectors.
    db, _ = embedded
    (chunk,) = db.execute("SELECT chunk FROM vectors LIMIT 1").fetchone()
    db.execute("UPDATE vectors SET vec = ? WHERE chunk = ?", (NAN, chunk))
    db.execute("DELETE FROM meta WHERE key = 'vectors_checked'")  # an index from before the check
    db.commit()
    capsys.readouterr()
    embed.embed_pending(db)
    (vec,) = db.execute("SELECT vec FROM vectors WHERE chunk = ?", (chunk,)).fetchone()
    assert np.isfinite(np.frombuffer(vec, np.float16)).all()
    assert "1 broken vector" in capsys.readouterr().err
    embed.embed_pending(db)  # checked once, not on every run
    assert "checking" not in capsys.readouterr().err


def test_intact_vectors_from_older_versions_stay(embedded, capsys):
    db, _ = embedded
    db.execute("DELETE FROM meta WHERE key = 'vectors_checked'")  # an index from before the check
    db.commit()
    before = dict(db.execute("SELECT chunk, vec FROM vectors"))
    capsys.readouterr()
    embed.embed_pending(db)
    assert dict(db.execute("SELECT chunk, vec FROM vectors")) == before
    err = capsys.readouterr().err
    assert "checking 3 vectors" in err
    assert "dropped" not in err
    assert "nothing to embed" in err


def test_embed_pending_reports_progress_after_every_batch(corpus, fake_models, monkeypatch, capsys):
    db, root = corpus
    for i in range(5):
        write(root / f"extra{i}.md", f"# Extra {i}\n\n" + "word " * (10 + 50 * i) + "\n")
    indexing.index(db)
    monkeypatch.setattr(fake_models.accelerator, "batch_size", 1)  # long before the slab ends
    embed.embed_pending(db)
    err = capsys.readouterr().err
    assert all(f" {n}/8 chunks" in err for n in range(1, 9))
    # sorting by length for the progress steps must not mix up which vector belongs to which chunk
    rows = db.execute("""SELECT c.seq, t.title, t.text, v.vec FROM chunks c JOIN chunk_text t ON t.id = c.id
                         JOIN vectors v ON v.chunk = c.id""").fetchall()
    assert len(rows) == 8
    for seq, title, text, vec in rows:
        expected = fake_vector(embed.text_of(seq, title, text)).astype(np.float16)
        assert np.frombuffer(vec, np.float16).tolist() == expected.tolist()


def test_embed_pending_in_slabs_with_title_on_later_chunks(corpus, fake_models, monkeypatch):
    db, root = corpus
    write(root / "long.md", "# Long\n" + "word " * 1000 + "\n" * 2 + "more " * 1000 + "\n")
    indexing.index(db)
    seen = []
    real = fake_models.encode
    monkeypatch.setattr(fake_models, "encode", lambda texts, query=False: seen.extend(texts) or real(texts))
    monkeypatch.setattr(embed, "SLAB", 2)
    embed.embed_pending(db)
    assert vector_count(db) == db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    later = [t for t in seen if t.startswith("Long\n\n")]
    assert later
    assert all("more" in t or "word" in t for t in later)


def test_embed_pending_survives_concurrent_writer(corpus, fake_models, monkeypatch):
    db, _ = corpus
    real = fake_models.encode

    def racing(texts, query=False):  # another process embeds the first chunk meanwhile
        vecs = real(texts)
        db.execute("INSERT INTO vectors(chunk, vec) SELECT min(id), x'00' FROM chunks")
        return vecs

    monkeypatch.setattr(fake_models, "encode", racing)
    embed.embed_pending(db)
    assert vector_count(db) == 3


def test_embed_refuses_mixing_models(embedded, fake_models):
    db, _ = embedded
    store.set_meta(db, "model", "another/model")
    with pytest.raises(SystemExit, match="--reset"):
        embed.embed_pending(db)
    embed.embed_pending(db, reset=True)
    assert store.get_meta(db, "model") == fake_models.name
    assert vector_count(db) == 3


def test_embed_pending_calibrates_first(corpus, make_local, gpu_memory):
    embed.embed_pending(corpus[0], models=make_local("cuda"))
    assert gpu_memory.batches[:3] == [1, 2, 4]  # calibration, then the 3 chunks in one batch
    assert gpu_memory.batches[-1] == 3


def test_embed_pending_calibrates_on_the_longest_chunk(corpus, fake_models, monkeypatch):
    db, root = corpus
    write(root / "long.md", "# Long\n" + "word " * 1000 + "\n\n" + "more " * 1000 + "\n")
    indexing.index(db)
    samples, embedded = [], []
    real = fake_models.encode
    monkeypatch.setattr(fake_models, "prepare", lambda longest: samples.append(longest()))
    monkeypatch.setattr(fake_models, "encode", lambda texts, query=False: embedded.extend(texts) or real(texts))
    embed.embed_pending(db)
    assert samples == [max(embedded, key=len)]  # no batch in this run can be longer


def test_embed_pending_skips_chunks_without_text(embedded, capsys):
    # A chunk without a document can only come from an index changed meanwhile; there's nothing to embed.
    db, _ = embedded
    capsys.readouterr()
    db.execute("INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(999, 9, 1, 1, 0, 0)")
    embed.embed_pending(db)  # raised "max() arg is an empty sequence"
    assert "nothing to embed" in capsys.readouterr().err


# --- what the user sees ---------------------------------------------------------


def slow_batches(monkeypatch, clock, seconds, then=None):
    """Make every batch take `seconds` on the fake clock, and run then() while it runs."""
    real = FakeSentenceTransformer.encode_document

    def encode_document(self, texts, *args, **kwargs):
        clock.advance(seconds)
        if then:
            then()
        return real(self, texts, *args, **kwargs)

    monkeypatch.setattr(FakeSentenceTransformer, "encode_document", encode_document)


def test_embed_says_what_it_embeds_before_loading_the_model(corpus, fake_models, monkeypatch, capsys):
    # Loading a model can take a minute: say what's coming first.
    loaded, said = fake_models.embedder, []
    monkeypatch.setattr(fake_models, "embedder", lambda: said.append(capsys.readouterr().err) or loaded())
    embed.embed_pending(corpus[0])
    assert f"embedding 3 chunks with {fake_models.name} on cpu" in said[0]


def test_embed_shows_a_progress_bar_with_time_taken_and_left(corpus, fake_models, clock, monkeypatch, capsys):
    monkeypatch.setattr(fake_models.accelerator, "batch_size", 1)
    slow_batches(monkeypatch, clock, 10)  # as on a CPU
    embed.embed_pending(corpus[0])
    err = capsys.readouterr().err
    assert "[                    ]   0% 0/3 chunks, 0:00:00 elapsed" in err  # before the first batch is done
    assert "[#######             ]  33% 1/3 chunks, 0.1/s, 0:00:10 elapsed, 0:00:20 left" in err
    assert "[####################] 100% 3/3 chunks, 0.1/s, 0:00:30 elapsed" in err
    assert "embedded 3 chunks in 0:00:30" in err


def progress_threads():
    return [t for t in threading.enumerate() if t.name == "sift-progress"]


def test_the_progress_line_stops_when_embedding_fails(corpus, fake_models, clock, monkeypatch):
    def fail():
        raise KeyboardInterrupt  # Ctrl+C

    slow_batches(monkeypatch, clock, 0, then=fail)
    with pytest.raises(KeyboardInterrupt):
        embed.embed_pending(corpus[0])
    assert progress_threads() == []
