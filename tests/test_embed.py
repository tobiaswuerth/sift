"""`sift embed`: embedding every chunk without a vector, slab by slab, with progress. The models are fakes."""

import sqlite3
import threading
import time

import numpy as np
import pytest

from sift import embed, indexing, store

from .conftest import FakeSentenceTransformer, fake_vector, write


def vector_count(catalog):
    return catalog.store("notes").execute("SELECT count(*) FROM vectors").fetchone()[0]


def test_embed_pending(corpus, fake_models, capsys):
    catalog, _ = corpus
    embed.embed_pending(catalog)
    assert vector_count(catalog) == 3
    assert store.get_meta(catalog.store("notes"), "model") == fake_models.name
    assert store.get_meta(catalog.store("notes"), "dim") == "64"
    assert "embedded 3 chunks" in capsys.readouterr().err
    embed.embed_pending(catalog)
    assert "nothing to embed" in capsys.readouterr().err


def pending(catalog):
    """Chunks without a vector, per collection that has any."""
    sql = "SELECT count(*) FROM chunks c WHERE NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id)"
    counts = {c.name: catalog.store(c).execute(sql).fetchone()[0] for c in catalog.collections()}
    return {name: n for name, n in counts.items() if n}


@pytest.fixture
def two_collections(corpus, tmp_path):
    """The notes and a second collection, indexed, neither embedded."""
    catalog, _ = corpus
    write(tmp_path / "other" / "a.md", "# Other\n\nSomething else entirely.\n")
    catalog.add(tmp_path / "other", "other")
    indexing.index(catalog)
    return catalog


def test_embed_only_the_collections_named(two_collections, fake_models, capsys):
    # Embedding a few collections first, before a big one that takes days.
    catalog = two_collections
    embed.embed_pending(catalog, collections=["other"])
    assert pending(catalog) == {"notes": 3}
    assert "embedding 1 chunk of other with" in capsys.readouterr().err
    embed.embed_pending(catalog, collections=["other", "notes"])
    assert pending(catalog) == {}


def test_embed_an_unknown_collection_names_those_there_are(two_collections, fake_models):
    with pytest.raises(LookupError, match="no collection named 'nope'; the collections are notes, other"):
        embed.embed_pending(two_collections, collections=["nope"])
    assert pending(two_collections) == {"notes": 3, "other": 1}  # nothing embedded before the error


def watching(models, monkeypatch, then):
    """Make models.encode_batches call then(texts) on each batch before it embeds it."""
    real = models.encode_batches

    def encode_batches(batches, query=False):
        return real((then(texts) or texts for texts in batches), query)

    monkeypatch.setattr(models, "encode_batches", encode_batches)


def test_embed_pending_calibrates_once_before_embedding(corpus, fake_models, monkeypatch):
    # With the longest input the model takes (see test_local): sorting every pending chunk by length to find the
    # longest took minutes on a large index, before the first vector.
    calls, real = [], fake_models.prepare
    monkeypatch.setattr(fake_models, "prepare", lambda: calls.append("prepare") or real())
    watching(fake_models, monkeypatch, lambda texts: calls.append("encode"))
    embed.embed_pending(corpus[0])
    assert calls == ["prepare", "encode"]


class Ones:
    """The least a set of models needs to offer embed_pending: a name, a place, a batch size, lengths, prepare and
    encode_batches."""

    name, address, batch = "ones/v1", "nowhere", 2

    def lengths(self, texts):
        return [len(t) for t in texts]

    def prepare(self):
        pass

    def encode(self, texts, query=False):
        return np.ones((len(texts), 4), np.float16)

    def encode_batches(self, batches, query=False):
        return (self.encode(texts, query) for texts in batches)


class Tokens(Ones):
    """Texts as long as their words, whatever their characters; records each batch."""

    def __init__(self):
        self.batches = []

    def lengths(self, texts):
        return [len(t.split()) for t in texts]

    def encode(self, texts, query=False):
        self.batches.append(texts)
        return np.array([[len(t), 0, 0, 0] for t in texts], np.float16)


def test_a_batch_holds_texts_of_similar_length_to_the_model():
    # A batch is padded to its longest text: grouped by characters, which tell tokens poorly, PubMed abstracts were
    # padded to 1.5 times their tokens, and embedding took 1.5 times as long.
    texts = ["a b c d e f", "longwordsonly", "x y", "averyveryverylongword with two"]
    models = Tokens()
    vectors = embed.encode(models, texts)
    assert models.batches == [["a b c d e f", "averyveryverylongword with two"], ["x y", "longwordsonly"]]
    assert vectors[:, 0].tolist() == [len(t) for t in texts]  # each vector with its text


def test_embed_pending_uses_the_models_it_is_given(corpus, capsys):
    catalog, _ = corpus
    embed.embed_pending(catalog, models=Ones())
    assert vector_count(catalog) == 3
    assert (store.get_meta(catalog.store("notes"), "model"), store.get_meta(catalog.store("notes"), "dim")) == (
        "ones/v1",
        "4",
    )
    assert "embedding 3 chunks with ones/v1 on nowhere" in capsys.readouterr().err


NAN = np.full(64, np.nan, np.float16).tobytes()


def test_vectors_the_model_broke_are_not_stored(corpus, fake_models, monkeypatch, capsys):
    real = fake_models.encode

    def breaking(texts, query=False):
        vectors = real(texts, query).copy()
        vectors[[i for i, t in enumerate(texts) if "Reward" in t]] = np.nan  # an overflow inside the model
        return vectors

    monkeypatch.setattr(fake_models, "encode_batches", lambda batches, query=False: map(breaking, batches))
    embed.embed_pending(corpus[0])
    assert vector_count(corpus[0]) == 2  # the third stays pending, for another try
    assert "1 chunk could not be embedded" in capsys.readouterr().err


def test_broken_vectors_from_older_versions_are_embedded_again(embedded, capsys):
    # sift 0.5 and older embedded in fp16, where EmbeddingGemma overflows now and then into NaN vectors.
    catalog, _ = embedded
    (chunk,) = catalog.store("notes").execute("SELECT chunk FROM vectors LIMIT 1").fetchone()
    catalog.store("notes").execute("UPDATE vectors SET vec = ? WHERE chunk = ?", (NAN, chunk))
    catalog.store("notes").execute("DELETE FROM meta WHERE key = 'vectors_checked'")  # an index from before the check
    catalog.store("notes").commit()
    capsys.readouterr()
    embed.embed_pending(catalog)
    (vec,) = catalog.store("notes").execute("SELECT vec FROM vectors WHERE chunk = ?", (chunk,)).fetchone()
    assert np.isfinite(np.frombuffer(vec, np.float16)).all()
    assert "1 broken vector" in capsys.readouterr().err
    embed.embed_pending(catalog)  # checked once, not on every run
    assert "checking" not in capsys.readouterr().err


def test_intact_vectors_from_older_versions_stay(embedded, capsys):
    catalog, _ = embedded
    catalog.store("notes").execute("DELETE FROM meta WHERE key = 'vectors_checked'")  # an index from before the check
    catalog.store("notes").commit()
    before = dict(catalog.store("notes").execute("SELECT chunk, vec FROM vectors"))
    capsys.readouterr()
    embed.embed_pending(catalog)
    assert dict(catalog.store("notes").execute("SELECT chunk, vec FROM vectors")) == before
    err = capsys.readouterr().err
    assert "checking 3 vectors" in err
    assert "dropped" not in err
    assert "nothing to embed" in err


def test_embed_pending_reports_progress_after_every_batch(corpus, fake_models, monkeypatch, capsys):
    catalog, root = corpus
    for i in range(5):
        write(root / f"extra{i}.md", f"# Extra {i}\n\n" + "word " * (10 + 50 * i) + "\n")
    indexing.index(catalog)
    monkeypatch.setattr(fake_models.accelerator, "batch_size", 1)  # long before the slab ends
    embed.embed_pending(catalog)
    err = capsys.readouterr().err
    assert all(f" {n}/8 chunks" in err for n in range(1, 9))
    # sorting by length for the progress steps must not mix up which vector belongs to which chunk
    rows = (
        catalog.store("notes")
        .execute("""SELECT c.seq, t.title, t.text, v.vec FROM chunks c JOIN chunk_text t ON t.id = c.id
                         JOIN vectors v ON v.chunk = c.id""")
        .fetchall()
    )
    assert len(rows) == 8
    for seq, title, text, vec in rows:
        expected = fake_vector(embed.text_of(seq, title, text)).astype(np.float16)
        assert np.frombuffer(vec, np.float16).tolist() == expected.tolist()


def test_embed_pending_in_slabs_with_title_on_later_chunks(corpus, fake_models, monkeypatch):
    catalog, root = corpus
    write(root / "long.md", "# Long\n" + "word " * 1000 + "\n" * 2 + "more " * 1000 + "\n")
    indexing.index(catalog)
    seen = []
    watching(fake_models, monkeypatch, seen.extend)
    monkeypatch.setattr(embed, "SLAB", 2)
    embed.embed_pending(catalog)
    assert vector_count(catalog) == catalog.store("notes").execute("SELECT count(*) FROM chunks").fetchone()[0]
    later = [t for t in seen if t.startswith("Long\n\n")]
    assert later
    assert all("more" in t or "word" in t for t in later)


def test_embed_pending_survives_concurrent_writer(corpus, fake_models, monkeypatch):
    catalog, _ = corpus
    other = store.connect(catalog.get("notes").store, catalog.get("notes").id)  # another sift's connection

    def racing(texts):  # it embeds the first chunk meanwhile
        other.execute("INSERT INTO vectors(chunk, vec) SELECT min(id), x'00' FROM chunks")
        other.commit()

    watching(fake_models, monkeypatch, racing)
    embed.embed_pending(catalog)
    assert vector_count(catalog) == 3
    other.close()


def test_embed_refuses_mixing_models(embedded, fake_models):
    catalog, _ = embedded
    store.set_meta(catalog.store("notes"), "model", "another/model")
    with pytest.raises(SystemExit, match="--reset"):
        embed.embed_pending(catalog)
    embed.embed_pending(catalog, reset=True)
    assert store.get_meta(catalog.store("notes"), "model") == fake_models.name
    assert vector_count(catalog) == 3


def test_embed_pending_calibrates_first(corpus, make_local, gpu_memory):
    embed.embed_pending(corpus[0], models=make_local("cuda"))
    assert gpu_memory.batches[:3] == [1, 2, 4]  # calibration, then the 3 chunks in one batch
    assert gpu_memory.batches[-1] == 3


def test_embed_pending_skips_chunks_without_text(embedded, capsys):
    # A chunk without a document can only come from an index changed meanwhile; there's nothing to embed. The count of
    # pending chunks at the start reads no document, so it counts the chunk; the slabs, which read the documents, pass
    # it over.
    catalog, _ = embedded
    capsys.readouterr()
    catalog.store("notes").execute(
        "INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(999, 9, 1, 1, 0, 0)"
    )
    embed.embed_pending(catalog)  # raised "max() arg is an empty sequence"
    assert "embedded 0 chunks" in capsys.readouterr().err
    assert vector_count(catalog) == 3


def test_counting_the_pending_chunks_reads_no_document(embedded, monkeypatch):
    # Reading each pending chunk's document to count it took 190 s of a start on 15 million PubMed chunks, against
    # 19 s without.
    catalog, _ = embedded
    statements = []
    catalog.store("notes").set_trace_callback(statements.append)
    try:
        embed.embed_pending(catalog)
    finally:
        catalog.store("notes").set_trace_callback(None)
    counts = [s for s in statements if "count(*)" in s and "vectors" in s]
    assert counts
    assert not any("docs" in s for s in counts)


# --- the store read and written beside the model ----------------------------------
# The GPU waited 12 s in every 75 while the next slab was read and sorted by length, and the vectors written.


def embed_threads():
    return [t for t in threading.enumerate() if t.name.startswith("sift-embed")]


def until(condition, seconds=5):
    """Whether condition() comes true within `seconds`."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_the_store_is_read_and_written_off_the_models_thread(corpus, fake_models, monkeypatch):
    threads, lengths, put_vectors = {}, fake_models.lengths, store.put_vectors

    def seen(name, then):
        def recorded(*args):
            threads.setdefault(name, threading.current_thread())
            return then(*args)

        return recorded

    monkeypatch.setattr(fake_models, "lengths", seen("read", lengths))
    monkeypatch.setattr(store, "put_vectors", seen("write", put_vectors))
    watching(fake_models, monkeypatch, seen("embed", lambda texts: None))
    embed.embed_pending(corpus[0])
    assert threads["embed"] is threading.main_thread()  # where PyTorch must run
    assert threading.main_thread() not in (threads["read"], threads["write"])
    assert vector_count(corpus[0]) == 3
    assert embed_threads() == []


def test_the_next_slab_is_read_while_the_model_embeds(corpus, fake_models, monkeypatch):
    monkeypatch.setattr(embed, "SLAB", 1)  # a slab per chunk: 3
    read, lengths, ahead = [], fake_models.lengths, []
    monkeypatch.setattr(fake_models, "lengths", lambda texts: read.append(texts) or lengths(texts))
    watching(fake_models, monkeypatch, lambda texts: ahead.append(until(lambda: len(read) >= min(len(ahead) + 2, 3))))
    embed.embed_pending(corpus[0])
    assert ahead == [True, True, True]  # the slab after each read before the model had its batch
    assert vector_count(corpus[0]) == 3


def test_an_error_reading_stops_embedding(corpus, fake_models, monkeypatch):
    monkeypatch.setattr(embed, "SLAB", 1)
    lengths, read = fake_models.lengths, []

    def failing(texts):
        read.append(texts)
        if len(read) == 3:
            raise sqlite3.OperationalError("disk I/O error")
        return lengths(texts)

    monkeypatch.setattr(fake_models, "lengths", failing)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        embed.embed_pending(corpus[0])
    assert vector_count(corpus[0]) == 1  # the first slab's; the one on the GPU then is embedded next time
    assert embed_threads() == []


def test_an_error_embedding_stops_the_reading(corpus, fake_models, monkeypatch):
    monkeypatch.setattr(embed, "SLAB", 1)
    monkeypatch.setattr(embed, "AHEAD", 1)
    read, lengths = [], fake_models.lengths
    monkeypatch.setattr(fake_models, "lengths", lambda texts: read.append(texts) or lengths(texts))

    def failing(texts):
        assert until(lambda: len(read) == 3)  # the second slab waits in the queue, the third to get in
        time.sleep(0.3)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    watching(fake_models, monkeypatch, failing)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        embed.embed_pending(corpus[0])
    assert embed_threads() == []


def test_an_error_writing_stops_embedding_and_leaves_none_of_the_slab(corpus, fake_models, monkeypatch):
    # Left in the open transaction, the vectors written before the error went in with the next slab, codes or not.
    put_vectors = store.put_vectors

    def failing(db, chunks, matrix):
        put_vectors(db, chunks[:1], matrix[:1])
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(store, "put_vectors", failing)
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        embed.embed_pending(corpus[0])
    assert vector_count(corpus[0]) == 0
    assert embed_threads() == []


def test_embedding_stopped_keeps_the_slabs_embedded_and_resumes(corpus, fake_models, monkeypatch):
    monkeypatch.setattr(embed, "SLAB", 1)
    batches = []

    def stop(texts):
        batches.append(texts)
        if len(batches) == 3:
            raise KeyboardInterrupt  # Ctrl+C, once

    watching(fake_models, monkeypatch, stop)
    with pytest.raises(KeyboardInterrupt):
        embed.embed_pending(corpus[0])
    assert vector_count(corpus[0]) == 1  # the slab before the one on the GPU then
    assert embed_threads() == []
    embed.embed_pending(corpus[0])
    assert vector_count(corpus[0]) == 3


# --- what the user sees ---------------------------------------------------------


def slow_batches(monkeypatch, clock, seconds, then=None):
    """Make every batch take `seconds` on the fake clock, and run then() while it runs."""
    real = FakeSentenceTransformer.__call__

    def forward(self, features, task=None):
        clock.advance(seconds)
        if then:
            then()
        return real(self, features, task)

    monkeypatch.setattr(FakeSentenceTransformer, "__call__", forward)


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
