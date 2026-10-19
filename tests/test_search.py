import pytest

from sift import search, store
from sift.search import Index, _any_word

from .conftest import write


def ids(hits):
    return [h["id"] for h in hits]


# --- keyword search ----------------------------------------------------------


def test_keyword_search(corpus):
    index = Index(corpus[0])
    assert ids(index.search("reinforcement")) == ["notes/rl.md"]
    assert ids(index.search('"brush strokes"')) == ["notes/robots/brush.md"]
    assert ids(index.search("Kubelka-Munk")) == ["notes/robots/brush.md"]  # not valid FTS5, falls back to phrases
    assert ids(index.search("title: shaping")) == ["notes/rl.md"]  # column filter
    assert index.search("nonexistentword") == []


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_returns_nothing(corpus, query):
    assert Index(corpus[0]).search(query) == []


def test_unusable_query_is_a_clear_error(corpus, monkeypatch):
    monkeypatch.setattr(search, "_phrases", lambda q: "(")
    with pytest.raises(ValueError, match="invalid search query"):
        Index(corpus[0]).search("NEAR(")


@pytest.mark.parametrize("k", [0, -1, "3", 1.5])
def test_k_must_be_positive_integer(corpus, k):
    with pytest.raises(ValueError, match="k must be"):
        Index(corpus[0]).search("reward", k=k)


def test_hit_fields(corpus):
    hit = Index(corpus[0]).search("reward")[0]
    assert hit == {
        "id": "notes/rl.md",
        "title": "Reward shaping",
        "lines": "1-3",
        "score": hit["score"],
        "text": "# Reward shaping\n\nReward shaping speeds up reinforcement learning.",
    }
    assert hit["score"] > 0


def test_filters(corpus):
    index = Index(corpus[0])
    assert ids(index.search("brush", filters={"categories": "cs.GR"})) == ["notes/robots/brush.md"]
    assert index.search("brush", filters={"categories": "cs.LG"}) == []
    assert ids(index.search("brush", filters={"year": 2024})) == ["notes/robots/brush.md"]
    assert ids(index.search("reward OR brush", path="robots/*")) == ["notes/robots/brush.md"]
    assert ids(index.search("reward", collection="notes")) == ["notes/rl.md"]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"filters": {"bad key": 1}}, ValueError),
        ({"filters": {"categories": ["cs.RO"]}}, ValueError),
        ({"filters": {"categories": None}}, ValueError),
        ({"collection": "missing"}, LookupError),
    ],
)
def test_bad_filters_are_clear_errors(corpus, kwargs, error):
    with pytest.raises(error):
        Index(corpus[0]).search("brush", **kwargs)


def test_nan_in_front_matter_does_not_break_filters(corpus):
    db, root = corpus
    write(root / "odd.md", "---\nscore: NaN\nlimit: Infinity\n---\n# Odd\nbrush\n")
    store.index(db)
    assert ids(Index(db).search("brush", filters={"categories": "cs.RO"})) == ["notes/robots/brush.md"]
    assert ids(Index(db).search("brush", filters={"score": "NaN"})) == ["notes/odd.md"]


# --- semantic and hybrid search ----------------------------------------------


def test_vsearch_needs_embeddings(corpus, fake_models):
    with pytest.raises(LookupError, match="sift embed"):
        Index(corpus[0]).vsearch("anything")


def test_vsearch(embedded):
    index = Index(embedded[0])
    assert ids(index.vsearch("painting robot brush strokes", k=1)) == ["notes/robots/brush.md"]
    assert ids(index.vsearch("reward shaping", k=1, path="*.md")) == ["notes/rl.md"]
    assert ids(index.vsearch("anything", filters={"categories": "cs.RO"})) == ["notes/robots/brush.md"]
    assert index.vsearch("anything", filters={"categories": "none"}) == []
    assert len(index.vsearch("anything", k=100)) == 3


@pytest.mark.parametrize("fp16, dtype", [(True, "float16"), (False, "float32")])
def test_vectors_stay_fp16_on_gpus(embedded, fake_models, monkeypatch, fp16, dtype):
    import torch

    monkeypatch.setattr(fake_models, "fp16", lambda: fp16)
    index = Index(embedded[0])
    assert ids(index.vsearch("reward shaping", k=1)) == ["notes/rl.md"]
    assert index.mat.dtype == getattr(torch, dtype)


def test_vsearch_reloads_after_index_changes(embedded):
    db, root = embedded
    index = Index(db)
    assert ids(index.vsearch("reward shaping", k=1)) == ["notes/rl.md"]
    (root / "rl.md").unlink()
    store.index(db)  # deletes rl.md and its vector; the cached matrix must not serve it
    assert "notes/rl.md" not in ids(index.vsearch("reward shaping"))


def test_vsearch_after_interrupted_reset_asks_for_embedding(embedded, fake_models, monkeypatch):
    db, _ = embedded
    index = Index(db)
    index.vsearch("reward")  # vectors cached

    def interrupted(texts, query=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_models, "encode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        fake_models.embed_pending(db, reset=True)
    with pytest.raises(LookupError, match="sift embed"):
        index.vsearch("reward")


def test_hits_skip_chunks_removed_meanwhile(corpus):
    assert Index(corpus[0])._hits([(10**9, 1.0)]) == []


def test_query(embedded):
    hits = Index(embedded[0]).query("how does a painting robot plan brush strokes", k=2)
    assert ids(hits)[0] == "notes/robots/brush.md"
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]


def test_query_with_only_stopwords_uses_semantic_candidates(embedded):
    assert len(Index(embedded[0]).query("what is the", k=5)) == 3


def test_query_without_matches(embedded):
    assert Index(embedded[0]).query("brush", filters={"categories": "none"}) == []


def test_any_word_drops_stopwords_and_duplicates():
    assert _any_word("How do robots mix the pigments, robots?") == '"robots" OR "mix" OR "pigments"'
    assert _any_word("the of a") == ""


# --- documents and status ----------------------------------------------------


def test_get(corpus):
    index = Index(corpus[0])
    assert index.get("notes/rl.md").startswith("# Reward shaping")
    for bad in ("notes/../outside.md", "missing/rl.md", "notes", "notes/"):
        with pytest.raises(LookupError, match="unknown document"):
            index.get(bad)


def test_get_unreadable_documents(corpus):
    db, root = corpus
    (root / "rl.md").unlink()
    with pytest.raises(LookupError, match="cannot read"):
        Index(db).get("notes/rl.md")
    with pytest.raises(LookupError, match="cannot read"):
        Index(db).get("notes/robots")  # a folder


def test_status(embedded):
    status = Index(embedded[0]).status()
    assert status["index"].endswith("sift.db")
    assert status["model"]
    assert status["collections"] == [
        {"name": "notes", "path": status["collections"][0]["path"], "docs": 3, "chunks": 3, "embedded": 3}
    ]


def test_vectors_leave_the_gpu_when_idle(embedded, fake_models, monkeypatch):
    # The fakes run on the CPU, so a "GPU" copy is a second CPU tensor here; what matters is when it exists.
    monkeypatch.setattr(fake_models, "gpu", lambda: True)
    index = Index(embedded[0])
    first = ids(index.vsearch("brush strokes"))
    assert index.mat is not None
    assert fake_models.release_idle(now=fake_models._last_use + fake_models.IDLE_SECONDS)
    assert index.mat is None  # only the copy in RAM is left
    assert ids(index.vsearch("brush strokes")) == first
    assert index.mat is not None


def test_vectors_reload_while_on_the_gpu(embedded, fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, "gpu", lambda: True)
    db, root = embedded
    index = Index(db)
    assert "notes/rl.md" in ids(index.vsearch("reward shaping"))
    (root / "rl.md").unlink()
    store.index(db)
    assert "notes/rl.md" not in ids(index.vsearch("reward shaping"))
