"""sift's settings: four optional environment variables, read once per process."""

from pathlib import Path

from sift import settings
from sift.settings import Settings


def test_the_defaults_need_no_environment():
    assert Settings.from_env({}) == Settings(
        home=Path.home() / ".sift",
        embed_model="google/embeddinggemma-2",
        rerank_model="Alibaba-NLP/gte-reranker-modernbert-base",
        remote=None,
    )


def test_each_variable_overrides_its_default(tmp_path):
    environ = {
        "SIFT_HOME": str(tmp_path),
        "SIFT_EMBED_MODEL": "a/embedder",
        "SIFT_RERANK_MODEL": "b/reranker",
        "SIFT_EMBED_REMOTE": "https://key@worker:8766",
    }
    assert Settings.from_env(environ) == Settings(tmp_path, "a/embedder", "b/reranker", "https://key@worker:8766")


def test_where_scans_are_read():
    assert Settings.from_env({}).ocr == "auto"
    assert Settings.from_env({"SIFT_OCR": " Off "}).ocr == "off"


def test_an_empty_variable_counts_as_unset():
    # SIFT_HOME= used to mean the current folder, and SIFT_EMBED_REMOTE= a worker without an address.
    assert Settings.from_env(dict.fromkeys(("SIFT_HOME", "SIFT_EMBED_REMOTE"), "")) == Settings.from_env({})


def test_the_environment_is_read_once(monkeypatch):
    settings.current.cache_clear()
    try:
        monkeypatch.setenv("SIFT_EMBED_MODEL", "first/model")
        first = settings.current()
        monkeypatch.setenv("SIFT_EMBED_MODEL", "second/model")
        assert settings.current() is first
        assert first.embed_model == "first/model"
    finally:
        settings.current.cache_clear()  # the next test reads its own environment


def test_a_test_can_change_the_settings(configure):
    configure(embed_model="x/y")
    assert settings.current().embed_model == "x/y"
    configure(remote="https://k@w:1")
    assert (settings.current().embed_model, settings.current().remote) == ("x/y", "https://k@w:1")  # changes add up
