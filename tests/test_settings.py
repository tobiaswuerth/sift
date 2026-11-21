"""sift's settings: the optional environment variables in the README's table, read once per process."""

from pathlib import Path

from sift import settings
from sift.settings import Settings

# The default models, each at the commit of its repository on the Hugging Face Hub that sift pins it to
EMBEDDER = "google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e"
RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base@f7481e6055501a30fb19d090657df9ec1f79ab2c"


def test_the_defaults_need_no_environment():
    assert Settings.from_env({}) == Settings(
        home=Path.home() / ".sift", embed_model=EMBEDDER, rerank_model=RERANKER, remote=None
    )


def test_the_default_models_are_pinned_to_a_commit_however_named():
    # They loaded from the latest commit of their repositories: a push of their owners' changed the vectors of new
    # passages beside the old ones, or the code that computes them, unseen. A release of sift moves a pin.
    named = {"SIFT_EMBED_MODEL": EMBEDDER.split("@")[0], "SIFT_RERANK_MODEL": RERANKER.split("@")[0]}
    assert (Settings.from_env(named).embed_model, Settings.from_env(named).rerank_model) == (EMBEDDER, RERANKER)
    assert Settings.from_env({"SIFT_EMBED_MODEL": "org/model"}).embed_model == "org/model"  # another, as named
    at = "google/embeddinggemma-2@main"
    assert Settings.from_env({"SIFT_EMBED_MODEL": at}).embed_model == at  # a revision of one's own


def test_a_model_is_loaded_at_the_revision_it_names_or_at_its_pin():
    assert settings.hub("org/model@v2.0") == ("org/model", "v2.0")
    assert settings.hub("org/model") == ("org/model", None)  # its latest commit
    assert settings.hub(EMBEDDER.split("@")[0]) == tuple(EMBEDDER.split("@"))
    assert settings.hub("D:\\models\\a@b\\model") == ("D:\\models\\a@b\\model", None)  # a folder, no revision
    assert settings.hub("/models/a@b/model") == ("/models/a@b/model", None)
    # A folder whose own name holds an @ was cut there, and the folder before it loaded, or none.
    assert settings.hub("D:\\models\\gemma@v2") == ("D:\\models\\gemma@v2", None)
    assert settings.hub("/models/gemma@v2") == ("/models/gemma@v2", None)
    assert settings.hub("./gemma@v2") == ("./gemma@v2", None)
    assert settings.hub("gemma@v2") == ("gemma", "v2")  # a repository of the Hub without an owner


def test_each_variable_overrides_its_default(tmp_path):
    environ = {
        "SIFT_HOME": str(tmp_path),
        "SIFT_EMBED_MODEL": "a/embedder",
        "SIFT_RERANK_MODEL": "b/reranker",
        "SIFT_EMBED_REMOTE": "https://key@worker:8766",
    }
    assert Settings.from_env(environ) == Settings(tmp_path, "a/embedder", "b/reranker", "https://key@worker:8766")


def test_the_home_is_a_whole_path_however_it_is_written(tmp_path, monkeypatch):
    # An MCP client's config or Windows' setx passes "~/sift" unexpanded: it was a folder "~" wherever a command ran.
    # And a relative home was another for every folder a command ran in, and for the update's process at the root.
    assert Settings.from_env({"SIFT_HOME": "~/sift-data"}).home == Path.home() / "sift-data"
    monkeypatch.chdir(tmp_path)
    assert Settings.from_env({"SIFT_HOME": "data"}).home == tmp_path / "data"


def test_the_gpus_cuda_may_use_are_as_cuda_reads_them():
    # CUDA's own variable: empty, it hides every GPU, so it doesn't count as unset.
    assert Settings.from_env({}).gpus is None
    assert Settings.from_env({"CUDA_VISIBLE_DEVICES": ""}).gpus == ""
    assert Settings.from_env({"CUDA_VISIBLE_DEVICES": "-1"}).gpus == "-1"


def test_where_scans_are_read():
    assert Settings.from_env({}).ocr == "auto"
    assert Settings.from_env({"SIFT_OCR": " Off "}).ocr == "off"


def test_asking_pypi_for_a_newer_sift_is_on_unless_switched_off_or_offline():
    assert Settings.from_env({}).update_check
    assert not Settings.from_env({"SIFT_UPDATE_CHECK": " Off "}).update_check
    assert not Settings.from_env({"SIFT_UPDATE_CHECK": "0"}).update_check
    assert not Settings.from_env({"HF_HUB_OFFLINE": "1"}).update_check
    assert Settings.from_env({"SIFT_UPDATE_CHECK": "on", "HF_HUB_OFFLINE": "0"}).update_check


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
