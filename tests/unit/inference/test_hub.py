import os
import subprocess
import sys
from pathlib import Path

import pytest

from sift.api.errors import ModelUnavailableError
from sift.inference.hub import EMBED_MODEL, PINNED, hub, pinned

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_a_pinned_model_is_named_and_loaded_at_its_commit() -> None:
    assert pinned(EMBED_MODEL) == f"{EMBED_MODEL}@{PINNED[EMBED_MODEL]}"
    assert hub(EMBED_MODEL) == (EMBED_MODEL, PINNED[EMBED_MODEL])
    assert hub(pinned(EMBED_MODEL)) == (EMBED_MODEL, PINNED[EMBED_MODEL])


def test_another_model_is_loaded_at_the_commit_it_names() -> None:
    named = f"owner/model@{COMMIT}"
    assert (pinned(named), hub(named)) == (named, ("owner/model", COMMIT))


def test_a_folder_is_named_by_where_it_lies_whole_and_how_its_files_stand(tmp_path: Path) -> None:
    folder = tmp_path / "mine@home"
    folder.mkdir()
    (folder / "config.json").write_bytes(b"{}")
    (folder / "weights").mkdir()
    (folder / "weights" / "model.safetensors").write_bytes(b"\x00" * 8)
    named = pinned(str(folder))
    assert named.startswith(f"{folder.resolve()}#")
    asked = "from sift.inference.hub import pinned; print(pinned('mine@home'))"
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    found = subprocess.run(
        [sys.executable, "-c", asked],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert found.stdout.strip() == named
    assert hub(str(folder)) == (str(folder), None)
    (folder / "weights" / "model.safetensors").write_bytes(b"\x01" * 9)
    assert pinned(str(folder)) != named


MODEL_FILES = (
    ".git/index",
    ".cache/huggingface/download/model.lock",
    "README.md",
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "1_Pooling/config.json",
    "sentencepiece.model",
)


@pytest.mark.parametrize(
    "changed, kept",
    [
        (".git/index", True),
        (".cache/huggingface/download/model.lock", True),
        ("README.md", True),
        ("model.safetensors", False),
        ("config.json", False),
        ("tokenizer.json", False),
        ("1_Pooling/config.json", False),
        ("sentencepiece.model", False),
    ],
)
def test_a_folder_s_stamp_follows_the_files_its_model_reads_alone(tmp_path: Path, changed: str, *, kept: bool) -> None:
    folder = tmp_path / "model"
    for each in MODEL_FILES:
        (folder / each).parent.mkdir(parents=True, exist_ok=True)
        (folder / each).write_bytes(b"a")
    before = pinned(str(folder))
    (folder / changed).write_bytes(b"changed")
    assert (pinned(str(folder)) == before) is kept


def test_a_folder_named_with_an_at_is_loaded_from_the_folder(tmp_path: Path) -> None:
    (tmp_path / f"model@{COMMIT[:8]}").mkdir()
    asked = f"from sift.inference.hub import hub; print(hub('model@{COMMIT[:8]}'))"
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    found = subprocess.run(
        [sys.executable, "-c", asked],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert found.stdout.strip() == repr((f"model@{COMMIT[:8]}", None))


@pytest.mark.parametrize("name", ["owner/model", "owner/model@", "C:/models/absent@home"])
def test_a_model_of_the_hub_named_without_a_revision_is_refused(name: str) -> None:
    with pytest.raises(ModelUnavailableError, match="names no commit") as refused:
        pinned(name)
    assert (
        refused.value.remedy
        == f"name a commit of it, e.g. {name.rstrip('@')}@<commit> from its page under Files and versions"
    )


@pytest.mark.parametrize("revision", ["main", "v2.1", "0123abc"])
def test_a_branch_a_tag_or_a_short_commit_is_refused_as_they_move_or_may_mean_several(revision: str) -> None:
    with pytest.raises(ModelUnavailableError, match=f"names {revision}, no whole commit"):
        pinned(f"owner/model@{revision}")
