"""The tests' own fixtures: what they build once and copy for each test (conftest.Built), and the home each has."""

import dataclasses
import os
import re
from pathlib import Path

from sift import postings, settings
from sift.catalog import Catalog

from .conftest import HEEDED, own_environment, write


def made_by(made):
    """What a fixture builds: a collection of one note, counted in `made`."""

    def make(root):
        made.append(root)
        write(root / "notes" / "a.md", "# A\n")
        catalog = Catalog(root / "home")
        catalog.add(root / "notes", "notes")
        catalog.close()

    return make


def test_a_fixtures_collection_is_built_once_and_copied(built_once, tmp_path):
    # Every test built its collection anew, 0.1 to 0.3 s each: 173 of the 280 CPU seconds the tests took were their
    # fixtures', and 109 of 206 with the five most used built once.
    made = []
    for copy in ("first", "second"):
        built_once.copy("copied", made_by(made), tmp_path / copy)
    assert len(made) == 1
    for copy in ("first", "second"):  # each a collection of its own: its catalog names its own folders
        catalog = Catalog(tmp_path / copy / "home")
        notes = catalog.get("notes")
        assert (notes.path, notes.store) == (
            str((tmp_path / copy / "notes").resolve()),
            str((tmp_path / copy / "home" / "collections" / "notes.sqlite").resolve()),
        )
        catalog.close()
    assert made[0] not in (tmp_path / "first", tmp_path / "second")


def test_tests_that_patch_sift_otherwise_share_no_build(built_once, tmp_path, monkeypatch, configure):
    # A collection built under small lists, or with other settings, is another: handed to a test that patches
    # nothing, it would test what that test never asked for.
    made = []
    built_once.copy("patched", made_by(made), tmp_path / "plain")
    monkeypatch.setattr(postings, "LONG", 8)
    built_once.copy("patched", made_by(made), tmp_path / "small")
    built_once.copy("patched", made_by(made), tmp_path / "small again")
    assert len(made) == 2
    monkeypatch.setattr(postings, "update", lambda *args, **kwargs: None)  # no plain value: by its name
    built_once.copy("patched", made_by(made), tmp_path / "unfollowed")
    assert len(made) == 3
    configure(ocr="off")
    built_once.copy("patched", made_by(made), tmp_path / "other settings")
    assert len(made) == 4


def test_every_test_has_a_sift_home_of_its_own(tmp_path):
    # Tests stayed out of the user's SIFT_HOME by asking for the `home` fixture: one that forgot, or a process one
    # started, wrote its catalog, logs and keys in the user's.
    assert settings.current().home == tmp_path / "home"
    assert os.environ["SIFT_HOME"] == str(tmp_path / "home")  # for the processes it starts too


def test_every_test_starts_from_sift_s_defaults_whatever_the_environment_holds(tmp_path, monkeypatch):
    # Run from a shell with SIFT_UPDATE_CHECK=off, twenty tests of the update's offer failed: the tests read the
    # settings of whoever ran them. A worker's address or another model there would have had them ask the network.
    heeded = set(re.findall(r'environ\.get\("([A-Z_]+)"\)', Path(settings.__file__).read_text(encoding="utf-8")))
    assert heeded - {"SIFT_HOME", "CUDA_VISIBLE_DEVICES"} == set(HEEDED)  # every one sift reads, but CUDA's own
    for name in HEEDED:
        monkeypatch.setenv(name, "off")  # as a shell may hold them
    own_environment(monkeypatch, tmp_path / "other")
    defaults = settings.Settings.from_env({"SIFT_HOME": str(tmp_path / "other")})
    assert dataclasses.replace(settings.current(), gpus=None) == defaults
    assert not set(HEEDED) & set(os.environ)  # for the processes a test starts too
