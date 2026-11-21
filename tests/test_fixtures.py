"""The tests' own fixtures: what they build once and copy for each test (conftest.Built)."""

from sift import postings
from sift.catalog import Catalog

from .conftest import write


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
