"""The hook that keeps what is private out of the repository, `scripts/private.py`: its patterns are made up here."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture
def private():
    spec = importlib.util.spec_from_file_location("private", ROOT / "scripts" / "private.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patterns(tmp_path):
    """A maintainer's patterns, outside any repository: an address, and a folder however its path is written."""
    path = tmp_path / "home" / ".private-patterns"
    path.parent.mkdir()
    path.write_text("# what must not be published\n\njane\\.doe@example\\.org\n[\\\\/]vault[\\\\/]\n", encoding="utf-8")
    return path


def test_a_file_holding_what_is_private_is_refused_without_repeating_it(private, patterns, tmp_path, capsys):
    # The check was a grep by hand before each commit, of what whoever ran it remembered to look for.
    note = tmp_path / "note.md"
    note.write_text("# Setup\n\nAsk Jane.Doe@Example.org, or look in D:\\vault\\setup.\n", encoding="utf-8")
    clean = tmp_path / "clean.md"
    clean.write_text("Ask the maintainer.\n", encoding="utf-8")
    assert private.main([str(note), str(clean)], {}, patterns.parent) == 1
    said = capsys.readouterr().out
    assert f"{note}:3: private (pattern of line 3 of {patterns})" in said  # the address, in any case
    assert f"{note}:3: private (pattern of line 4 of {patterns})" in said  # the folder
    assert "clean.md" not in said
    assert "example.org" not in said.lower()
    assert "vault" not in said  # what is private isn't printed again


def test_a_commit_message_is_a_file_like_any_other(private, patterns, tmp_path, capsys):
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("Fix the reader\n\nAs /vault/letters showed.\n", encoding="utf-8")
    assert private.main([str(message)], {}, patterns.parent) == 1
    assert f"{message}:3: private" in capsys.readouterr().out


def test_nothing_private_passes_and_says_nothing(private, patterns, tmp_path, capsys):
    clean = tmp_path / "clean.md"
    clean.write_text("Ask the maintainer.\n", encoding="utf-8")
    assert private.main([str(clean)], {}, patterns.parent) == 0
    assert capsys.readouterr().out == ""


def test_without_patterns_nothing_is_checked(private, tmp_path, capsys):
    # The patterns are private themselves, so they are in no repository: CI and contributors have none, and pass.
    note = tmp_path / "note.md"
    note.write_text("jane.doe@example.org\n", encoding="utf-8")
    assert private.main([str(note)], {}, tmp_path / "home") == 0
    assert capsys.readouterr().out == ""


def test_patterns_named_and_not_there_are_an_error(private, tmp_path):
    # A maintainer who names the file means to be checked: a path mistyped must not pass for nothing private.
    missing = tmp_path / "nowhere" / "patterns"
    with pytest.raises(SystemExit, match=r"PRIVATE_PATTERNS names .*patterns, which isn't there"):
        private.main([], {"PRIVATE_PATTERNS": str(missing)}, tmp_path)


def test_patterns_are_read_from_the_file_the_environment_names(private, patterns, tmp_path):
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("secret-project\n", encoding="utf-8")
    note = tmp_path / "note.md"
    note.write_text("About Secret-Project.\n", encoding="utf-8")
    assert private.main([str(note)], {"PRIVATE_PATTERNS": str(elsewhere)}, patterns.parent) == 1


def test_a_pattern_that_is_no_regular_expression_is_named_by_its_line(private, tmp_path):
    patterns = tmp_path / ".private-patterns"
    patterns.write_text("fine\n(unclosed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"line 2 of .*\.private-patterns is no regular expression"):
        private.main([], {}, tmp_path)


def test_a_file_that_is_no_text_or_gone_is_passed_over(private, patterns, tmp_path):
    picture = tmp_path / "picture.png"
    picture.write_bytes(b"\x89PNG\xff\xfe\x00 not text")
    assert private.main([str(picture), str(tmp_path / "deleted.md")], {}, patterns.parent) == 0


def test_the_hook_reads_the_files_of_a_commit_and_its_message():
    # Other hooks keep to the files: one that fixes whitespace would rewrite the message it was shown.
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "default_install_hook_types: [pre-commit, commit-msg]" in config
    assert "default_stages: [pre-commit]" in config
    hook = config.split("- id: private\n", 1)[1].split("\n\n", 1)[0]
    assert "entry: python scripts/private.py" in hook
    assert "stages: [pre-commit, commit-msg]" in hook
