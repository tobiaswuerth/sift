from pathlib import Path

import pytest

from tests.tools.repositories import tracked_in
from tools import check_prose
from tools.check_prose import ALLOWED, Prose, allowed_entries, checked, findings, prose
from tools.findings import Finding


def test_code_without_prose_holds_none() -> None:
    assert prose("a.py", "x = 1\n") == ()


def test_a_comment_is_found_with_its_line() -> None:
    assert prose("a.py", "x = 1\ny = 2  # two\n") == (Prose("a.py", 2, "comment", "# two"),)


def test_a_tool_pragma_is_a_comment_like_any_other() -> None:
    (found,) = prose("a.py", "x = f()  # type: ignore[arg-type]\n")
    assert (found.kind, found.text) == ("comment", "# type: ignore[arg-type]")


@pytest.mark.parametrize(
    "source",
    ['"""module."""\n', 'def f() -> None:\n    """does."""\n', 'class C:\n    "about"\n', 'x = 1\n"stray"\n'],
)
def test_a_string_standing_as_a_statement_is_found(source: str) -> None:
    (found,) = prose("a.py", source)
    assert found.kind == "docstring"


def test_a_docstring_is_named_by_its_first_line() -> None:
    (found,) = prose("a.py", '"""\n  first\n  second\n"""\n')
    assert found.text == "first"


def test_strings_in_expressions_are_code() -> None:
    assert prose("a.py", 'x = "# not a comment"\nprint(f"{x}")\n') == ()


def test_prose_that_is_not_allowed_is_a_finding() -> None:
    written = [Prose("a.py", 3, "comment", "# why")]
    assert findings(written, frozenset()) == (Finding("a.py", 3, "comment: # why"),)


def test_allowed_prose_passes_only_in_the_file_it_was_allowed_in() -> None:
    allowed = frozenset({("a.py", "# why")})
    assert findings([Prose("a.py", 3, "comment", "# why")], allowed) == ()
    assert len(findings([Prose("a.py", 3, "comment", "# why"), Prose("b.py", 1, "comment", "# why")], allowed)) == 1


def test_an_allowed_entry_whose_prose_is_gone_is_a_finding() -> None:
    stale = findings([], frozenset({("a.py", "# why")}))
    assert stale == (Finding(ALLOWED, 1, "allows what a.py no longer holds: # why"),)


def test_allowed_entries_are_read_from_toml() -> None:
    text = '[[allowed]]\npath = "a.py"\ntext = "# why: x"\nreason = "SQLite quirk"\n'
    assert allowed_entries(text) == frozenset({("a.py", "# why: x")})
    assert allowed_entries("") == frozenset()


def test_an_allowed_entry_without_a_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="path, text and reason"):
        allowed_entries('[[allowed]]\npath = "a.py"\ntext = "# x"\n')


def test_allowed_entries_must_be_tables() -> None:
    with pytest.raises(TypeError, match=r"\[\[allowed\]\]"):
        allowed_entries('allowed = "a.py"\n')


def test_findings_are_printed_with_the_remedy_and_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "a.py").write_text("x = 1  # say\n", encoding="utf-8")
    assert checked(["."], frozenset(), tracked_in(tmp_path, "a.py")) == 1
    output = capsys.readouterr().out
    assert "a.py:1: comment: # say" in output
    assert "1 found. Say it with a name" in output


def test_a_file_without_prose_passes_in_silence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert checked(["a.py"], frozenset(), tracked_in(tmp_path, "a.py")) == 0
    assert capsys.readouterr().out == ""


def test_the_repository_holds_no_prose() -> None:
    assert check_prose.main([]) == 0


@pytest.mark.parametrize("source", ['f"why {x}"\n', 'b"why"\n', '("why", "because")\n', '_ = "why"\n'])
def test_prose_in_any_other_form_of_statement_is_found(source: str) -> None:
    (found,) = prose("a.py", source)
    assert found.kind in {"prose", "docstring"}


@pytest.mark.parametrize("source", ["class P:\n    ...\n", "(...,)\n", "()\n", "_ = 1\n"])
def test_an_ellipsis_an_empty_tuple_and_values_are_no_prose(source: str) -> None:
    assert prose("a.py", source) == ()
