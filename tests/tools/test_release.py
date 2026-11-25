import pytest

from tools.release import CHANGELOG as CHANGELOG_FILE
from tools.release import RefusedError, main, notes, pre_release

CHANGELOG = """# Changelog

Lead.

## [Unreleased]

### Added

- Next.

## [2.0.0rc1] - 2026-12-01

### Changed

- Stores are new.

## [1.15.2] - 2026-11-22

- Fixed.

[2.0.0rc1]: https://example.org/compare/v1.15.2...v2.0.0rc1
[1.15.2]: https://example.org/compare/v1.15.1...v1.15.2
"""


def test_the_notes_of_a_version_are_its_dated_section_without_heading_or_links() -> None:
    assert notes(CHANGELOG, "2.0.0rc1") == "### Changed\n\n- Stores are new.\n"
    assert notes(CHANGELOG, "1.15.2") == "- Fixed.\n"


@pytest.mark.parametrize("version", ["2.0.0", "Unreleased", "2.0.0rc", "1.15"])
def test_a_version_without_its_dated_section_is_refused_saying_what_to_add(version: str) -> None:
    with pytest.raises(RefusedError, match=rf"no section '## \[{version}\] - YYYY-MM-DD'"):
        notes(CHANGELOG, version)


@pytest.mark.parametrize(
    "section", ["## [2.0.0] - 2026-12-1\n\n- Undated.\n", "## [2.0.0]\n\n- Undated.\n", "## [2.0.0] - 2026-12-01\n\n"]
)
def test_an_undated_or_empty_section_is_refused(section: str) -> None:
    with pytest.raises(RefusedError, match=r"2\.0\.0"):
        notes(f"# Changelog\n\n{section}## [1.0.0] - 2026-01-01\n\n- One.\n", "2.0.0")


@pytest.mark.parametrize(
    "version, pre",
    [
        ("2.0.0a0", True),
        ("2.0.0b2", True),
        ("2.0.0rc1", True),
        ("2.0.0.dev3", True),
        ("2.0.0rc1.dev1", True),
        ("2.0.0", False),
        ("1.15.2", False),
        ("2.0.0.post1", False),
    ],
)
def test_pre_releases_are_told_from_releases(version: str, *, pre: bool) -> None:
    assert pre_release(version) is pre


def test_the_tool_prints_the_notes_or_refuses_without_printing_any(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["notes", "0.0.0"]) == 1
    refused = capsys.readouterr()
    assert (refused.out, refused.err.startswith("error  CHANGELOG.md has no section")) == ("", True)
    assert (main(["prerelease", "2.0.0rc1"]), capsys.readouterr().out) == (0, "true\n")
    assert main(["notes"]) == 2
    held = notes(CHANGELOG_FILE.read_text(encoding="utf-8"), "1.15.2")
    assert (main(["notes", "1.15.2"]), capsys.readouterr().out) == (0, held)
