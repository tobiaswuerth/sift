from pathlib import Path

import pytest

from tools.check_package import project_version
from tools.findings import read
from tools.release import CHANGELOG as CHANGELOG_FILE
from tools.release import REMEDY, RefusedError, main, notes, pre_release, unpublishable

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


WHEELS = (
    "sift_rag-2.0.0rc1-cp312-abi3-manylinux_2_28_x86_64.whl",
    "sift_rag-2.0.0rc1-cp312-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    "sift_rag-2.0.0rc1-cp312-abi3-macosx_11_0_arm64.whl",
    "sift_rag-2.0.0rc1-cp312-abi3-win_amd64.whl",
)
SDIST = "sift_rag-2.0.0rc1.tar.gz"


def test_a_wheel_for_every_platform_and_the_sources_are_what_a_release_publishes() -> None:
    assert unpublishable([*WHEELS, SDIST], "2.0.0rc1") == []


@pytest.mark.parametrize(
    "files, problem",
    [
        ([*WHEELS], "no sift_rag-2.0.0rc1.tar.gz: the sources are published with the wheels"),
        ([*WHEELS[1:], SDIST], "no wheel for Linux on x86_64"),
        ([*WHEELS[:2], WHEELS[3], SDIST], "no wheel for macOS on arm64"),
        (
            [*WHEELS, WHEELS[0].replace("2_28", "2_17"), SDIST],
            (
                "2 wheels for Linux on x86_64: sift_rag-2.0.0rc1-cp312-abi3-manylinux_2_17_x86_64.whl, "
                "sift_rag-2.0.0rc1-cp312-abi3-manylinux_2_28_x86_64.whl"
            ),
        ),
        (
            [*WHEELS, "sift_rag-2.0.0rc1-cp312-abi3-linux_x86_64.whl", SDIST],
            "sift_rag-2.0.0rc1-cp312-abi3-linux_x86_64.whl is of no platform a release publishes",
        ),
        (
            [*WHEELS, "sift_rag-2.0.0rc1-cp312-abi3-manylinux_2_39_x86_64.whl", SDIST],
            "sift_rag-2.0.0rc1-cp312-abi3-manylinux_2_39_x86_64.whl is of no platform a release publishes",
        ),
        (
            [*WHEELS, "sift_rag-2.0.0rc1-cp312-abi3-macosx_15_0_arm64.whl", SDIST],
            "sift_rag-2.0.0rc1-cp312-abi3-macosx_15_0_arm64.whl is of no platform a release publishes",
        ),
        (
            [*WHEELS[:3], "sift_rag-2.0.0rc1-cp313-cp313-win_amd64.whl", SDIST],
            "sift_rag-2.0.0rc1-cp313-cp313-win_amd64.whl is of no platform a release publishes",
        ),
        ([*WHEELS, SDIST, "sift_rag-2.0.0.tar.gz"], "sift_rag-2.0.0.tar.gz is of no platform a release publishes"),
        ([*WHEELS, SDIST, "notes.md"], "notes.md is of no platform a release publishes"),
    ],
)
def test_what_a_release_lacks_or_must_not_publish_is_said(files: list[str], problem: str) -> None:
    assert problem in unpublishable(files, "2.0.0rc1")


def test_the_tool_says_what_keeps_a_folder_from_being_published(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    version = project_version(read("pyproject.toml"))
    for name in (*WHEELS[1:], SDIST):
        (tmp_path / name.replace("2.0.0rc1", version)).write_bytes(b"")
    assert main(["files", str(tmp_path)]) == 1
    said = capsys.readouterr()
    assert (said.out, said.err) == ("", f"error  no wheel for Linux on x86_64\n{REMEDY}\n")
    (tmp_path / WHEELS[0].replace("2.0.0rc1", version)).write_bytes(b"")
    assert (main(["files", str(tmp_path)]), capsys.readouterr().err) == (0, "")
