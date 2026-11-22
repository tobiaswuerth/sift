import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import check_package
from tools.check_package import problems, project_version, tracked_sources

TRACKED = ("src/sift/__init__.py", "src/sift/api/ids.py")
WHEEL = ("sift/__init__.py", "sift/api/ids.py", "sift_rag-2.0.dist-info/METADATA", "sift_rag-2.0.dist-info/RECORD")
SDIST = (
    "sift_rag-2.0/PKG-INFO",
    "sift_rag-2.0/pyproject.toml",
    "sift_rag-2.0/pyproject.toml.orig",
    "sift_rag-2.0/README.md",
    "sift_rag-2.0/LICENSE",
    "sift_rag-2.0/src/sift/__init__.py",
    "sift_rag-2.0/src/sift/api/ids.py",
)


def test_the_version_is_read_from_pyproject() -> None:
    assert project_version('[project]\nversion = "2.0.0a0"\n') == "2.0.0a0"


def test_a_version_that_is_no_string_is_refused() -> None:
    with pytest.raises(TypeError, match="string"):
        project_version("[project]\nversion = 2\n")


def test_a_package_of_the_tracked_sources_that_says_its_version_passes() -> None:
    assert problems(WHEEL, SDIST, TRACKED, "2.0", "sift 2.0") == []


def test_a_wheel_holds_the_tracked_sources_and_nothing_else() -> None:
    wheel = (*WHEEL[1:], "sift/scratch.py", "tests/test_a.py")
    assert problems(wheel, SDIST, TRACKED, "2.0", "sift 2.0") == [
        "the wheel holds sift/scratch.py, which git does not track",
        "the wheel holds tests/test_a.py, which git does not track",
        "the wheel lacks src/sift/__init__.py",
    ]


def test_an_sdist_holds_the_tracked_sources_and_project_files_and_nothing_else() -> None:
    sdist = (*SDIST[:-1], "sift_rag-2.0/research_notes/a.md", "sift_rag-2.0/tests/test_a.py")
    assert problems(WHEEL, sdist, TRACKED, "2.0", "sift 2.0") == [
        "the sdist holds research_notes/a.md, no tracked source or project file",
        "the sdist holds tests/test_a.py, no tracked source or project file",
        "the sdist lacks src/sift/api/ids.py",
    ]


def test_the_installed_command_must_say_the_version() -> None:
    assert problems(WHEEL, SDIST, TRACKED, "2.0", "sift 1.9") == ["the installed wheel says 'sift 1.9', not 'sift 2.0'"]


def built(folder: Path, version: str, extra: str | None = None) -> Path:
    sources = [*tracked_sources(), *([extra] if extra else [])]
    with zipfile.ZipFile(folder / f"sift_rag-{version}-py3-none-any.whl", "w") as wheel:
        wheel.mkdir("sift")
        for name in sources:
            wheel.writestr(name.removeprefix("src/"), "")
        wheel.writestr(f"sift_rag-{version}.dist-info/METADATA", "")
    with tarfile.open(folder / f"sift_rag-{version}.tar.gz", "w:gz") as sdist:
        folder_entry = tarfile.TarInfo(f"sift_rag-{version}/src")
        folder_entry.type = tarfile.DIRTYPE
        sdist.addfile(folder_entry)
        for name in (*sources, "pyproject.toml", "PKG-INFO"):
            sdist.addfile(tarfile.TarInfo(f"sift_rag-{version}/{name}"), io.BytesIO())
    return folder


def test_archives_of_this_repository_s_sources_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    version = project_version(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert check_package.main([str(built(tmp_path, version)), f"sift {version}"]) == 0
    assert capsys.readouterr().out == ""


def test_archives_with_a_stray_file_fail_and_name_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    version = project_version(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert check_package.main([str(built(tmp_path, version, "src/sift/scratch.py")), f"sift {version}"]) == 1
    assert capsys.readouterr().out.split("\n")[:2] == [
        "the wheel holds sift/scratch.py, which git does not track",
        "the sdist holds src/sift/scratch.py, no tracked source or project file",
    ]
