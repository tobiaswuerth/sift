import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import check_package
from tools.check_package import problems, project_version, tracked_sources

TRACKED = ("src/sift/__init__.py", "src/sift/api/ids.py", "native/Cargo.toml", "native/src/lib.rs")
WHEEL = (
    "sift/__init__.py",
    "sift/api/ids.py",
    "sift/_native.abi3.so",
    "sift_rag-2.0.dist-info/METADATA",
    "sift_rag-2.0.dist-info/RECORD",
    "sift_rag-2.0.dist-info/licenses/LICENSE",
    "sift_rag-2.0.dist-info/licenses/native/THIRD-PARTY-NOTICES",
)
SDIST = (
    "sift_rag-2.0/PKG-INFO",
    "sift_rag-2.0/pyproject.toml",
    "sift_rag-2.0/pyproject.toml.orig",
    "sift_rag-2.0/README.md",
    "sift_rag-2.0/LICENSE",
    "sift_rag-2.0/native/Cargo.toml",
    "sift_rag-2.0/native/src/lib.rs",
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


def test_a_wheel_without_the_notices_of_the_crates_built_into_it_is_found() -> None:
    lacking = "the wheel lacks the notices of the crates built into it: sift_rag-2.0.dist-info/licenses/native/"
    named = "THIRD-PARTY-NOTICES, which pyproject.toml names under license-files"
    assert problems(WHEEL[:-1], SDIST, TRACKED, "2.0", "sift 2.0") == [lacking + named]


def test_a_wheel_holds_the_tracked_sources_of_the_package_and_nothing_else() -> None:
    wheel = (*WHEEL[1:], "sift/scratch.py", "tests/test_a.py", "native/Cargo.toml")
    assert problems(wheel, SDIST, TRACKED, "2.0", "sift 2.0") == [
        "the wheel holds native/Cargo.toml, no tracked source of the package",
        "the wheel holds sift/scratch.py, no tracked source of the package",
        "the wheel holds tests/test_a.py, no tracked source of the package",
        "the wheel lacks src/sift/__init__.py",
    ]


@pytest.mark.parametrize("extension", ["sift/_native.abi3.so", "sift/_native.pyd"])
def test_a_wheel_holds_the_extension_as_its_platform_names_it(extension: str) -> None:
    wheel = tuple(name for name in WHEEL if "_native" not in name)
    assert problems((*wheel, extension), SDIST, TRACKED, "2.0", "sift 2.0") == []
    assert problems(wheel, SDIST, TRACKED, "2.0", "sift 2.0") == ["the wheel lacks the extension sift/_native"]
    assert problems((*wheel, "sift/_native.pyd", "sift/_native.abi3.so"), SDIST, TRACKED, "2.0", "sift 2.0") == [
        "the wheel holds the extension twice: sift/_native.abi3.so, sift/_native.pyd"
    ]


def test_an_sdist_holds_the_tracked_sources_the_crate_and_project_files_and_nothing_else() -> None:
    kept = tuple(name for name in SDIST if not name.endswith(("ids.py", "lib.rs")))
    sdist = (*kept, "sift_rag-2.0/research_notes/a.md", "sift_rag-2.0/tests/test_a.py", "sift_rag-2.0/native/target/x")
    assert problems(WHEEL, sdist, TRACKED, "2.0", "sift 2.0") == [
        "the sdist holds native/target/x, no tracked source or project file",
        "the sdist holds research_notes/a.md, no tracked source or project file",
        "the sdist holds tests/test_a.py, no tracked source or project file",
        "the sdist lacks native/src/lib.rs",
        "the sdist lacks src/sift/api/ids.py",
    ]


def test_the_installed_command_must_say_the_version() -> None:
    assert problems(WHEEL, SDIST, TRACKED, "2.0", "sift 1.9") == ["the installed wheel says 'sift 1.9', not 'sift 2.0'"]


def built(folder: Path, version: str, extra: str | None = None) -> Path:
    sources = [*tracked_sources(), *([extra] if extra else [])]
    with zipfile.ZipFile(folder / f"sift_rag-{version}-cp312-abi3-linux_x86_64.whl", "w") as wheel:
        wheel.mkdir("sift")
        for name in (*(each for each in sources if each.startswith("src/")), "src/sift/_native.abi3.so"):
            wheel.writestr(name.removeprefix("src/"), "")
        wheel.writestr(f"sift_rag-{version}.dist-info/METADATA", "")
        wheel.writestr(f"sift_rag-{version}.dist-info/licenses/native/THIRD-PARTY-NOTICES", "")
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
        "the wheel holds sift/scratch.py, no tracked source of the package",
        "the sdist holds src/sift/scratch.py, no tracked source or project file",
    ]
