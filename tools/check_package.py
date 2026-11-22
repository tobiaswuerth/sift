import subprocess
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable
from pathlib import Path

SOURCES = "src/sift"
PROJECT_FILES = frozenset({"pyproject.toml", "pyproject.toml.orig", "README.md", "LICENSE", "PKG-INFO"})


def project_version(pyproject: str) -> str:
    document: dict[str, object] = tomllib.loads(pyproject)
    match document:
        case {"project": {"version": str(version)}}:
            return version
        case _:
            raise TypeError("project.version in pyproject.toml is a string")


def problems(wheel: Iterable[str], sdist: Iterable[str], tracked: Iterable[str], version: str, said: str) -> list[str]:
    sources = frozenset(tracked)
    metadata, root = f"sift_rag-{version}.dist-info/", f"sift_rag-{version}/"
    packaged = sorted(name for name in wheel if not name.startswith(metadata))
    shipped = {name.removeprefix(root) for name in sdist}
    return [
        *(f"the wheel holds {name}, which git does not track" for name in packaged if f"src/{name}" not in sources),
        *(f"the wheel lacks {name}" for name in sorted(sources - {f"src/{name}" for name in packaged})),
        *(
            f"the sdist holds {name}, no tracked source or project file"
            for name in sorted(shipped - sources - PROJECT_FILES)
        ),
        *(f"the sdist lacks {name}" for name in sorted(sources - shipped)),
        *([] if said == f"sift {version}" else [f"the installed wheel says {said!r}, not 'sift {version}'"]),
    ]


def files_of_wheel(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return [info.filename for info in archive.infolist() if not info.is_dir()]


def files_of_sdist(path: Path) -> list[str]:
    with tarfile.open(path) as archive:
        return [member.name for member in archive.getmembers() if member.isfile()]


def tracked_sources() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", SOURCES], check=True, capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    return [name for name in listed.stdout.split("\0") if name]


def main(argv: list[str]) -> int:
    dist, said = Path(argv[0]), argv[1]
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    version = project_version(Path("pyproject.toml").read_text(encoding="utf-8"))
    found = problems(files_of_wheel(wheel), files_of_sdist(sdist), tracked_sources(), version, said)
    for problem in found:
        print(problem)
    return 1 if found else 0
