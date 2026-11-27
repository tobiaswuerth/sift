import tarfile
import tomllib
import zipfile
from collections.abc import Iterable
from pathlib import Path

from tools.findings import read, tracked

PACKAGE = "src/"
SOURCES = ("src/sift", "native", "rust-toolchain.toml")
EXTENSIONS = frozenset({"sift/_native.abi3.so", "sift/_native.pyd"})
PROJECT_FILES = frozenset({"pyproject.toml", "pyproject.toml.orig", "README.md", "LICENSE", "PKG-INFO"})


def project_version(pyproject: str) -> str:
    document: dict[str, object] = tomllib.loads(pyproject)
    match document:
        case {"project": {"version": str(version)}}:
            return version
        case _:
            raise TypeError("project.version in pyproject.toml is a string")


def _extension(built: list[str]) -> list[str]:
    if len(built) == 1:
        return []
    return [
        f"the wheel holds the extension twice: {', '.join(built)}"
        if built
        else "the wheel lacks the extension sift/_native"
    ]


def problems(wheel: Iterable[str], sdist: Iterable[str], tracked: Iterable[str], version: str, said: str) -> list[str]:
    sources = frozenset(tracked)
    metadata, root = f"sift_rag-{version}.dist-info/", f"sift_rag-{version}/"
    held = sorted(name for name in wheel if not name.startswith(metadata))
    packaged = [name for name in held if name not in EXTENSIONS]
    package = {name for name in sources if name.startswith(PACKAGE)}
    shipped = {name.removeprefix(root) for name in sdist}
    return [
        *(
            f"the wheel holds {name}, no tracked source of the package"
            for name in packaged
            if PACKAGE + name not in package
        ),
        *(f"the wheel lacks {name}" for name in sorted(package - {PACKAGE + name for name in packaged})),
        *_extension([name for name in held if name in EXTENSIONS]),
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
    return tracked(SOURCES)


def main(argv: list[str]) -> int:
    dist, said = Path(argv[0]), argv[1]
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    version = project_version(read("pyproject.toml"))
    found = problems(files_of_wheel(wheel), files_of_sdist(sdist), tracked_sources(), version, said)
    for problem in found:
        print(problem)
    return 1 if found else 0
