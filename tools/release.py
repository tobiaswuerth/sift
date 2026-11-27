import re
import sys
from collections.abc import Iterable
from pathlib import Path
from types import MappingProxyType

from tools.check_package import project_version
from tools.findings import read

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
SECTION = re.compile(r"^(## \[.*)$", re.MULTILINE)
DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
LINKS = re.compile(r"^\[[^\]\n]+\]: .*\n?", re.MULTILINE)
PRE_RELEASE = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+(?:\.dev[0-9]+)?|\.dev[0-9]+)")
USAGE = "usage: python -m tools release (notes VERSION | prerelease VERSION | files FOLDER)"
WHEEL = "cp312-abi3"
OLD_LINUX = "manylinux(?:2014|_2_(?:1[7-9]|2[0-8]))"
PLATFORMS = MappingProxyType(
    {
        "Linux on x86_64": re.compile(f"{OLD_LINUX}_x86_64"),
        "Linux on aarch64": re.compile(f"{OLD_LINUX}_aarch64"),
        "macOS on arm64": re.compile("macosx_11_0_arm64"),
        "Windows on x86_64": re.compile("win_amd64"),
    }
)
REMEDY = (
    "A release publishes the sources and one wheel for each platform, built by .github/workflows/wheels.yml: "
    "Linux's in its manylinux image, so that it runs where glibc is 2.28 or newer, macOS's for 11.0 on. A wheel "
    "built on another machine names the system it was built on, and few could install it."
)


class RefusedError(Exception):
    pass


def _dated(heading: str, version: str) -> bool:
    opening = f"## [{version}] - "
    return heading.startswith(opening) and DAY.fullmatch(heading.removeprefix(opening)) is not None


def notes(changelog: str, version: str) -> str:
    parts = SECTION.split(changelog)
    found = [body for heading, body in zip(parts[1::2], parts[2::2], strict=True) if _dated(heading, version)]
    if not found:
        raise RefusedError(
            f"CHANGELOG.md has no section '## [{version}] - YYYY-MM-DD': add it, dated the day it is out"
        )
    said = LINKS.sub("", found[0]).strip()
    if not said:
        raise RefusedError(f"the section of {version} in CHANGELOG.md says nothing: write what changed")
    return said + "\n"


def pre_release(version: str) -> bool:
    return PRE_RELEASE.fullmatch(version) is not None


def _platform(name: str, version: str) -> str | None:
    opening, ending = f"sift_rag-{version}-{WHEEL}-", ".whl"
    if not (name.startswith(opening) and name.endswith(ending)):
        return None
    tags = name.removeprefix(opening).removesuffix(ending).split(".")
    return next((platform for platform, tag in PLATFORMS.items() if all(map(tag.fullmatch, tags))), None)


def unpublishable(files: Iterable[str], version: str) -> list[str]:
    held, sources = sorted(files), f"sift_rag-{version}.tar.gz"
    wheels = {platform: [name for name in held if _platform(name, version) == platform] for platform in PLATFORMS}
    known = {sources, *(name for names in wheels.values() for name in names)}
    return [
        *([] if sources in held else [f"no {sources}: the sources are published with the wheels"]),
        *(f"no wheel for {platform}" for platform, names in wheels.items() if not names),
        *(
            f"{len(names)} wheels for {platform}: {', '.join(names)}"
            for platform, names in wheels.items()
            if len(names) > 1
        ),
        *(f"{name} is of no platform a release publishes" for name in held if name not in known),
    ]


def main(argv: list[str]) -> int:
    match argv:
        case ["notes", version]:
            try:
                sys.stdout.write(notes(CHANGELOG.read_text(encoding="utf-8"), version))
            except RefusedError as refused:
                sys.stderr.write(f"error  {refused}\n")
                return 1
            return 0
        case ["prerelease", version]:
            sys.stdout.write(f"{str(pre_release(version)).lower()}\n")
            return 0
        case ["files", folder]:
            names = [each.name for each in Path(folder).iterdir()]
            found = unpublishable(names, project_version(read("pyproject.toml")))
            sys.stderr.write("".join(f"error  {problem}\n" for problem in found) + (f"{REMEDY}\n" if found else ""))
            return 1 if found else 0
        case _:
            sys.stderr.write(f"{USAGE}\n")
            return 2
