import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
SECTION = re.compile(r"^(## \[.*)$", re.MULTILINE)
DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
LINKS = re.compile(r"^\[[^\]\n]+\]: .*\n?", re.MULTILINE)
PRE_RELEASE = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+(?:\.dev[0-9]+)?|\.dev[0-9]+)")
USAGE = "usage: python -m tools release (notes | prerelease) VERSION"


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
        case _:
            sys.stderr.write(f"{USAGE}\n")
            return 2
