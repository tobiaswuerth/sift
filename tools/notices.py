import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, cast

from tools.findings import ROOT, said

NOTICES = ROOT / "native" / "THIRD-PARTY-NOTICES"
MANIFEST = "native/Cargo.toml"
NAMED = re.compile(r"(?:licen[sc]e|copying|notice|copyright)(?:[-_.].*)?", re.IGNORECASE)
TEXTS = frozenset({"", ".txt", ".md"})
MACRO = "proc-macro"
RULE = "=" * 100
USAGE = "usage: python -m tools notices [--write] [FILE]"
HEAD = (
    "Third-party notices\n\n"
    "The compiled part of sift, the extension sift._native, is built of the Rust crates named below, and it is\n"
    "linked with the Rust standard library (https://github.com/rust-lang/rust), which is licensed as most of them\n"
    "are: under the MIT licence or the Apache License 2.0, at your option. The licence files those crates bring\n"
    "follow, each text once, under the crates and files that carry it.\n\n"
)
STALE = (
    "{file} is not what the crates of native/Cargo.lock bring as their licences: after a change of the crate's "
    "dependencies, run `python -m tools notices --write` and commit the file"
)


class NoticeError(Exception):
    pass


class Crate(NamedTuple):
    name: str
    version: str
    licence: str
    folder: Path


def _linked(needs: Mapping[str, Sequence[str]], macros: frozenset[str], root: str) -> set[str]:
    found: set[str] = set()
    waiting = [root]
    while waiting:
        for needed in needs[waiting.pop()]:
            if needed not in found and needed not in macros:
                found.add(needed)
                waiting.append(needed)
    return found


def linked(described: Mapping[str, object]) -> list[Crate]:
    packages = cast("list[dict[str, object]]", described["packages"])
    resolve = cast("dict[str, object]", described["resolve"])
    nodes = cast("list[dict[str, object]]", resolve["nodes"])
    needs = {
        str(node["id"]): [
            str(dep["pkg"])
            for dep in cast("list[dict[str, object]]", node["deps"])
            if any(kind["kind"] is None for kind in cast("list[dict[str, object]]", dep["dep_kinds"]))
        ]
        for node in nodes
    }
    kinds = {
        str(each["id"]): {
            kind for target in cast("list[dict[str, list[str]]]", each["targets"]) for kind in target["kind"]
        }
        for each in packages
    }
    found = _linked(needs, frozenset(name for name, held in kinds.items() if MACRO in held), str(resolve["root"]))
    crates = [
        Crate(
            str(each["name"]), str(each["version"]), str(each["license"] or ""), Path(str(each["manifest_path"])).parent
        )
        for each in packages
        if each["id"] in found
    ]
    return sorted(crates)


def _tidy(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip("\n") + "\n"


def texts(crate: Crate) -> list[tuple[str, str]]:
    files = sorted(
        each
        for each in crate.folder.rglob("*")
        if each.is_file() and each.suffix.lower() in TEXTS and NAMED.fullmatch(each.stem) is not None
    )
    if not files:
        raise NoticeError(f"{crate.name} {crate.version} brings no licence file: its notice cannot be shipped")
    found: list[tuple[str, str]] = []
    for each in files:
        try:
            found.append((each.relative_to(crate.folder).as_posix(), _tidy(each.read_bytes().decode("utf-8"))))
        except UnicodeDecodeError:
            raise NoticeError(f"{each.name} of {crate.name} {crate.version} is no UTF-8 text") from None
    return sorted(found)


def notices(crates: Iterable[tuple[Crate, Sequence[tuple[str, str]]]]) -> str:
    carried: dict[str, list[str]] = {}
    for crate, files in sorted(crates):
        licence = f" ({crate.licence})" if crate.licence else ""
        for name, text in sorted(files):
            carried.setdefault(text, []).append(f"{crate.name} {crate.version}{licence}: {name}")
    parts = [f"{RULE}\n{'\n'.join(names)}\n{RULE}\n\n{text}" for text, names in sorted(carried.items(), key=_first)]
    return HEAD + "\n".join(parts)


def _first(held: tuple[str, list[str]]) -> str:
    return held[1][0]


def current() -> str:
    asked = (
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--manifest-path",
        MANIFEST,
        "--features",
        "python",
        "--locked",
    )
    described = cast("dict[str, object]", json.loads(said(*asked)))
    return notices((crate, texts(crate)) for crate in linked(described))


def main(argv: list[str], shipped: Path = NOTICES) -> int:
    match argv:
        case ["--write", *files] if len(files) <= 1:
            Path(*files or [shipped]).write_bytes(current().encode("utf-8"))
            return 0
        case [*files] if len(files) <= 1 and not any(each.startswith("-") for each in files):
            file = Path(*files or [shipped])
            if file.is_file() and file.read_bytes() == current().encode("utf-8"):
                return 0
            sys.stderr.write(f"error  {STALE.format(file=file)}\n")
            return 1
        case _:
            sys.stderr.write(f"{USAGE}\n")
            return 2
