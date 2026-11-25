import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from sift.api.filters import Filter
from sift.interfaces.cli.written import WRITTEN_SYNTAX, listed, written
from sift.settings import USER, whole_path

RESERVED = frozenset({"command", "json", "color", "help", "version"})
FROM_HERE = tuple(dots + separator for dots in (".", "..") for separator in sorted({"/", os.sep}))


@dataclass(frozen=True, slots=True)
class Flag:
    long: str | None
    short: str | None = None
    read: Callable[[str], tuple[object, ...]] | None = None
    syntax: str = ""


def one(text: str) -> tuple[str, ...]:
    return (text,)


def filtered(text: str) -> tuple[Filter, ...]:
    return (written(text),)


def file_or_id(user: Path | None, cwd: Path, text: str) -> str:
    if Path(text).parts[:1] == (USER,):
        return os.path.normpath(whole_path(text, user, cwd))
    path = cwd / text
    return os.path.normpath(path) if text.startswith(FROM_HERE) or os.path.lexists(path) else text


LISTED_SYNTAX = "; comma-separated, e.g. url,created"
FLAGS: Mapping[str, Flag] = MappingProxyType(
    {
        "k": Flag(None, "-k"),
        "collections": Flag("--collection", "-c", one),
        "filters": Flag("--filter", "-f", filtered, f"; {WRITTEN_SYNTAX}"),
        "fields": Flag("--fields", None, listed, LISTED_SYNTAX),
        "text_fields": Flag("--text-fields", None, listed, LISTED_SYNTAX),
        "also": Flag("--also", None, listed, LISTED_SYNTAX),
        "links_to": Flag("--links-to", None, one),
        "linked_from": Flag("--linked-from", None, one),
        "files": Flag("--file", None, one),
    }
)


def flag_of(name: str) -> Flag:
    return FLAGS.get(name) or Flag(f"--{name.replace('_', '-')}")
