import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from sift.api.fields import listed
from sift.api.filters import Filter, written

RESERVED = frozenset({"command", "json", "color", "help", "version"})
FROM_HERE = tuple(dots + separator for dots in (".", "..") for separator in sorted({"/", os.sep}))


@dataclass(frozen=True, slots=True)
class Flag:
    long: str | None
    short: str | None = None
    read: Callable[[str], tuple[object, ...]] | None = None


def one(text: str) -> tuple[str, ...]:
    return (text,)


def filtered(text: str) -> tuple[Filter, ...]:
    return (written(text),)


def file_or_id(text: str) -> tuple[str, ...]:
    from_here = text.startswith(FROM_HERE)
    return (os.path.normpath(Path(text).absolute()) if from_here or os.path.lexists(text) else text,)


FLAGS: Mapping[str, Flag] = MappingProxyType(
    {
        "k": Flag(None, "-k"),
        "collections": Flag("--collection", "-c", one),
        "filters": Flag("--filter", "-f", filtered),
        "fields": Flag("--fields", None, listed),
        "text_fields": Flag("--text-fields", None, listed),
        "also": Flag("--also", None, listed),
        "links_to": Flag("--links-to", None, one),
        "linked_from": Flag("--linked-from", None, one),
        "files": Flag("--file", None, file_or_id),
    }
)


def flag_of(name: str) -> Flag:
    return FLAGS.get(name) or Flag(f"--{name.replace('_', '-')}")
