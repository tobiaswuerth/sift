from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sift.api.indexing import FileState


@dataclass(frozen=True, slots=True)
class Changes:
    new: tuple[str, ...]
    changed: tuple[str, ...]
    gone: tuple[str, ...]


def changes(held: Mapping[str, str], read: Sequence[tuple[str, str]]) -> Changes:
    found = {key for key, _ in read}
    return Changes(
        new=tuple(key for key, _ in read if key not in held),
        changed=tuple(key for key, digest in read if key in held and held[key] != digest),
        gone=tuple(key for key in held if key not in found),
    )


@dataclass(frozen=True, slots=True)
class FileSeen:
    refusal: str | None
    held: bool
    listed: bool
    gone: bool
    left_out_because: str


@dataclass(frozen=True, slots=True)
class FileOutcome:
    state: FileState
    note: str | None = None
    system: bool = False


def state_unread(seen: FileSeen) -> FileOutcome | None:
    if seen.refusal is not None:
        return FileOutcome("unreadable", seen.refusal, system=True)
    if seen.held or seen.listed:
        return None
    return FileOutcome("missing") if seen.gone else FileOutcome("left out", seen.left_out_because)


def state_read(seen: FileSeen, *, held_as_it_is: bool, problem: str | None, changed: bool) -> FileOutcome:
    if not seen.listed:
        return FileOutcome("removed", None if seen.gone else seen.left_out_because)
    if not held_as_it_is:
        return FileOutcome("unreadable", problem)
    return FileOutcome("added" if not seen.held else "changed" if changed else "unchanged", problem)
