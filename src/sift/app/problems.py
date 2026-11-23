from collections import Counter
from collections.abc import Set as AbstractSet
from pathlib import Path

from sift.api.events import Noted
from sift.domain.plan.indexing import ROOT, IndexPlan, folders_of
from sift.ingest.scan import Scanned

MOST_NAMED = 3
LEFT_AS_INDEXED = (
    "none of its {files} is left in {folder}: kept as indexed, as a drive not mounted and a folder being filled anew "
    "look so"
)
EMPTY_INSIDE = "{folder} is empty: {files} once in it kept as indexed, as a drive not mounted looks so"
EMPTY_INSIDE_REMEDY = "connect the drive, or delete the empty folder to let them go"


def counted(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def _named(paths: tuple[str, ...]) -> str:
    return ", ".join(paths[:MOST_NAMED]) + (", ..." if len(paths) > MOST_NAMED else "")


def _held(paths: tuple[str, ...]) -> Counter[str]:
    return Counter(folder for path in paths for folder in folders_of(path))


def _unlisted_folder(scanned: Scanned, kept: Counter[str], folder: str) -> Noted:
    held = kept[folder]
    why = scanned.refusals.get(folder, "")
    return Noted(f"{folder or 'its folder'} can't be read ({why}): {counted(held, 'file')} in it kept as indexed")


def _unlisted_file(scanned: Scanned, known: AbstractSet[str], path: str) -> Noted:
    kept = "kept as indexed" if path in known else "left for the next index"
    return Noted(f"{path} can't be read ({scanned.refusals.get(path, '')}): {kept}")


def _emptied(folder: Path, left: Counter[str], emptied: str) -> Noted:
    held = counted(left[emptied], "file")
    if emptied == ROOT:
        return Noted(LEFT_AS_INDEXED.format(files=held, folder=folder))
    return Noted(EMPTY_INSIDE.format(files=held, folder=emptied), EMPTY_INSIDE_REMEDY)


def problems(folder: Path, scanned: Scanned, plan: IndexPlan, known: AbstractSet[str]) -> tuple[Noted, ...]:
    found, said = scanned.found, list[Noted]()
    kept, left = _held(plan.kept), _held(plan.left_as_indexed)
    said.extend(_unlisted_folder(scanned, kept, unlisted) for unlisted in found.unlisted_folders)
    said.extend(_unlisted_file(scanned, known, unlisted) for unlisted in found.unlisted_files)
    if found.outside:
        outside = counted(len(found.outside), "file")
        said.append(Noted(f"{outside} linked to outside the folder left out: {_named(found.outside)}"))
    said.extend(_emptied(folder, left, emptied) for emptied in plan.emptied)
    if scanned.unnamed:
        unnamed = counted(len(scanned.unnamed), "file")
        said.append(Noted(f"{unnamed} left out, whose names are no text: {_named(scanned.unnamed)}"))
    return tuple(said)
