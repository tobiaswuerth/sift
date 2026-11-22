from collections.abc import Set as AbstractSet
from pathlib import Path

from sift.api.events import Noted
from sift.domain.plan.indexing import ROOT, IndexPlan
from sift.ingest.scan import Scanned

MOST_NAMED = 3
LEFT_AS_INDEXED = (
    "none of its {files} is left in {folder}: kept as indexed, as a drive not mounted and a folder being filled anew "
    "look so"
)


def counted(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def _named(paths: tuple[str, ...]) -> str:
    return ", ".join(paths[:MOST_NAMED]) + (", ..." if len(paths) > MOST_NAMED else "")


def _unlisted_folder(scanned: Scanned, plan: IndexPlan, folder: str) -> Noted:
    held = sum(folder == ROOT or path.startswith(folder) for path in plan.kept)
    why = scanned.refusals.get(folder, "")
    return Noted(f"{folder or 'its folder'} can't be read ({why}): {counted(held, 'file')} in it kept as indexed")


def _unlisted_file(scanned: Scanned, known: AbstractSet[str], path: str) -> Noted:
    kept = "kept as indexed" if path in known else "left for the next index"
    return Noted(f"{path} can't be read ({scanned.refusals.get(path, '')}): {kept}")


def problems(folder: Path, scanned: Scanned, plan: IndexPlan, known: AbstractSet[str]) -> tuple[Noted, ...]:
    found, said = scanned.found, list[Noted]()
    said.extend(_unlisted_folder(scanned, plan, unlisted) for unlisted in found.unlisted_folders)
    said.extend(_unlisted_file(scanned, known, unlisted) for unlisted in found.unlisted_files)
    if found.outside:
        outside = counted(len(found.outside), "file")
        said.append(Noted(f"{outside} linked to outside the folder left out: {_named(found.outside)}"))
    if plan.left_as_indexed:
        said.append(Noted(LEFT_AS_INDEXED.format(files=counted(len(plan.left_as_indexed), "file"), folder=folder)))
    if scanned.unnamed:
        unnamed = counted(len(scanned.unnamed), "file")
        said.append(Noted(f"{unnamed} left out, whose names are no text: {_named(scanned.unnamed)}"))
    return tuple(said)
