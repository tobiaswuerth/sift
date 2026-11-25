import unicodedata
from collections import Counter
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path

from sift.api.errors import Problem
from sift.api.events import Noted
from sift.domain.plan.indexing import ROOT, IndexPlan, Stat, folders_of
from sift.domain.text.wording import counted
from sift.ingest.scan import Scanned

MOST_NAMED = 3
LEFT_AS_INDEXED = (
    "none of its {files} is left in {folder}: kept as indexed, as a drive not mounted and a folder being filled anew "
    "look so"
)
EMPTY_INSIDE = "{folder} is empty: {files} once in it kept as indexed, as a drive not mounted looks so"
NAMED_AS_RECORD_REMEDY = "rename them: a # after .jsonl or .ndjson names a record"
EMPTY_INSIDE_REMEDY = "connect the drive, or delete the empty folder to let them go"
APART = "rename one of each: their ids print alike"


def _named(paths: tuple[str, ...]) -> str:
    return ", ".join(paths[:MOST_NAMED]) + (", ..." if len(paths) > MOST_NAMED else "")


def _held(paths: tuple[str, ...]) -> Counter[str]:
    return Counter(folder for path in paths for folder in folders_of(path))


def _unlisted_folder(scanned: Scanned, kept: Counter[str], folder: str) -> Noted:
    held = kept[folder]
    why = scanned.refusals.get(folder, "")
    said = f"{folder or 'its folder'} can't be read ({why}): {counted(held, 'file')} in it kept as indexed"
    return Noted(said, code=Problem.OS)


def _unlisted_file(scanned: Scanned, known: AbstractSet[str], path: str) -> Noted:
    kept = "kept as indexed" if path in known else "left for the next index"
    return Noted(f"{path} can't be read ({scanned.refusals.get(path, '')}): {kept}", code=Problem.OS)


def _emptied(folder: Path, left: Counter[str], emptied: str) -> Noted:
    held = counted(left[emptied], "file")
    if emptied == ROOT:
        return Noted(LEFT_AS_INDEXED.format(files=held, folder=folder))
    return Noted(EMPTY_INSIDE.format(files=held, folder=emptied), EMPTY_INSIDE_REMEDY)


def _written_apart(files: Mapping[str, Stat]) -> tuple[str, ...]:
    other = Counter(unicodedata.normalize("NFC", path) for path in files if not unicodedata.is_normalized("NFC", path))
    return tuple(sorted(name for name, count in other.items() if count + (name in files) > 1))


def problems(folder: Path, scanned: Scanned, plan: IndexPlan, known: AbstractSet[str]) -> tuple[Noted, ...]:
    found, said = scanned.found, list[Noted]()
    kept, left = _held(plan.kept), _held(plan.left_as_indexed)
    said.extend(_unlisted_folder(scanned, kept, unlisted) for unlisted in found.unlisted_folders)
    said.extend(_unlisted_file(scanned, known, unlisted) for unlisted in found.unlisted_files)
    looping = frozenset(scanned.looped)
    led_out = tuple(path for path in found.outside if path not in looping)
    if led_out:
        said.append(Noted(f"{counted(len(led_out), 'file')} linked to outside the folder left out: {_named(led_out)}"))
    if scanned.looped:
        looped = counted(len(scanned.looped), "file")
        said.append(Noted(f"{looped} whose links go round in a loop left out: {_named(scanned.looped)}"))
    apart = _written_apart(found.files)
    if apart:
        names = counted(len(apart), "name")
        said.append(Noted(f"{names} written both composed and decomposed, a file each way: {_named(apart)}", APART))
    if scanned.named_as_records:
        named, listed = counted(len(scanned.named_as_records), "file"), _named(scanned.named_as_records)
        said.append(Noted(f"{named} left out, whose names read as a record's id: {listed}", NAMED_AS_RECORD_REMEDY))
    if scanned.linked_folders:
        linked = counted(len(scanned.linked_folders), "linked folder")
        said.append(Noted(f"{linked} left out: {_named(scanned.linked_folders)}"))
    said.extend(_emptied(folder, left, emptied) for emptied in plan.emptied)
    if scanned.unnamed:
        unnamed = counted(len(scanned.unnamed), "file")
        said.append(Noted(f"{unnamed} left out, whose names are no text: {_named(scanned.unnamed)}"))
    return tuple(said)
