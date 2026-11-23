from pathlib import Path
from types import MappingProxyType

from sift.api.events import Noted
from sift.app.problems import EMPTY_INSIDE_REMEDY, LEFT_AS_INDEXED, counted, problems
from sift.domain.plan.indexing import Found, Stat, index_plan
from sift.ingest.scan import Scanned

A = Stat(1, 1)


def said(scanned: Scanned, known: dict[str, Stat]) -> tuple[Noted, ...]:
    return problems(Path("/data/notes"), scanned, index_plan(scanned.found, known), frozenset(known))


def test_a_scan_without_problems_says_nothing() -> None:
    assert said(Scanned(Found({"a.md": A})), {"a.md": A}) == ()


def test_folders_and_files_that_could_not_be_read_are_said_with_why_and_what_was_kept() -> None:
    found = Found({"a.md": A}, unlisted_folders=("robots/", ""), unlisted_files=("rl.md", "new.md"))
    refusals = MappingProxyType({"robots/": "Access is denied", "": "Input/output error", "rl.md": "Access is denied"})
    known = {"a.md": A, "rl.md": A, "robots/brush.md": A}
    assert said(Scanned(found, (), refusals), known) == (
        Noted("robots/ can't be read (Access is denied): 1 file in it kept as indexed"),
        Noted("its folder can't be read (Input/output error): 2 files in it kept as indexed"),
        Noted("rl.md can't be read (Access is denied): kept as indexed"),
        Noted("new.md can't be read (): left for the next index"),
    )


def test_links_left_out_and_names_that_are_no_text_are_counted_with_a_few_named() -> None:
    found = Found({"a.md": A}, outside=("1.md", "2.md", "3.md", "4.md"))
    assert said(Scanned(found, ("r\\udce9sum\\udce9.md",)), {}) == (
        Noted("4 files linked to outside the folder left out: 1.md, 2.md, 3.md, ..."),
        Noted("1 file left out, whose names are no text: r\\udce9sum\\udce9.md"),
    )


def test_linked_folders_are_left_out_and_said() -> None:
    assert said(Scanned(Found({"a.md": A}), linked_folders=("loop/",)), {}) == (
        Noted("1 linked folder left out: loop/"),
    )


def test_a_folder_left_empty_keeps_what_it_held_and_says_so() -> None:
    assert said(Scanned(Found({})), {"a.md": A}) == (
        Noted(LEFT_AS_INDEXED.format(files="1 file", folder=Path("/data/notes"))),
    )
    assert LEFT_AS_INDEXED.startswith("none of its {files} is left in {folder}: kept as indexed")


def test_a_folder_inside_left_empty_keeps_what_it_held_and_says_how_to_let_it_go() -> None:
    found = Found({"a.md": A}, empty_folders=("archive/", "new/"))
    assert said(Scanned(found), {"a.md": A, "archive/b.md": A, "archive/deep/c.md": A}) == (
        Noted(
            "archive/ is empty: 2 files once in it kept as indexed, as a drive not mounted looks so",
            EMPTY_INSIDE_REMEDY,
        ),
    )


def test_counts_are_written_with_their_noun() -> None:
    assert [counted(n, "file") for n in (0, 1, 2, 12_345)] == ["0 files", "1 file", "2 files", "12,345 files"]
