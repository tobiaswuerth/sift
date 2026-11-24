import stat
import sys
from pathlib import Path

import pytest

from sift.domain.plan.ignore import ignore_of
from sift.domain.plan.indexing import MOST_TIME, Stat, index_plan
from sift.ingest.files import Identity
from sift.ingest.scan import HIDDEN_SYSTEM, Reach, scan
from tests.files import write
from tests.listings import FAKE, Breaking, Fake, listing, paths


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    for name in ("rl.md", "plain.txt", "robots/brush.md", "UPPER.MD", "long.markdown", ".notes.md"):
        write(root / name)
    for name in (".md", "script.py", ".hidden/secret.md", "node_modules/dep.md", "__pycache__/x.md"):
        write(root / name)
    return root


def test_text_and_markdown_files_are_found_but_in_hidden_and_vendor_folders(notes: Path) -> None:
    assert paths(scan(notes)) == [
        ".notes.md",
        "UPPER.MD",
        "long.markdown",
        "plain.txt",
        "rl.md",
        "robots/brush.md",
    ]


def test_a_file_is_found_with_its_time_and_size(notes: Path) -> None:
    write(notes / "rl.md", "four")
    found = (notes / "rl.md").stat()
    assert scan(notes).found.files["rl.md"] == Stat(found.st_mtime_ns, 4)


def test_a_folder_gone_before_its_scan_holds_nothing_and_is_named_as_unlisted(tmp_path: Path) -> None:
    scanned = scan(tmp_path / "missing")
    assert (scanned.found.files, scanned.found.unlisted_folders) == ({}, ("",))


def test_a_folder_that_cannot_be_listed_is_named(notes: Path) -> None:
    denied = PermissionError(13, "Access is denied")
    scanned = scan(notes, links=FAKE, listing=listing({str(notes / "robots"): denied}))
    assert (paths(scanned)[-1], scanned.found.unlisted_folders) == ("rl.md", ("robots/",))
    assert scan(notes, links=FAKE, listing=listing({str(notes): denied})).found.unlisted_folders == ("",)


def test_a_listing_that_breaks_off_names_its_folder(notes: Path) -> None:
    broken = Breaking((), OSError(5, "Input/output error"))
    scanned = scan(notes, links=FAKE, listing=listing({str(notes / "robots"): broken}))
    assert scanned.found.unlisted_folders == ("robots/",)


def test_a_folder_that_vanishes_while_scanned_keeps_what_it_held(notes: Path) -> None:
    for gone in (NotADirectoryError(20, "Not a directory"), FileNotFoundError(2, "No such file")):
        scanned = scan(notes, links=FAKE, listing=listing({str(notes / "robots"): gone}))
        assert ("robots/brush.md" not in scanned.found.files, scanned.found.unlisted_folders) == (True, ("robots/",))
        assert index_plan(scanned.found, {"robots/brush.md": Stat(1, 1)}).kept == ("robots/brush.md",)


def test_a_file_that_cannot_be_stated_is_named_and_one_gone_passed_over(tmp_path: Path) -> None:
    root = str(tmp_path)
    entries = [
        Fake("denied.md", "denied.md", refusal=PermissionError(13, "Access is denied")),
        Fake("gone.md", "gone.md", refusal=FileNotFoundError(2, "No such file")),
        Fake("pipe.md", "pipe.md", mode=stat.S_IFIFO | 0o644),
        Fake("ok.md", "ok.md"),
    ]
    scanned = scan(tmp_path, links=FAKE, listing=listing({root: entries}))
    assert (paths(scanned), scanned.found.unlisted_files) == (["ok.md"], ("denied.md",))


def test_a_file_name_that_is_no_text_is_named_apart(tmp_path: Path) -> None:
    entries = [Fake("r\udce9sum\udce9.md", "x"), Fake("ok.md", "ok.md")]
    scanned = scan(tmp_path, links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.unnamed) == (["ok.md"], ("r\\udce9sum\\udce9.md",))


def test_a_time_past_what_a_store_counts_to_is_held_at_its_end(tmp_path: Path) -> None:
    entries = [Fake("old.md", "old.md", modified_ns=-(2**70)), Fake("new.md", "new.md", modified_ns=2**70)]
    files = scan(tmp_path, links=FAKE, listing=listing({str(tmp_path): entries})).found.files
    assert (files["old.md"].modified_ns, files["new.md"].modified_ns) == (-MOST_TIME, MOST_TIME)


def test_an_entry_whose_kind_cannot_be_told_is_named_as_a_file_and_as_a_folder(tmp_path: Path) -> None:
    entries = [Fake("odd", "odd", untyped=PermissionError(13, "Access is denied")), Fake("ok.md", "ok.md")]
    scanned = scan(tmp_path, links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.found.unlisted_files, scanned.found.unlisted_folders) == (
        ["ok.md"],
        ("odd",),
        ("odd/",),
    )
    assert scanned.refusals == {"odd": "Access is denied", "odd/": "Access is denied"}


def test_folders_inside_that_are_there_but_empty_are_named(notes: Path) -> None:
    for name in ("archive", "robots/none", "denied", "gone"):
        (notes / name).mkdir()
    fake = listing(
        {str(notes / "denied"): PermissionError(13, "Access is denied"), str(notes / "gone"): FileNotFoundError()}
    )
    assert sorted(scan(notes, links=FAKE, listing=fake).found.empty_folders) == [
        "archive/",
        "robots/none/",
    ]


def test_an_empty_root_is_no_empty_folder_inside(tmp_path: Path) -> None:
    assert scan(tmp_path).found.empty_folders == ()


def test_a_long_scan_says_how_many_files_it_has_found_every_thousand_entries(tmp_path: Path) -> None:
    entries = [Fake(f"{n}.md" if n % 2 else f"{n}.png", str(n)) for n in range(2500)]
    said: list[int] = []
    scan(tmp_path, links=FAKE, listing=listing({str(tmp_path): entries}), progress=said.append)
    assert said == [500, 1000]


def test_each_file_found_is_known_by_its_identity_unless_links_are_followed(notes: Path) -> None:
    held = scan(notes).identities
    found = (notes / "rl.md").stat()
    assert (held["rl.md"], len(held)) == (Identity(found.st_dev, found.st_ino), 6)
    assert scan(notes, Reach(follow_links=True)).identities == {}


@pytest.mark.skipif(sys.platform != "win32", reason="folders are hidden and system on Windows alone")
def test_folders_both_hidden_and_system_are_passed_over(tmp_path: Path) -> None:
    hidden = stat.FILE_ATTRIBUTE_HIDDEN
    entries = [
        Fake("$RECYCLE.BIN", "bin", folder=True, attributes=HIDDEN_SYSTEM),
        Fake("hidden", "hidden", folder=True, attributes=hidden),
    ]
    fake = listing({str(tmp_path): entries, "bin": [], "hidden": []})
    assert scan(tmp_path, links=FAKE, listing=fake).found.empty_folders == ("hidden/",)


def test_a_file_whose_path_reads_as_a_record_s_id_is_named_not_read(tmp_path: Path) -> None:
    for name in ("a.jsonl#k.md", "b.NDJSON#x/c.md", "a.jsonl"):
        write(tmp_path / name)
    scanned = scan(tmp_path)
    assert (paths(scanned), sorted(scanned.named_as_records)) == (["a.jsonl"], ["a.jsonl#k.md", "b.NDJSON#x/c.md"])
    assert index_plan(scanned.found, {"a.jsonl#k.md": Stat(1, 1)}).drop == ("a.jsonl#k.md",)


def test_what_siftignore_names_is_left_out_and_a_folder_left_out_is_not_looked_into(notes: Path) -> None:
    write(notes / "robots" / "deep" / "x.md")
    denied = listing({str(notes / "robots"): PermissionError(13, "Access is denied")})
    scanned = scan(notes, Reach(ignore=ignore_of(["plain.txt", "robots/"], caseless=False)), links=FAKE, listing=denied)
    assert (paths(scanned), scanned.found.unlisted_folders) == ([".notes.md", "UPPER.MD", "long.markdown", "rl.md"], ())


def test_a_file_in_a_folder_left_out_stays_out_though_taken_back(tmp_path: Path) -> None:
    for name in ("private/keep.md", "x/private/a/b.md", "public/keep.md"):
        write(tmp_path / name)
    ignore = ignore_of(["private/", "!private/keep.md"], caseless=False)
    scanned = scan(tmp_path, Reach(ignore=ignore))
    assert (paths(scanned), scanned.found.left_out) == (["public/keep.md"], 2)


def test_what_siftignore_leaves_out_is_named_for_nothing(tmp_path: Path) -> None:
    denied = PermissionError(13, "Access is denied")
    entries = [
        Fake("loop", str(tmp_path), folder=True, junction=True),
        Fake("denied.md", "denied.md", refusal=denied),
        Fake("odd", "odd", untyped=denied),
        Fake("ok.md", "ok.md"),
    ]
    ignore = ignore_of(["loop/", "denied.md", "odd"], caseless=False)
    scanned = scan(tmp_path, Reach(ignore=ignore), links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.linked_folders, scanned.found.unlisted_files) == (["ok.md"], (), ())
    assert (scanned.found.unlisted_folders, scanned.refusals) == ((), {})


def test_an_entry_of_untold_kind_left_out_as_a_file_alone_is_named_as_a_folder(tmp_path: Path) -> None:
    entries = [Fake("odd.md", "odd.md", untyped=PermissionError(13, "Access is denied"))]
    ignore = ignore_of(["*.md", "!*/"], caseless=False)
    scanned = scan(tmp_path, Reach(ignore=ignore), links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (scanned.found.unlisted_files, scanned.found.unlisted_folders) == ((), ("odd.md/",))
