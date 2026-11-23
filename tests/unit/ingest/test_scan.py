import os
import stat
import sys
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from sift.domain.plan.indexing import MOST_TIME, Stat, index_plan
from sift.ingest.files import CHANGED, Identity
from sift.ingest.scan import HIDDEN_SYSTEM, DirEntry, LinkChecks, Listing, Scanned, scan
from tests.files import linked, write


@dataclass(frozen=True, slots=True)
class Fake:
    name: str
    path: str
    folder: bool = False
    junction: bool = False
    symlink: bool = False
    mode: int = stat.S_IFREG | 0o644
    modified_ns: int = 7
    refusal: OSError | None = None
    untyped: OSError | None = None
    attributes: int = 0

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        if self.untyped is not None:
            raise self.untyped
        return self.folder and (follow_symlinks or not self.symlink)

    def is_junction(self) -> bool:
        return self.junction

    def is_symlink(self) -> bool:
        return self.symlink

    def stat(self) -> os.stat_result:
        if self.refusal is not None:
            raise self.refusal
        given = {"st_mtime_ns": self.modified_ns, "st_file_attributes": self.attributes}
        held = {name: value for name, value in given.items() if hasattr(os.stat_result, name)}
        return os.stat_result((self.mode, 0, 0, 0, 0, 0, 3, 0, 0, 0), held)


@dataclass(frozen=True, slots=True)
class Breaking:
    entries: tuple[DirEntry, ...]
    error: OSError

    def __iter__(self) -> Iterator[DirEntry]:
        yield from self.entries
        raise self.error


type Listed = Iterable[DirEntry] | OSError


def links(targets: Mapping[str, str]) -> LinkChecks:
    def target(path: Path) -> str | None:
        return targets.get(str(path))

    def identity(path: Path) -> Identity:
        return Identity(1, len(path.parts))

    return LinkChecks(target, identity, identity)


FAKE = links({})


def listing(folders: Mapping[str, Listed]) -> Listing:
    @contextmanager
    def listed(folder: str) -> Generator[Iterable[DirEntry]]:
        given = folders.get(folder)
        if given is None:
            with os.scandir(folder) as real:
                yield list(real)
            return
        if isinstance(given, OSError):
            raise given
        yield given

    def opened(folder: str) -> AbstractContextManager[Iterable[DirEntry]]:
        given = folders.get(folder)
        if isinstance(given, OSError):
            raise given
        return listed(folder)

    return opened


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    for name in ("rl.md", "plain.txt", "robots/brush.md", "UPPER.MD", "long.markdown", ".notes.md"):
        write(root / name)
    for name in (".md", "script.py", ".hidden/secret.md", "node_modules/dep.md", "__pycache__/x.md"):
        write(root / name)
    return root


def paths(scanned: Scanned) -> list[str]:
    return sorted(scanned.found.files)


def test_text_and_markdown_files_are_found_but_in_hidden_and_vendor_folders(notes: Path) -> None:
    assert paths(scan(notes, follow_links=False)) == [
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
    assert scan(notes, follow_links=False).found.files["rl.md"] == Stat(found.st_mtime_ns, 4)


def test_a_folder_gone_before_its_scan_holds_nothing_and_is_named_as_unlisted(tmp_path: Path) -> None:
    scanned = scan(tmp_path / "missing", follow_links=False)
    assert (scanned.found.files, scanned.found.unlisted_folders) == ({}, ("",))


def test_a_folder_that_cannot_be_listed_is_named(notes: Path) -> None:
    denied = PermissionError(13, "Access is denied")
    scanned = scan(notes, follow_links=False, links=FAKE, listing=listing({str(notes / "robots"): denied}))
    assert (paths(scanned)[-1], scanned.found.unlisted_folders) == ("rl.md", ("robots/",))
    assert scan(
        notes, follow_links=False, links=FAKE, listing=listing({str(notes): denied})
    ).found.unlisted_folders == ("",)


def test_a_listing_that_breaks_off_names_its_folder(notes: Path) -> None:
    broken = Breaking((), OSError(5, "Input/output error"))
    scanned = scan(notes, follow_links=False, links=FAKE, listing=listing({str(notes / "robots"): broken}))
    assert scanned.found.unlisted_folders == ("robots/",)


def test_a_folder_that_vanishes_while_scanned_keeps_what_it_held(notes: Path) -> None:
    for gone in (NotADirectoryError(20, "Not a directory"), FileNotFoundError(2, "No such file")):
        scanned = scan(notes, follow_links=False, links=FAKE, listing=listing({str(notes / "robots"): gone}))
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
    scanned = scan(tmp_path, follow_links=False, links=FAKE, listing=listing({root: entries}))
    assert (paths(scanned), scanned.found.unlisted_files) == (["ok.md"], ("denied.md",))


def test_no_junction_and_no_linked_folder_is_followed(tmp_path: Path) -> None:
    entries = [
        Fake("loop", str(tmp_path), folder=True, junction=True),
        Fake("away", str(tmp_path / "elsewhere"), folder=True, symlink=True),
        Fake("ok.md", "ok.md"),
    ]
    write(tmp_path / "elsewhere" / "outside.md")
    scanned = scan(tmp_path, follow_links=False, links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.linked_folders) == (["ok.md"], ("loop/",))


def test_a_file_linked_from_outside_the_folder_is_left_out_unless_links_are_followed(tmp_path: Path) -> None:
    root, secret = tmp_path / "notes", write(tmp_path / "outside" / "secret.txt")
    write(root / "rl.md")
    entries = [Fake("notes.txt", str(root / "notes.txt"), symlink=True), Fake("again.md", "again.md", symlink=True)]
    fake, inside = listing({str(root): entries}), root.resolve()
    targets = links({str(inside / "notes.txt"): str(secret), str(inside / "again.md"): "rl.md"})
    left = scan(root, follow_links=False, links=targets, listing=fake)
    assert (paths(left), left.found.outside) == (["again.md"], ("notes.txt",))
    followed = scan(root, follow_links=True, links=targets, listing=fake)
    assert (paths(followed), followed.found.outside) == (["again.md", "notes.txt"], ())


def test_a_file_name_that_is_no_text_is_named_apart(tmp_path: Path) -> None:
    entries = [Fake("r\udce9sum\udce9.md", "x"), Fake("ok.md", "ok.md")]
    scanned = scan(tmp_path, follow_links=False, links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.unnamed) == (["ok.md"], ("r\\udce9sum\\udce9.md",))


def test_a_time_past_what_a_store_counts_to_is_held_at_its_end(tmp_path: Path) -> None:
    entries = [Fake("old.md", "old.md", modified_ns=-(2**70)), Fake("new.md", "new.md", modified_ns=2**70)]
    files = scan(tmp_path, follow_links=False, links=FAKE, listing=listing({str(tmp_path): entries})).found.files
    assert (files["old.md"].modified_ns, files["new.md"].modified_ns) == (-MOST_TIME, MOST_TIME)


def test_a_link_to_a_folder_whose_name_begins_as_the_root_s_leads_outside(tmp_path: Path) -> None:
    root, sibling = tmp_path / "notes", write(tmp_path / "notes2" / "a.md")
    root.mkdir()
    entries = [Fake("a.md", "a.md", symlink=True)]
    targets = links({str(root.resolve() / "a.md"): str(sibling)})
    assert scan(root, follow_links=False, links=targets, listing=listing({str(root): entries})).found.outside == (
        "a.md",
    )


def test_a_link_whose_file_is_out_of_reach_is_named_not_taken_for_gone(tmp_path: Path) -> None:
    root, missing = tmp_path / "notes", FileNotFoundError(2, "No such file or directory")
    root.mkdir()
    entries = [
        Fake("inside.md", "inside.md", symlink=True, refusal=missing),
        Fake("outside.md", "outside.md", symlink=True, refusal=missing),
        Fake("deleted.md", "deleted.md", refusal=missing),
    ]
    fake, inside = listing({str(root): entries}), root.resolve()
    targets = {
        str(inside / "inside.md"): str(inside / "elsewhere.md"),
        str(inside / "outside.md"): str(tmp_path / "drive" / "x.md"),
    }
    left = scan(root, follow_links=False, links=links(targets), listing=fake)
    assert (left.found.unlisted_files, left.found.outside, left.found.files) == (("inside.md",), ("outside.md",), {})
    assert left.refusals == {"inside.md": "No such file or directory"}
    assert scan(root, follow_links=True, listing=fake).found.unlisted_files == ("inside.md", "outside.md")


def test_an_entry_whose_kind_cannot_be_told_is_named_as_a_file_and_as_a_folder(tmp_path: Path) -> None:
    entries = [Fake("odd", "odd", untyped=PermissionError(13, "Access is denied")), Fake("ok.md", "ok.md")]
    scanned = scan(tmp_path, follow_links=False, links=FAKE, listing=listing({str(tmp_path): entries}))
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
    assert sorted(scan(notes, follow_links=False, links=FAKE, listing=fake).found.empty_folders) == [
        "archive/",
        "robots/none/",
    ]


def test_an_empty_root_is_no_empty_folder_inside(tmp_path: Path) -> None:
    assert scan(tmp_path, follow_links=False).found.empty_folders == ()


def test_a_long_scan_says_how_many_files_it_has_found_every_thousand_entries(tmp_path: Path) -> None:
    entries = [Fake(f"{n}.md" if n % 2 else f"{n}.png", str(n)) for n in range(2500)]
    said: list[int] = []
    scan(tmp_path, follow_links=False, links=FAKE, listing=listing({str(tmp_path): entries}), progress=said.append)
    assert said == [500, 1000]


@pytest.mark.parametrize(
    "targets, outside",
    [
        ({"a.md": "b.md", "b.md": "far"}, ("a.md",)),
        ({"a.md": "sub/x.md", "sub": "far"}, ("a.md",)),
        ({"a.md": "../x.md"}, ("a.md",)),
        ({"a.md": "b.md", "b.md": "a.md"}, ("a.md",)),
        ({"a.md": "sub/../b.md", "b.md": "sub/c.md"}, ()),
        ({"a.md": "d/../../x.md", "d": "deep"}, ("a.md",)),
        ({"a.md": "d/../x.md", "d": "far"}, ("a.md",)),
        ({"a.md": "root"}, ()),
    ],
)
def test_a_link_is_followed_hop_by_hop_and_left_out_at_the_first_leading_outside(
    tmp_path: Path, targets: dict[str, str], outside: tuple[str, ...]
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    inside, far = root.resolve(), str(tmp_path / "far" / "x.md")
    spelled = {"far": far, "root": str(inside / "c.md"), "deep": str(inside / "p" / "q")}
    whole = {str(inside / path): spelled.get(target, target) for path, target in targets.items()}
    never_read = OSError(5, "a link leading outside was read")
    entries = [Fake("a.md", "a.md", symlink=True, refusal=never_read if outside else None)]
    scanned = scan(root, follow_links=False, links=links(whole), listing=listing({str(root): entries}))
    assert (scanned.found.outside, len(scanned.found.files)) == (outside, 1 - len(outside))


def test_each_file_found_is_known_by_its_identity_unless_links_are_followed(notes: Path) -> None:
    held = scan(notes, follow_links=False).identities
    found = (notes / "rl.md").stat()
    assert (held["rl.md"], len(held)) == (Identity(found.st_dev, found.st_ino), 6)
    assert scan(notes, follow_links=True).identities == {}


@pytest.mark.skipif(sys.platform != "win32", reason="folders are hidden and system on Windows alone")
def test_folders_both_hidden_and_system_are_passed_over(tmp_path: Path) -> None:
    hidden = stat.FILE_ATTRIBUTE_HIDDEN
    entries = [
        Fake("$RECYCLE.BIN", "bin", folder=True, attributes=HIDDEN_SYSTEM),
        Fake("hidden", "hidden", folder=True, attributes=hidden),
    ]
    fake = listing({str(tmp_path): entries, "bin": [], "hidden": []})
    assert scan(tmp_path, follow_links=False, links=FAKE, listing=fake).found.empty_folders == ("hidden/",)


def test_a_file_whose_path_reads_as_a_record_s_id_is_named_not_read(tmp_path: Path) -> None:
    for name in ("a.jsonl#k.md", "b.NDJSON#x/c.md", "a.jsonl"):
        write(tmp_path / name)
    scanned = scan(tmp_path, follow_links=False)
    assert (paths(scanned), sorted(scanned.named_as_records)) == (["a.jsonl"], ["a.jsonl#k.md", "b.NDJSON#x/c.md"])
    assert index_plan(scanned.found, {"a.jsonl#k.md": Stat(1, 1)}).drop == ("a.jsonl#k.md",)


def test_a_link_stepping_up_past_a_linked_folder_is_left_out_whichever_way_the_system_reads_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    (root / "p" / "q").mkdir(parents=True)
    write(tmp_path / "x.md", "outside")
    linked(root / "d", str(root / "p" / "q"), folder=True)
    linked(root / "f.md", "d/../../x.md")
    assert scan(root, follow_links=False).found.outside == ("f.md",)


def test_a_link_written_through_another_spelling_of_the_folder_stays_inside(tmp_path: Path) -> None:
    real = tmp_path / "real"
    write(real / "x.md", "inside")
    linked(tmp_path / "alias", str(real), folder=True)
    linked(real / "f.md", str(tmp_path / "alias" / "x.md"))
    scanned = scan(real, follow_links=False)
    assert (paths(scanned), scanned.found.outside) == (["f.md", "x.md"], ())


def test_a_link_changed_while_it_was_checked_is_named_not_read(tmp_path: Path) -> None:
    inside = tmp_path.resolve()
    swapped: list[Path] = []

    def target(path: Path) -> str | None:
        return None if path != inside / "a.md" else str(tmp_path.parent / "far.md") if swapped else "b.md"

    def identity(path: Path) -> Identity:
        swapped.append(path)
        return Identity(1, 1)

    entries = [Fake("a.md", str(tmp_path / "a.md"), symlink=True)]
    checks = LinkChecks(target, identity, identity)
    scanned = scan(tmp_path, follow_links=False, links=checks, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.found.unlisted_files, scanned.refusals["a.md"]) == ([], ("a.md",), CHANGED)
