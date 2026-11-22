import os
import stat
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from sift.domain.plan.indexing import MOST_TIME, Stat
from sift.ingest.scan import Entry, Listing, Scanned, scan


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

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        return self.folder and (follow_symlinks or not self.symlink)

    def is_junction(self) -> bool:
        return self.junction

    def is_symlink(self) -> bool:
        return self.symlink

    def stat(self) -> os.stat_result:
        if self.refusal is not None:
            raise self.refusal
        return os.stat_result((self.mode, 0, 0, 0, 0, 0, 3, 0, 0, 0), {"st_mtime_ns": self.modified_ns})


@dataclass(frozen=True, slots=True)
class Breaking:
    entries: tuple[Entry, ...]
    error: OSError

    def __iter__(self) -> Iterator[Entry]:
        yield from self.entries
        raise self.error


type Listed = Iterable[Entry] | OSError


def listing(folders: Mapping[str, Listed]) -> Listing:
    @contextmanager
    def listed(folder: str) -> Generator[Iterable[Entry]]:
        given = folders.get(folder)
        if given is None:
            with os.scandir(folder) as real:
                yield list(real)
            return
        if isinstance(given, OSError):
            raise given
        yield given

    def opened(folder: str) -> AbstractContextManager[Iterable[Entry]]:
        given = folders.get(folder)
        if isinstance(given, OSError):
            raise given
        return listed(folder)

    return opened


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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


def test_a_missing_folder_holds_nothing(tmp_path: Path) -> None:
    scanned = scan(tmp_path / "missing", follow_links=False)
    assert (scanned.found.files, scanned.found.unlisted_folders) == ({}, ())


def test_a_folder_that_cannot_be_listed_is_named(notes: Path) -> None:
    denied = PermissionError(13, "Access is denied")
    scanned = scan(notes, follow_links=False, listing=listing({str(notes / "robots"): denied}))
    assert (paths(scanned)[-1], scanned.found.unlisted_folders) == ("rl.md", ("robots/",))
    assert scan(notes, follow_links=False, listing=listing({str(notes): denied})).found.unlisted_folders == ("",)


def test_a_listing_that_breaks_off_names_its_folder(notes: Path) -> None:
    broken = Breaking((), OSError(5, "Input/output error"))
    scanned = scan(notes, follow_links=False, listing=listing({str(notes / "robots"): broken}))
    assert scanned.found.unlisted_folders == ("robots/",)


def test_a_folder_gone_or_become_a_file_meanwhile_holds_nothing(notes: Path) -> None:
    for gone in (NotADirectoryError(20, "Not a directory"), FileNotFoundError(2, "No such file")):
        scanned = scan(notes, follow_links=False, listing=listing({str(notes / "robots"): gone}))
        assert ("robots/brush.md" not in scanned.found.files, scanned.found.unlisted_folders) == (True, ())


def test_a_file_that_cannot_be_stated_is_named_and_one_gone_passed_over(tmp_path: Path) -> None:
    root = str(tmp_path)
    entries = [
        Fake("denied.md", "denied.md", refusal=PermissionError(13, "Access is denied")),
        Fake("gone.md", "gone.md", refusal=FileNotFoundError(2, "No such file")),
        Fake("pipe.md", "pipe.md", mode=stat.S_IFIFO | 0o644),
        Fake("ok.md", "ok.md"),
    ]
    scanned = scan(tmp_path, follow_links=False, listing=listing({root: entries}))
    assert (paths(scanned), scanned.found.unlisted_files) == (["ok.md"], ("denied.md",))


def test_no_junction_and_no_linked_folder_is_followed(tmp_path: Path) -> None:
    entries = [
        Fake("loop", str(tmp_path), folder=True, junction=True),
        Fake("away", str(tmp_path / "elsewhere"), folder=True, symlink=True),
        Fake("ok.md", "ok.md"),
    ]
    write(tmp_path / "elsewhere" / "outside.md")
    assert paths(scan(tmp_path, follow_links=False, listing=listing({str(tmp_path): entries}))) == ["ok.md"]


def test_a_file_linked_from_outside_the_folder_is_left_out_unless_links_are_followed(tmp_path: Path) -> None:
    root, secret = tmp_path / "notes", write(tmp_path / "outside" / "secret.txt")
    own = write(root / "rl.md")
    entries = [Fake("notes.txt", str(secret), symlink=True), Fake("again.md", str(own), symlink=True)]
    fake = listing({str(root): entries})
    left = scan(root, follow_links=False, listing=fake)
    assert (paths(left), left.found.outside) == (["again.md"], ("notes.txt",))
    followed = scan(root, follow_links=True, listing=fake)
    assert (paths(followed), followed.found.outside) == (["again.md", "notes.txt"], ())


def test_a_file_name_that_is_no_text_is_named_apart(tmp_path: Path) -> None:
    entries = [Fake("r\udce9sum\udce9.md", "x"), Fake("ok.md", "ok.md")]
    scanned = scan(tmp_path, follow_links=False, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.unnamed) == (["ok.md"], ("r\\udce9sum\\udce9.md",))


def test_a_time_past_what_a_store_counts_to_is_held_at_its_end(tmp_path: Path) -> None:
    entries = [Fake("old.md", "old.md", modified_ns=-(2**70)), Fake("new.md", "new.md", modified_ns=2**70)]
    files = scan(tmp_path, follow_links=False, listing=listing({str(tmp_path): entries})).found.files
    assert (files["old.md"].modified_ns, files["new.md"].modified_ns) == (-MOST_TIME, MOST_TIME)


def test_a_link_to_a_folder_whose_name_begins_as_the_root_s_leads_outside(tmp_path: Path) -> None:
    root, sibling = tmp_path / "notes", write(tmp_path / "notes2" / "a.md")
    root.mkdir()
    entries = [Fake("a.md", str(sibling), symlink=True)]
    assert scan(root, follow_links=False, listing=listing({str(root): entries})).found.outside == ("a.md",)
