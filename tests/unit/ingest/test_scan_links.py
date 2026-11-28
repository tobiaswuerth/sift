import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from sift.ingest.files import CHANGED, Identity
from sift.ingest.scan import LinkChecks, Reach, scan
from tests.files import linked, write
from tests.listings import FAKE, Fake, links, listing, paths


def test_a_link_that_cannot_be_followed_is_taken_for_a_link_to_a_file_not_to_a_folder(tmp_path: Path) -> None:
    denied = PermissionError(13, "Access is denied")
    entries = [Fake("odd.md", "odd.md", symlink=True, unfollowed=denied), Fake("ok.md", "ok.md")]
    scanned = scan(tmp_path, links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.linked_folders) == (["odd.md", "ok.md"], ())


def test_no_junction_and_no_linked_folder_is_followed_and_each_is_named_as_left_out(tmp_path: Path) -> None:
    entries = [
        Fake("loop", str(tmp_path), folder=True, junction=True),
        Fake("away", str(tmp_path / "elsewhere"), folder=True, symlink=True),
        Fake("ok.md", "ok.md"),
    ]
    write(tmp_path / "elsewhere" / "outside.md")
    scanned = scan(tmp_path, links=FAKE, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.linked_folders) == (["ok.md"], ("loop/", "away/"))


def test_a_file_linked_from_outside_the_folder_is_left_out_unless_links_are_followed(tmp_path: Path) -> None:
    root, secret = tmp_path / "notes", write(tmp_path / "outside" / "secret.txt")
    write(root / "rl.md")
    entries = [Fake("notes.txt", str(root / "notes.txt"), symlink=True), Fake("again.md", "again.md", symlink=True)]
    fake, inside = listing({str(root): entries}), root.resolve()
    targets = links({str(inside / "notes.txt"): str(secret), str(inside / "again.md"): "rl.md"})
    left = scan(root, links=targets, listing=fake)
    assert (paths(left), left.found.outside) == (["again.md"], ("notes.txt",))
    followed = scan(root, Reach(follow_links=True), links=targets, listing=fake)
    assert (paths(followed), followed.found.outside) == (["again.md", "notes.txt"], ())


def test_a_link_to_a_folder_whose_name_begins_as_the_root_s_leads_outside(tmp_path: Path) -> None:
    root, sibling = tmp_path / "notes", write(tmp_path / "notes2" / "a.md")
    root.mkdir()
    entries = [Fake("a.md", "a.md", symlink=True)]
    targets = links({str(root.resolve() / "a.md"): str(sibling)})
    assert scan(root, links=targets, listing=listing({str(root): entries})).found.outside == ("a.md",)


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
    left = scan(root, links=links(targets), listing=fake)
    assert (left.found.unlisted_files, left.found.outside, left.found.files) == (("inside.md",), ("outside.md",), {})
    assert left.refusals == {"inside.md": "No such file or directory"}
    assert scan(root, Reach(follow_links=True), listing=fake).found.unlisted_files == ("inside.md", "outside.md")


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
    scanned = scan(root, links=links(whole), listing=listing({str(root): entries}))
    assert (scanned.found.outside, len(scanned.found.files)) == (outside, 1 - len(outside))


def test_a_link_stepping_up_past_a_linked_folder_is_left_out_whichever_way_the_system_reads_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    (root / "p" / "q").mkdir(parents=True)
    write(tmp_path / "x.md", "outside")
    linked(root / "d", str(root / "p" / "q"), folder=True)
    linked(root / "f.md", "d/../../x.md")
    assert scan(root).found.outside == ("f.md",)


def test_a_link_written_through_another_spelling_of_the_folder_stays_inside(tmp_path: Path) -> None:
    real = tmp_path / "real"
    write(real / "x.md", "inside")
    linked(tmp_path / "alias", str(real), folder=True)
    linked(real / "f.md", str(tmp_path / "alias" / "x.md"))
    scanned = scan(real)
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
    scanned = scan(tmp_path, links=checks, listing=listing({str(tmp_path): entries}))
    assert (paths(scanned), scanned.found.unlisted_files, scanned.refusals["a.md"]) == ([], ("a.md",), CHANGED)


@dataclass(frozen=True, slots=True)
class Untouchable:
    name: str
    path: str

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        if follow_symlinks:
            raise AssertionError(f"{self.name} was followed before sift knew where it leads")
        return False

    def is_junction(self) -> bool:
        return False

    def is_symlink(self) -> bool:
        return True

    def stat(self) -> os.stat_result:
        raise AssertionError(f"{self.name} was followed before sift knew where it leads")


def test_a_link_that_leads_outside_is_named_without_being_followed(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    entries = [
        Untouchable("x.md", str(root / "x.md")),
        Untouchable("shared", str(root / "shared")),
        Fake("data.bin", str(root / "data.bin"), symlink=True),
    ]
    outside = {str(root.resolve() / "x.md"): str(tmp_path / "share" / "x.md")}
    targets = links({**outside, str(root.resolve() / "shared"): str(tmp_path / "share")})
    scanned = scan(root, links=targets, listing=listing({str(root): entries}))
    assert (scanned.found.outside, scanned.found.files, scanned.linked_folders) == (("x.md",), {}, ())
    assert scanned.found.unlisted_files == ()
