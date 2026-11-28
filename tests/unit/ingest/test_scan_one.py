from pathlib import Path

from sift.domain.plan.ignore import ignore_of
from sift.domain.plan.left_out import WhyLeftOut
from sift.ingest.scan import Reach, scan_one
from tests.files import write


def test_one_file_named_is_found_as_the_scan_finds_it(tmp_path: Path) -> None:
    write(tmp_path / "notes" / "sub" / "a.md", "# A\n")
    scanned = scan_one(tmp_path / "notes", "sub/a.md")
    assert list(scanned.found.files) == ["sub/a.md"]
    assert "sub/a.md" in scanned.identities


def test_a_file_in_a_folder_the_scan_passes_over_is_not_found(tmp_path: Path) -> None:
    for folder in (".git", "node_modules", "kept"):
        write(tmp_path / "notes" / folder / "a.md", "# A\n")
    reach = Reach(ignore=ignore_of(["kept/"], caseless=False))
    for folder in (".git", "node_modules", "kept"):
        assert scan_one(tmp_path / "notes", f"{folder}/a.md", reach).found.files == {}
    assert scan_one(tmp_path / "notes", "kept/a.md", reach).found.left_out == 1


def test_a_file_of_no_kind_sift_reads_or_gone_is_not_found(tmp_path: Path) -> None:
    write(tmp_path / "notes" / "a.exe", "MZ")
    for named in ("a.exe", "gone.md"):
        scanned = scan_one(tmp_path / "notes", named)
        assert (scanned.found.files, scanned.refusals) == ({}, {})


def test_a_file_named_through_a_step_up_or_in_place_leads_outside_on_every_system(tmp_path: Path) -> None:
    write(tmp_path / "other" / "a.md", "# A\n")
    write(tmp_path / "notes" / "b.md", "# B\n")
    for platform in ("linux", "win32"):
        for named in ("../other/a.md", "./b.md"):
            scanned = scan_one(tmp_path / "notes", named, Reach(platform=platform))
            assert (scanned.found.files, scanned.why[named]) == ({}, WhyLeftOut.OUTSIDE)
