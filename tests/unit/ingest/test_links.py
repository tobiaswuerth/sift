from pathlib import Path

import pytest

from sift.ingest.links import collapsed, link_target, plain
from tests.files import write


@pytest.mark.parametrize(
    "written, meant",
    [
        ("\\\\?\\C:\\notes\\a.md", "C:\\notes\\a.md"),
        ("\\\\?\\UNC\\host\\share\\a.md", "\\\\host\\share\\a.md"),
        ("a.md", "a.md"),
    ],
)
def test_a_link_s_target_is_read_without_the_long_path_prefix(written: str, meant: str) -> None:
    assert plain(written) == meant


def test_only_a_link_has_a_target(tmp_path: Path) -> None:
    file, link = write(tmp_path / "a.md", "x"), tmp_path / "b.md"
    link.symlink_to("a.md")
    assert (link_target(file), link_target(link)) == (None, "a.md")


def test_steps_up_are_collapsed_by_text_and_none_may_leave_the_drive() -> None:
    assert collapsed(("/", "a", "b", "..", "c")) == ("/", "a", "c")
    assert collapsed(("/", "a", "..", "..")) is None
