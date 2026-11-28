import os
from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError
from sift.interfaces.cli.flags import file_or_id
from tests.files import linked


def test_a_file_named_is_a_path_where_it_is_found_or_written_from_here_and_an_id_else(tmp_path: Path) -> None:
    found = tmp_path / "notes" / "a.md"
    found.parent.mkdir()
    found.write_bytes(b"# A\n")
    user = tmp_path / "user"
    assert file_or_id(user, tmp_path, str(found)) == str(found)
    assert file_or_id(user, tmp_path, "notes/a.md") == str(found)
    assert file_or_id(user, tmp_path, "./gone.md") == str(tmp_path / "gone.md")
    assert file_or_id(user, tmp_path, "../gone.md") == os.path.normpath(tmp_path.parent / "gone.md")
    assert file_or_id(user, tmp_path, "~/x.md") == str(user / "x.md")
    assert file_or_id(user, tmp_path, "notes-nowhere/a.md") == "notes-nowhere/a.md"
    with pytest.raises(InvalidRequestError, match="no home folder is set"):
        file_or_id(None, tmp_path, "~/x.md")


def test_a_link_named_is_the_link_and_not_where_it_leads(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "x.md"
    target.parent.mkdir()
    target.write_bytes(b"# X\n")
    link = tmp_path / "notes" / "link.md"
    link.parent.mkdir()
    linked(link, str(target))
    assert file_or_id(None, tmp_path, str(link)) == str(link)
