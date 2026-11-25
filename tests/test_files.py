from pathlib import Path

import pytest

from tests.files import linked


def test_a_link_into_a_folder_that_is_not_there_fails_rather_than_being_skipped(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        linked(tmp_path / "missing" / "link.md", str(tmp_path / "target.md"))
