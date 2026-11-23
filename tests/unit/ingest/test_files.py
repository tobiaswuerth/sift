from pathlib import Path

import pytest

from sift.ingest.files import ChangedError, at_most, identity_at
from tests.files import write


def test_a_file_is_read_only_while_it_is_the_one_scanned(tmp_path: Path) -> None:
    path, other = write(tmp_path / "a.md", "first"), write(tmp_path / "b.md", "second")
    scanned = identity_at(path)
    assert (at_most(path, 10, scanned), at_most(path, 10, None)) == (b"first", b"first")
    other.replace(path)
    with pytest.raises(ChangedError, match="read again at the next index"):
        at_most(path, 10, scanned)
    assert at_most(path, 3, None) == b"seco"
