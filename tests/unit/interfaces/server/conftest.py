from pathlib import Path

import pytest

from tests.files import write


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    write(tmp_path / "notes" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    write(tmp_path / "notes" / "brush.md", "# Brush\n\nA robot plans brush strokes.\n")
    return tmp_path / "notes"
