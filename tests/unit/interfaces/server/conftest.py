from pathlib import Path

import pytest

from sift.settings import Settings
from tests.files import write


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", user=None, cwd=tmp_path, state=tmp_path / "state")


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    write(tmp_path / "notes" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    write(tmp_path / "notes" / "brush.md", "# Brush\n\nA robot plans brush strokes.\n")
    return tmp_path / "notes"
