from pathlib import Path

import pytest

from sift.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", user=None, cwd=tmp_path, state=tmp_path / "state")
