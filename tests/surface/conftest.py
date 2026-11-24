from pathlib import Path

import pytest

from sift.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    for name in ("user", "work"):
        (tmp_path / name).mkdir()
    return Settings(home=tmp_path / "home", user=tmp_path / "user", cwd=tmp_path / "work", state=tmp_path / "state")


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    path = tmp_path / "notes"
    path.mkdir()
    return path
