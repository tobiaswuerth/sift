import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from sift.storage.catalog import open_catalog


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def catalog(home: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(open_catalog(home)) as connection:
        yield connection
