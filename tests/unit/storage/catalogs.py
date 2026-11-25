import sqlite3
from pathlib import Path

from sift.storage.catalog import Collection, collections
from sift.storage.database import reading


def folder(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def known(catalog: sqlite3.Connection) -> tuple[Collection, ...]:
    with reading(catalog) as transaction:
        return collections(transaction)
