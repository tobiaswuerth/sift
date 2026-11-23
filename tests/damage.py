import contextlib
import sqlite3
from pathlib import Path
from typing import cast


def changed(store: Path, sql: str) -> None:
    with contextlib.closing(sqlite3.connect(store)) as database, database:
        database.execute(sql)


def damage_table(store: Path, table: str) -> None:
    with contextlib.closing(sqlite3.connect(store)) as database:
        found = cast(
            "tuple[int, int]",
            database.execute(
                "SELECT rootpage, (SELECT page_size FROM pragma_page_size) FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone(),
        )
    root, size = found
    raw = bytearray(store.read_bytes())
    start = (root - 1) * size
    raw[start : start + size] = b"\xff" * size
    store.write_bytes(bytes(raw))


def damage_pages(store: Path) -> None:
    store.write_bytes(b"\xff" * 4096 + store.read_bytes()[4096:])
