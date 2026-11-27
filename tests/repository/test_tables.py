import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

from tests.repository.texts import matches

ROOT = Path(__file__).parents[2]
KEYED_ALONE = r"CREATE (?:TEMP )?TABLE (?:IF NOT EXISTS )?\w+\([^\"]*?\)[^\"()]*?WITHOUT ROWID"


def declared() -> list[str]:
    sources = sorted((ROOT / "src" / "sift").rglob("*.py"))
    return [each for path in sources for each in matches(KEYED_ALONE, path.read_text(encoding="utf-8"))]


def blobs_beside_the_key(statement: str) -> list[str]:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(statement.replace("CREATE TEMP TABLE", "CREATE TABLE"))
        (name,) = cast("tuple[str]", connection.execute("SELECT name FROM sqlite_schema").fetchone())
        described = connection.execute("SELECT name, type, pk FROM pragma_table_info(?)", (name,)).fetchall()
        columns = cast("list[tuple[str, str, int]]", described)
    return [f"{name}.{column}" for column, kind, keyed in columns if kind == "BLOB" and not keyed]


def test_the_tables_sift_declares_without_row_ids_are_found() -> None:
    names = {re.findall(r"TABLE (?:IF NOT EXISTS )?(\w+)", each)[0] for each in declared()}
    assert {"meta", "keyword_terms", "links"} <= names


def test_no_table_searched_by_its_key_alone_holds_a_blob_beside_the_key() -> None:
    assert [found for each in declared() for found in blobs_beside_the_key(each)] == []
