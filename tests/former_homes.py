import contextlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from sift.api.collections import AddCollection, TitleFrom
from sift.api.embedding import EmbedCollections
from sift.api.ids import DocumentId
from sift.app.collections import add
from sift.app.embedding import embed
from sift.storage.catalog import Collection, known_collections
from sift.storage.database import reading
from sift.storage.store import connected
from sift.storage.store_files import FORMER_CATALOG
from sift.storage.texts import text_of
from tests.homes import indexed, quiet
from tests.models import Hashed

COMMIT = "0123456789abcdef0123456789abcdef01234567"
PINNED_FAKE = f"tests/hashed-words@{COMMIT}"
FORMER_CATALOG_LAYOUT = (
    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)",
    (
        "CREATE TABLE collections(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, path TEXT NOT NULL, "
        "fields TEXT, store TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)"
    ),
)
FORMER_STORE_LAYOUT = (
    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)",
    (
        "CREATE TABLE docs(id INTEGER PRIMARY KEY AUTOINCREMENT, source INTEGER NOT NULL, path TEXT UNIQUE NOT NULL, "
        "hash TEXT, title TEXT, meta TEXT, text TEXT NOT NULL)"
    ),
    (
        "CREATE TABLE chunks(id INTEGER PRIMARY KEY AUTOINCREMENT, doc INTEGER NOT NULL, seq INTEGER, "
        "line_start INTEGER, line_end INTEGER, pos INTEGER NOT NULL, len INTEGER NOT NULL)"
    ),
    "CREATE TABLE vectors(chunk INTEGER PRIMARY KEY, vec BLOB NOT NULL)",
    "CREATE TABLE codes(chunk INTEGER PRIMARY KEY, bits BLOB NOT NULL)",
    (
        "CREATE TRIGGER codes_go_with_vectors AFTER DELETE ON vectors "
        "BEGIN DELETE FROM codes WHERE chunk = OLD.chunk; END"
    ),
    "CREATE TABLE counts(name TEXT PRIMARY KEY, n INTEGER NOT NULL)",
)
COUNTED = (
    "INSERT INTO codes(chunk, bits) SELECT chunk, zeroblob(8) FROM vectors",
    "INSERT INTO counts(name, n) VALUES ('docs', (SELECT count(*) FROM docs))",
    "INSERT INTO counts(name, n) VALUES ('chunks', (SELECT count(*) FROM chunks))",
    "INSERT INTO counts(name, n) VALUES ('vectors', (SELECT count(*) FROM vectors))",
)
DOCUMENTS = "SELECT d.id, s.path || '#' || d.key, d.title FROM documents d JOIN sources s ON s.id = d.source"
CHUNKS = "SELECT id, document, start, length FROM chunks ORDER BY document, id"
VECTORS = "SELECT chunk, vector FROM vectors"


def hashed() -> Hashed:
    return Hashed(name=PINNED_FAKE)


def _texts(one: Collection, documents: list[tuple[int, str, str]]) -> dict[int, str]:
    with connected(one.store, one.id) as connection, reading(connection) as transaction:
        return {each[0]: text_of(transaction, DocumentId(each[0])) for each in documents}


def _made(one: Collection, store: Path, model: str | None) -> None:
    with contextlib.closing(sqlite3.connect(one.store)) as source, contextlib.closing(sqlite3.connect(store)) as target:
        for statement in FORMER_STORE_LAYOUT:
            target.execute(statement)
        documents = cast("list[tuple[int, str, str]]", source.execute(DOCUMENTS).fetchall())
        texts = _texts(one, documents)
        target.executemany(
            "INSERT INTO docs(id, source, path, title, text) VALUES(?, 1, ?, ?, ?)",
            [(*each, texts[each[0]]) for each in documents],
        )
        seen = dict[int, int]()
        for chunk, document, start, length in cast(
            "list[tuple[int, int, int, int]]", source.execute(CHUNKS).fetchall()
        ):
            seen[document] = seen.get(document, -1) + 1
            target.execute(
                "INSERT INTO chunks(id, doc, seq, line_start, line_end, pos, len) VALUES(?, ?, ?, 1, 1, ?, ?)",
                (chunk, document, seen[document], start, length),
            )
        target.executemany("INSERT INTO vectors(chunk, vec) VALUES(?, ?)", source.execute(VECTORS).fetchall())
        for statement in COUNTED:
            target.execute(statement)
        if model is not None:
            target.executemany(
                "INSERT INTO meta(key, value) VALUES(?, ?)", [("model", model.split("@")[0]), ("model_at", model)]
            )
        target.commit()


def former_home(
    tmp_path: Path,
    folders: Mapping[str, Path],
    model: Hashed | None = None,
    choices: Mapping[str, Mapping[str, object]] | None = None,
    inactive: tuple[str, ...] = (),
) -> Path:
    made, former = tmp_path / "made", tmp_path / "former"
    for name, folder in folders.items():
        chosen = (choices or {}).get(name, {}).get("title_from")
        add(made, AddCollection(str(folder), name, title_from=TitleFrom(str(chosen)) if chosen else TitleFrom.HEADING))
    indexed(made)
    if model is not None:
        embed(made, EmbedCollections(), quiet, model)
    (former / "collections").mkdir(parents=True)
    with contextlib.closing(sqlite3.connect(former / FORMER_CATALOG)) as catalog:
        for statement in FORMER_CATALOG_LAYOUT:
            catalog.execute(statement)
        for one in known_collections(made):
            store = former / "collections" / f"{one.name}.sqlite"
            _made(one, store, None if model is None else model.name)
            fields = json.dumps(dict((choices or {}).get(one.name, {}))) if choices and one.name in choices else None
            catalog.execute(
                "INSERT INTO collections(name, path, fields, store, active) VALUES(?, ?, ?, ?, ?)",
                (one.name, str(one.folder), fields, str(store), int(one.name not in inactive)),
            )
        catalog.commit()
    return former
