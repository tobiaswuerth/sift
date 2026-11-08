"""The catalog: the collections sift knows, each with its folder, its choices of fields, where its store (see `store`)
lies, and whether it is active. The catalog is sift.db in SIFT_HOME, beside the logs and the worker's key; the stores
lie in its folder `collections`, unless a collection is added or moved elsewhere (`sift add --store`, `sift move`).

An inactive collection stays as it is, but searches pass it by without opening its store, and so do `sift index` and
`sift embed` unless they name it.
"""

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import NamedTuple

from . import console, readers, settings, store

SCHEMA_VERSION = 6  # in SQLite's user_version; up to 5, sift.db was the whole index (see `upgrade`)
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS collections(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL, fields TEXT, store TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
"""
COLUMNS = "id, name, path, fields, store, active"
SUFFIX = ".sqlite"  # a store's file: a path ending in it names one, another a folder to put one in
COPY_PAGES = 16384  # pages per step of copying a store to another drive: 64 MB of 4 KB pages
MB = 1 << 20


class Collection(NamedTuple):
    id: int  # never handed out again: its documents' and chunks' ids start at store.first_id(id)
    name: str
    path: str  # its folder
    fields: str | None  # its choices, as stored: see _fields
    store: str  # its store's file
    active: bool

    @property
    def choices(self):
        """Its choices of fields, and where titles come from, as a dict."""
        return json.loads(self.fields) if self.fields else {}


def _collection(row):
    return Collection(*row[:5], bool(row[5]))


class Catalog:
    """The catalog in the folder `home` (default: SIFT_HOME), and the stores opened through it, each once. An index of
    sift 1.7 or older there is split into stores first, in the folder `split_into` (see `upgrade`)."""

    def __init__(self, home=None, split_into=None):
        self.home = Path(home or settings.current().home)
        self.path = self.home / "sift.db"
        self.db = _open(self.path, Path(split_into) if split_into else self.home / "collections")
        self._stores = {}  # collection id: (its store's file, the store opened)

    def close(self):
        for cid in list(self._stores):
            self.release(cid)
        self.db.close()

    def collections(self):
        """Every collection, by name."""
        return [_collection(row) for row in self.db.execute(f"SELECT {COLUMNS} FROM collections ORDER BY name")]

    def get(self, name):
        """The collection called `name`; an error naming those there are if there is none."""
        row = self.db.execute(f"SELECT {COLUMNS} FROM collections WHERE name = ?", (name,)).fetchone()
        if row is None:
            names = [c.name for c in self.collections()]
            there = f"the collections are {', '.join(names)}" if names else "none is registered yet: `sift add FOLDER`"
            raise LookupError(f"no collection named {name!r}; {there}")
        return _collection(row)

    def store(self, collection):
        """The store of a collection, or of the collection of that name, opened once, and anew where the collection
        has moved meanwhile. A store missing, damaged or of a newer sift is a LookupError, which commands of every
        collection pass over."""
        collection = self.get(collection) if isinstance(collection, str) else collection
        held = self._stores.get(collection.id)
        if held and held[0] == collection.store:
            return held[1]
        self.release(collection.id)
        try:
            db = store.connect(collection.store, collection.id)
        except FileNotFoundError:
            raise LookupError(
                f"{collection.name}: its index is missing: {collection.store}; is its drive connected? "
                f"`{console.command('sift', 'rm', collection.name)}` forgets the collection"
            ) from None
        except (sqlite3.DatabaseError, SystemExit) as error:  # damaged, or a newer sift's: passed over as missing
            if isinstance(error, sqlite3.OperationalError):  # locked, the disk full: not the store's own fault
                raise
            reason = error.code if isinstance(error, SystemExit) else f"{collection.store}: {error}"
            raise LookupError(f"{collection.name}: its index can't be opened: {reason}") from None
        self._stores[collection.id] = (collection.store, db)
        return db

    def release(self, cid):
        """Close the store of the collection `cid`, if it is open."""
        if held := self._stores.pop(cid, None):
            held[1].close()

    def release_gone(self):
        """Close the stores of collections no longer in the catalog, as another sift removed them."""
        there = {c.id for c in self.collections()}
        for cid in self._stores.keys() - there:
            self.release(cid)

    # --- changing the catalog --------------------------------------------------

    def add(self, path, name=None, fields=None, nested=False, into=None):
        """Register a folder, its store made at `into`: a file ending in .sqlite, or a folder to make NAME.sqlite in
        (default: `collections` in the catalog's folder). `fields` chooses JSON Lines fields instead of the defaults:
        {"id", "title", "text"}. A folder inside a collection, or one holding a collection, only if `nested`: the
        files of both would be indexed twice."""
        stored = _fields(fields)
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"not a directory: {path}")
        if other := self.collection_at(path):  # registered twice, its documents were found twice
            raise SystemExit(f"{path} is the collection {other!r} already")
        if not nested and (overlap := self._overlapping(path)):
            other, folder, inside = overlap
            where = "is inside" if inside else "holds"
            raise SystemExit(
                f"{path} {where} the collection {other!r} ({folder}): its files would be indexed twice, once in each "
                "collection, and found twice; add it anyway with --nested"
            )
        name = name or path.name
        if not re.fullmatch(r"[\w.-]+", name):
            raise SystemExit(
                f"invalid collection name {name!r}: choose one with --name, of letters, digits, '_', '-', '.'"
            )
        file = _target(into or self.home / "collections", name + SUFFIX)
        made = None
        self.db.execute("BEGIN IMMEDIATE")
        try:
            try:
                cid = self.db.execute(
                    "INSERT INTO collections(name, path, fields, store) VALUES(?, ?, ?, ?)",
                    (name, str(path), stored, str(file)),
                ).lastrowid
            except sqlite3.IntegrityError:
                raise SystemExit(f"collection {name!r} already exists") from None
            taken = self.db.execute("SELECT count(*) FROM collections WHERE store = ?", (str(file),)).fetchone()[0] > 1
            if taken or file.exists():  # a store left, of a collection whose drive is gone, or another sift's
                raise SystemExit(f"{file} exists already: move it away, or choose another place with --store")
            made = file
            store.connect(file, cid, create=True).close()
            self.db.commit()
        except BaseException:
            self.db.rollback()
            if made:
                _delete(made)
            raise
        return name

    def remove(self, name):
        """Remove a collection: its store is deleted, its files stay on disk. Returns the number of documents its
        store held, or None if it was missing."""
        collection = self.get(name)
        try:
            db = self.store(collection)
            docs = store.counts(db)["docs"]
        except LookupError as unusable:  # missing: only the catalog changes; damaged or a newer sift's: it goes too
            console.say(f"{unusable}{'' if os.path.isfile(collection.store) else ': only the catalog changes'}")
            docs = None
        self.release(collection.id)
        self.db.execute("DELETE FROM collections WHERE id = ?", (collection.id,))
        self.db.commit()
        if left := _delete(Path(collection.store)):
            console.say(
                f"{name}: {', '.join(map(str, left))} could not be deleted, as another sift has it open: "
                "delete it once that has stopped"
            )
        return docs

    def set_active(self, names, active):
        """Make the collections named active, or inactive. Returns the names of those that changed."""
        chosen = [self.get(name) for name in dict.fromkeys(names)]  # every name checked before anything changes
        self.db.executemany("UPDATE collections SET active = ? WHERE id = ?", [(int(active), c.id) for c in chosen])
        self.db.commit()
        if not active:
            for c in chosen:
                self.release(c.id)
        return [c.name for c in chosen if c.active != active]

    def set_title_from(self, name, title_from):
        """Choose where a collection's Markdown and text files take their titles from. Returns the collection as it
        is now."""
        collection = self.get(name)
        fields = {key: value for key, value in collection.choices.items() if key != "title_from"}
        stored = _fields(fields | {"title_from": title_from})
        self.db.execute("UPDATE collections SET fields = ? WHERE id = ?", (stored, collection.id))
        self.db.commit()
        return collection._replace(fields=stored)

    def move(self, names, to):
        """Move the stores of the collections named, or of all, to `to`: a file ending in .sqlite, for one collection,
        or a folder, where each keeps its file's name. On the same drive the file gets a second name, at once, else it
        is copied; then the catalog points at it, and the old file is deleted. No other sift may have a store open
        meanwhile. Returns the names of the collections moved."""
        chosen = [self.get(name) for name in dict.fromkeys(names)] if names else self.collections()
        if str(to).lower().endswith(SUFFIX) and len(chosen) != 1:
            raise SystemExit(f"{to} is the file of one collection's index: name one collection, or give a folder")
        return [c.name for c in chosen if self._move(c, _target(to, Path(c.store).name))]

    def _move(self, collection, new):
        old = Path(collection.store)
        if os.path.normcase(old.resolve()) == os.path.normcase(new):
            console.say(f"{collection.name}: its index is {new} already")
            return False
        if not old.is_file():
            self.store(collection)  # raises: missing
        if new.exists():
            raise SystemExit(f"{collection.name}: {new} exists already; move it away, or move the collection elsewhere")
        new.parent.mkdir(parents=True, exist_ok=True)
        self.release(collection.id)
        with store.alone(old, collection.name) as held:
            try:
                _transfer(held, old, new, collection.name)
                self.db.execute("UPDATE collections SET store = ? WHERE id = ?", (str(new), collection.id))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                _delete(new)
                raise
        if left := _delete(old):
            console.say(f"{collection.name}: moved, but {', '.join(map(str, left))} could not be deleted: delete it")
        return True

    # --- folders -----------------------------------------------------------------

    def collection_at(self, path):
        """The name of the collection of the folder `path`, or None."""
        path = os.path.normcase(Path(path).expanduser().resolve())
        return next((c.name for c in self.collections() if os.path.normcase(c.path) == path), None)

    def _overlapping(self, path):
        """A collection whose folder holds the folder `path`, or lies inside it: (its name, its folder, whether `path`
        is inside it); None if there is none."""
        mine = Path(os.path.normcase(path))
        for c in self.collections():
            theirs = Path(os.path.normcase(c.path))
            if mine.is_relative_to(theirs) or theirs.is_relative_to(mine):
                return c.name, c.path, mine.is_relative_to(theirs)
        return None


def _open(path, split_into):
    """The catalog at `path`: made if there is none, and split off an index of sift 1.7 or older there first."""
    db = store.database(path, create=not path.exists())
    try:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise SystemExit(f"{path} was made by a newer sift; upgrade sift to use it")
        tables = db.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
        if version < 2 and tables:
            raise SystemExit(
                f"this index was made by an older sift, which stored documents differently: delete {path}, "
                "then run `sift add`, `sift index` and `sift embed` again"
            )
        if not version:
            db.executescript(SCHEMA + f"PRAGMA user_version = {SCHEMA_VERSION};")
            return db
    except BaseException:
        db.close()
        raise
    if version == SCHEMA_VERSION:
        return db
    db.close()
    from . import upgrade

    upgrade.split(path, split_into)
    return _open(path, split_into)


def _fields(fields):
    """A collection's choices, as stored: JSON Lines fields, {"id": name, "title": name, "text": [names]}, and where
    its Markdown and text files take their titles from, {"title_from": "heading" or "name"}."""
    if not fields:
        return None
    for key, value in fields.items():
        if key == "title_from":
            if value not in readers.TITLE_FROM:
                raise SystemExit(f"titles come from 'heading' or 'name', got {value!r}")
            continue
        if key not in ("id", "title", "text"):
            raise SystemExit(f"unknown field choice {key!r}: choose the id, title or text field")
        names = value if key == "text" else [value]
        if not (isinstance(names, list) and names and all(isinstance(n, str) and n for n in names)):
            raise SystemExit(
                f"the {key} field must be {'one or more field names' if key == 'text' else 'a field name'}, "
                f"got {value!r}"
            )
    return json.dumps(fields)


def _target(to, name):
    """The file a store gets at `to`: `to` itself, if it ends in .sqlite, else the file `name` in the folder `to`."""
    to = Path(to).expanduser().resolve()
    return to if to.name.lower().endswith(SUFFIX) else to / name


def _transfer(held, old, new, name):
    """Give the store `held` at `old` the path `new`: a second name for the file, or on another drive, a copy, made
    through SQLite, as Windows keeps other readers out of the bytes SQLite locks."""
    try:
        os.link(old, new)
        return
    except OSError:  # another drive, or a file system without hard links
        pass
    part = new.with_name(new.name + ".part")
    part.unlink(missing_ok=True)
    pages = held.execute("PRAGMA page_count").fetchone()[0]
    page = held.execute("PRAGMA page_size").fetchone()[0]
    copy = sqlite3.connect(part)
    try:
        with console.Progress(-(-pages * page // MB), "MB", prefix=f"{name}: copying ") as progress:
            held.backup(
                copy, pages=COPY_PAGES, progress=lambda _, left, total: progress.update((total - left) * page // MB)
            )
            progress.update(progress.total)
        copy.close()
        os.replace(part, new)
    except BaseException:
        copy.close()
        part.unlink(missing_ok=True)
        raise


def _delete(path):
    """Delete a store's files; returns those that could not be, as another sift has them open (Windows)."""
    left = []
    for file in store.files(path):
        try:
            file.unlink(missing_ok=True)
        except OSError:
            left.append(file)
    return left
