"""Splitting an index of sift 1.7 or older, one database for every collection, into a store per collection and the
catalog: nothing lost, nothing read or embedded again, and resumed where it stopped. The old indexes are made from
indexes of this sift, as each older layout held them."""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import sift
from sift import catalog as catalog_module
from sift import cli, embed, indexing, readers, store, upgrade
from sift.catalog import Catalog
from sift.search import Index

from .conftest import keywords_agree, write

# The layout of sift 1.1 to 1.7, of one database holding every collection.
LAYOUT_5 = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE collections(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, path TEXT NOT NULL, fields TEXT);
CREATE TABLE sources(id INTEGER PRIMARY KEY, collection INTEGER NOT NULL, path TEXT NOT NULL, mtime INTEGER,
    size INTEGER, UNIQUE(collection, path));
CREATE TABLE docs(id INTEGER PRIMARY KEY, collection INTEGER NOT NULL, source INTEGER NOT NULL, path TEXT NOT NULL,
    hash TEXT, title TEXT, meta TEXT, text TEXT NOT NULL, UNIQUE(collection, path));
CREATE INDEX docs_source ON docs(source);
CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc INTEGER NOT NULL, seq INTEGER, line_start INTEGER, line_end INTEGER,
    pos INTEGER NOT NULL, len INTEGER NOT NULL);
CREATE INDEX chunks_doc ON chunks(doc);
CREATE TABLE vectors(chunk INTEGER PRIMARY KEY, vec BLOB NOT NULL);
CREATE VIEW chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text,
    (SELECT group_concat(j.atom, ' ') FROM json_tree(d.meta) j WHERE j.atom IS NOT NULL AND j.fullkey != '$.title')
    AS meta FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE fts USING fts5(title, text, meta, content='chunk_text', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
CREATE TABLE links(id INTEGER PRIMARY KEY, doc INTEGER NOT NULL, collection INTEGER NOT NULL, field TEXT,
    line INTEGER NOT NULL, kind TEXT NOT NULL, target TEXT NOT NULL, key TEXT NOT NULL, fragment TEXT, shown TEXT);
CREATE INDEX links_doc ON links(doc);
CREATE INDEX links_key ON links(collection, key);
CREATE TABLE names(doc INTEGER NOT NULL, collection INTEGER NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL);
CREATE INDEX names_doc ON names(doc);
CREATE INDEX names_key ON names(collection, key);
CREATE TABLE codes(chunk INTEGER PRIMARY KEY, bits BLOB NOT NULL);
CREATE TRIGGER codes_go_with_vectors AFTER DELETE ON vectors BEGIN DELETE FROM codes WHERE chunk = OLD.chunk; END;
"""
# What sift 1.0 lacked (layout 3), and what 1.1 to 1.6 (layout 4)
WITHOUT_CODES = "DROP TRIGGER codes_go_with_vectors; DROP TABLE codes; PRAGMA user_version = 4;"
WITHOUT_LINKS = "DROP TABLE links; DROP TABLE names; PRAGMA user_version = 3;"
# The keyword index of sift 0.7 and older (layout 2): title and text, no front matter.
KEYWORD_INDEX_0_7 = """DROP TABLE fts; DROP VIEW chunk_text;
CREATE VIEW chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text
    FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE fts USING fts5(title, text, content='chunk_text', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
INSERT INTO fts(fts) VALUES('rebuild');
PRAGMA user_version = 2;"""


def interleaved(groups):
    """The rows of several collections, (collection id, row) each, one of each in turn, as an index filled by turns
    holds them: their ids interleave."""
    order = [(i, cid, row) for cid, rows in groups.items() for i, row in enumerate(rows)]
    return [(cid, row) for _, cid, row in sorted(order, key=lambda o: (o[0], o[1]))]


def as_made_by_sift_1_7(made, path, layout=5):
    """The index of sift 1.7 or older at `path`, in `layout`, of what the catalog `made` holds."""
    old = sqlite3.connect(path)
    old.executescript(LAYOUT_5)
    collections = made.collections()
    old.executemany(
        "INSERT INTO collections(id, name, path, fields) VALUES(?, ?, ?, ?)",
        [(c.id, c.name, c.path, c.fields) for c in collections],
    )
    stores = {c.id: made.store(c) for c in collections}
    sources, docs, chunks = {}, {}, {}

    def table(name, select):
        return interleaved({cid: db.execute(select).fetchall() for cid, db in stores.items()})

    for cid, (sid, *row) in table("sources", "SELECT id, path, mtime, size FROM sources ORDER BY id"):
        sources[cid, sid] = old.execute(
            "INSERT INTO sources(collection, path, mtime, size) VALUES(?, ?, ?, ?)", (cid, *row)
        ).lastrowid
    for cid, (did, sid, *row) in table(
        "docs", "SELECT id, source, path, hash, title, meta, text FROM docs ORDER BY id"
    ):
        docs[did] = old.execute(
            "INSERT INTO docs(collection, source, path, hash, title, meta, text) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (cid, sources[cid, sid], *row),
        ).lastrowid
    select = "SELECT id, doc, seq, line_start, line_end, pos, len FROM chunks ORDER BY id"
    for _, (chunk, did, *row) in table("chunks", select):
        chunks[chunk] = old.execute(
            "INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?, ?, ?, ?, ?, ?)", (docs[did], *row)
        ).lastrowid
    for name in ("vectors", "codes"):
        rows = table(name, f"SELECT * FROM {name} ORDER BY chunk")
        old.executemany(
            f"INSERT INTO {name} VALUES(?, ?)", sorted((chunks[chunk], value) for _, (chunk, value) in rows)
        )
    select = "SELECT doc, field, line, kind, target, key, fragment, shown FROM links ORDER BY id"
    old.executemany(
        "INSERT INTO links(doc, collection, field, line, kind, target, key, fragment, shown) VALUES(?,?,?,?,?,?,?,?,?)",
        [(docs[row[0]], cid, *row[1:]) for cid, row in table("links", select)],
    )
    old.executemany(
        "INSERT INTO names(doc, collection, key, kind) VALUES(?, ?, ?, ?)",
        [(docs[row[0]], cid, *row[1:]) for cid, row in table("names", "SELECT doc, key, kind FROM names")],
    )
    meta = {key: value for db in stores.values() for key, value in db.execute("SELECT key, value FROM meta")}
    old.executemany(
        "INSERT INTO meta VALUES(?, ?)",
        [(key, meta[key]) for key in upgrade.CARRIED if key in meta] + [("version", "1")],
    )
    old.execute("INSERT INTO fts(rowid, title, text, meta) SELECT id, title, text, meta FROM chunk_text")
    old.execute("PRAGMA user_version = 5")
    old.commit()
    for below, script in ((5, WITHOUT_CODES), (4, WITHOUT_LINKS), (3, KEYWORD_INDEX_0_7)):
        if layout < below:
            old.executescript(script)
    old.execute("PRAGMA journal_mode=WAL")  # as sift kept it
    old.close()
    made.close()


def files(folder):
    """The files in a folder, but SQLite's logs of the databases open."""
    return sorted(p.name for p in folder.iterdir() if not p.name.endswith(("-wal", "-shm")))


def contents(db):
    """Everything a store holds, independent of row ids."""
    doc = "(SELECT path FROM docs WHERE id = {})"
    chunk = "(SELECT d.path || '#' || c.seq FROM chunks c JOIN docs d ON d.id = c.doc WHERE c.id = {})"
    queries = {
        "sources": "SELECT path, mtime, size FROM sources",
        "docs": "SELECT path, hash, title, meta, text FROM docs",
        "chunks": f"SELECT {doc.format('doc')}, seq, line_start, line_end, pos, len FROM chunks",
        "vectors": f"SELECT {chunk.format('chunk')}, vec FROM vectors",
        "codes": f"SELECT {chunk.format('chunk')}, bits FROM codes",
        "links": f"SELECT {doc.format('doc')}, field, line, kind, target, key, fragment, shown FROM links",
        "names": f"SELECT {doc.format('doc')}, key, kind FROM names",
        "meta": "SELECT key, value FROM meta WHERE key NOT IN ('version', 'collection')",
    }
    return {table: sorted(db.execute(sql).fetchall()) for table, sql in queries.items()}


@pytest.fixture
def made(vault, tmp_path, fake_models):
    """A catalog of this sift, of the vault and the work folder, indexed and embedded: what an old index is made of."""
    catalog = Catalog(tmp_path / "made")
    catalog.add(vault[0], "vault", {"title_from": "name"})
    catalog.add(vault[1], "work")
    indexing.index(catalog)
    embed.embed_pending(catalog)
    return catalog


def found(catalog):
    """What searches of every kind find in a catalog."""
    index = Index(catalog)
    hits = [index.search("Acme"), index.vsearch("maintenance contract"), index.query("who manages Acme")]
    hans = "vault/01_People/Hans Müller.md"
    return [[(h["id"], h["lines"]) for h in hit] for hit in hits] + [
        index.links(hans, "both"),
        index.unresolved(),
        index.browse(sort="-org"),
        index.get("work/contract.md"),
    ]


@pytest.fixture
def old(made, home):
    """An index of sift 1.7 in the home folder, of the catalog `made`: (what it holds, what searches find)."""
    held = {c.name: contents(made.store(c)) for c in made.collections()}
    searched = found(made)
    home.mkdir()
    as_made_by_sift_1_7(made, home / "sift.db")
    return held, searched


def test_an_index_of_sift_1_7_splits_into_a_store_per_collection(old, home, fake_models, monkeypatch, capsys):
    held, searched = old
    load, encode = readers.load, fake_models.encode
    monkeypatch.setattr(readers, "load", lambda *args: pytest.fail("a file was read again"))
    monkeypatch.setattr(fake_models, "encode", lambda *args, **kwargs: pytest.fail("something was embedded again"))
    catalog = Catalog(home)
    err = capsys.readouterr().err
    assert "splitting" in err
    assert "keyword index of 2 collections" in err
    assert catalog.db.execute("PRAGMA user_version").fetchone() == (catalog_module.SCHEMA_VERSION,)
    assert [(c.name, c.store) for c in catalog.collections()] == [
        (name, str(home / "collections" / f"{name}.sqlite")) for name in ("vault", "work")
    ]
    for c in catalog.collections():
        db = catalog.store(c)
        assert contents(db) == held[c.name], c.name
        assert keywords_agree(db)
        assert all(store.owner(i) == c.id for (i,) in db.execute("SELECT id FROM docs UNION SELECT id FROM chunks"))
    monkeypatch.setattr(readers, "load", load)  # not undo(): that would also undo the fake models and the home
    monkeypatch.setattr(fake_models, "encode", encode)
    assert found(catalog) == searched
    assert files(home) == ["collections", "sift.db"]  # the old index gone, nothing left of the split
    assert files(home / "collections") == ["vault.sqlite", "work.sqlite"]


def test_a_split_index_goes_on_as_any(old, home, vault, fake_models, capsys):
    catalog = Catalog(home)
    capsys.readouterr()
    write(vault[0] / "New note.md", "## Summary\n\nA note after the split, linking [[Hans Müller]].\n")
    indexing.index(catalog)
    assert "1 added, 0 changed, 0 removed" in capsys.readouterr().err  # nothing else to read again
    embed.embed_pending(catalog)
    new = catalog.store("vault").execute("SELECT id FROM docs WHERE path = 'New note.md'").fetchone()[0]
    assert store.owner(new) == catalog.get("vault").id
    index = Index(catalog)
    assert [h["id"] for h in index.vsearch("a note after the split", k=1)] == ["vault/New note.md"]
    assert "vault/New note.md" in [d["id"] for d in index.links("vault/01_People/Hans Müller.md", "in")["in"]]
    Catalog(home)
    assert "splitting" not in capsys.readouterr().err  # once


def test_a_split_goes_into_the_folder_chosen(old, home, tmp_path):
    held, _ = old
    catalog = Catalog(home, split_into=tmp_path / "big drive")
    assert catalog.get("work").store == str(tmp_path / "big drive" / "work.sqlite")
    assert contents(catalog.store("work")) == held["work"]
    assert not (home / "collections").exists()


def test_sift_move_splits_an_old_index_straight_into_its_folder(old, home, tmp_path, capsys):
    # A large index takes as much space again while it is split: into the folder it moves to, not beside the old one.
    cli.main(["move", str(tmp_path / "big drive")])
    catalog = Catalog(home)
    assert {c.store for c in catalog.collections()} == {
        str(tmp_path / "big drive" / f"{name}.sqlite") for name in ("vault", "work")
    }
    assert not (home / "collections").exists()
    assert "nothing moved" in capsys.readouterr().err  # they are there already


def test_a_split_stopped_goes_on_where_it_stopped(old, home, monkeypatch):
    held, _ = old
    finished, real = [], upgrade._finish

    def stopped(part, file, cid, name, *args):
        if finished:
            raise KeyboardInterrupt
        finished.append(name)
        return real(part, file, cid, name, *args)

    monkeypatch.setattr(upgrade, "_finish", stopped)
    with pytest.raises(KeyboardInterrupt):
        Catalog(home)
    old_index = sqlite3.connect(home / "sift.db")
    assert old_index.execute("PRAGMA user_version").fetchone() == (5,)  # the old index as it was
    old_index.close()
    monkeypatch.setattr(
        upgrade,
        "_finish",
        lambda part, file, cid, name, *args: finished.append(name) or real(part, file, cid, name, *args),
    )
    catalog = Catalog(home)
    assert finished == ["vault", "work"]  # the first not made again
    assert {c.name: contents(catalog.store(c)) for c in catalog.collections()} == held


def test_a_split_stopped_while_copying_changes_nothing(old, home, monkeypatch):
    held, _ = old
    original = upgrade._copy

    def stopped(held_db, parts, layout):
        for part in parts.values():
            part.execute("INSERT INTO meta VALUES('half', 'way')")
        raise KeyboardInterrupt

    monkeypatch.setattr(upgrade, "_copy", stopped)
    with pytest.raises(KeyboardInterrupt):
        Catalog(home)
    assert files(home / "collections") == ["vault.sqlite.part", "work.sqlite.part"]
    monkeypatch.setattr(upgrade, "_copy", original)
    catalog = Catalog(home)
    assert {c.name: contents(catalog.store(c)) for c in catalog.collections()} == held
    assert files(home / "collections") == ["vault.sqlite", "work.sqlite"]


def test_the_stores_can_be_finished_in_processes_of_their_own(old, home, monkeypatch):
    # The keyword index of a collection of millions of passages takes a while: the collections' are made side by side.
    held, _ = old
    monkeypatch.setattr(upgrade, "PARALLEL_CHUNKS", 0)
    code = str(Path(sift.__file__).parent.parent)  # this sift, not one installed elsewhere, in the processes
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [code, os.environ.get("PYTHONPATH")])))
    catalog = Catalog(home)
    assert {c.name: contents(catalog.store(c)) for c in catalog.collections()} == held


@pytest.mark.parametrize("layout, what", [(4, "codes"), (3, "links")])
def test_an_older_index_gets_what_its_layout_lacks(made, home, layout, what, fake_models, monkeypatch, capsys):
    # sift 1.1 to 1.6 kept no codes, 1.0 no links: they come from what the index holds, nothing read or embedded.
    held = {c.name: contents(made.store(c)) for c in made.collections()}
    home.mkdir()
    as_made_by_sift_1_7(made, home / "sift.db", layout)
    monkeypatch.setattr(readers, "load", lambda *args: pytest.fail("a file was read again"))
    monkeypatch.setattr(fake_models, "encode", lambda *args, **kwargs: pytest.fail("something was embedded again"))
    catalog = Catalog(home)
    assert ("adding the codes" if what == "codes" else "links:") in capsys.readouterr().err
    assert {c.name: contents(catalog.store(c)) for c in catalog.collections()} == held


def test_an_index_of_sift_1_0_reads_its_front_matter_again(made, home, capsys):
    # sift 1.0 read lists of unquoted items as text, and gave a note of front matter alone no passage.
    vault = made.store("vault")
    (doc,) = vault.execute("SELECT id FROM docs WHERE path = '01_People/Hans Müller.md'").fetchone()
    store.keywords(vault, [doc], add=False)
    meta = json.loads(vault.execute("SELECT meta FROM docs WHERE id = ?", (doc,)).fetchone()[0])
    meta["aliases"] = "[Hans Mueller, H. Müller]"
    vault.execute("UPDATE docs SET meta = ? WHERE id = ?", (json.dumps(meta, ensure_ascii=False), doc))
    store.keywords(vault, [doc])
    vault.commit()
    home.mkdir()
    as_made_by_sift_1_7(made, home / "sift.db", 3)
    catalog = Catalog(home)
    index = Index(catalog)
    assert index.document("vault/01_People/Hans Müller.md", fields=["aliases"])["fields"] == {
        "aliases": ["Hans Mueller", "H. Müller"]
    }
    offer = index.links("vault/03_Interactions/2026-10-14 Offerte.md")["out"][0]
    assert (offer["target"], offer["id"], offer["via"]) == ("Hans Mueller", "vault/01_People/Hans Müller.md", "alias")
    assert keywords_agree(catalog.store("vault"))


def test_an_index_of_sift_0_7_gets_front_matter_into_its_keyword_index(notes, tmp_path, home, fake_models):
    made = Catalog(tmp_path / "made")
    made.add(notes, "notes")
    indexing.index(made)
    home.mkdir()
    as_made_by_sift_1_7(made, home / "sift.db", 2)
    assert [h["id"] for h in Index(Catalog(home)).search("cs.GR")] == ["notes/robots/brush.md"]


def test_an_index_without_collections_becomes_an_empty_catalog(tmp_path, home):
    made = Catalog(tmp_path / "made")
    home.mkdir()
    as_made_by_sift_1_7(made, home / "sift.db")
    assert Catalog(home).collections() == []


def test_chunks_and_vectors_of_documents_gone_are_passed_over(made, home):
    # Left by an older sift stopped half-way: they belong to no collection.
    held = {c.name: contents(made.store(c)) for c in made.collections()}
    home.mkdir()
    as_made_by_sift_1_7(made, home / "sift.db")
    old = sqlite3.connect(home / "sift.db")
    old.execute("INSERT INTO chunks(id, doc, seq, line_start, line_end, pos, len) VALUES(9999, 9999, 0, 1, 1, 0, 1)")
    old.execute("INSERT INTO vectors VALUES(9999, x'00')")
    old.commit()
    old.close()
    catalog = Catalog(home)
    assert {c.name: contents(catalog.store(c)) for c in catalog.collections()} == held


# --- what keeps a split from starting ------------------------------------------


def untouched(home):
    old = sqlite3.connect(home / "sift.db")
    try:
        return old.execute("PRAGMA user_version").fetchone() == (5,) and not (home / "collections").exists()
    finally:
        old.close()


def test_a_split_needs_room(old, home, monkeypatch):
    monkeypatch.setattr(upgrade.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 10})())
    with pytest.raises(SystemExit, match=r"takes up to .* GB in .*, which has 0\.0 GB free: .*sift move"):
        Catalog(home)
    assert untouched(home) or not any((home / "collections").iterdir())


def test_a_split_waits_for_every_other_sift_to_stop(old, home):
    other = sqlite3.connect(home / "sift.db")  # a sift 1.7 server, say
    other.execute("SELECT count(*) FROM docs").fetchone()
    with pytest.raises(SystemExit, match="another sift has its index open"):
        Catalog(home)
    other.close()
    assert untouched(home)
    assert len(Catalog(home).collections()) == 2  # once it has stopped


def test_a_file_where_a_store_goes_stops_the_split(old, home):
    (home / "collections").mkdir()
    write(home / "collections" / "work.sqlite", "mine")
    with pytest.raises(SystemExit, match=r"work\.sqlite is in the way of the split: move it away, then start sift"):
        Catalog(home)
    assert (home / "collections" / "work.sqlite").read_text() == "mine"
    old_index = sqlite3.connect(home / "sift.db")
    assert old_index.execute("PRAGMA user_version").fetchone() == (5,)
    old_index.close()


def test_an_index_of_sift_1_0_gets_the_names_of_other_files_and_passages_for_notes_of_front_matter_alone(
    vault, tmp_path, home, fake_models
):
    # sift 1.0 kept no names, and gave a note of front matter alone no passage.
    write(vault[0] / "page.html", "<html><body><h1>Price list</h1><p>Rates for 2027.</p></body></html>")
    write(vault[0] / "Prices.md", "See ![[page.html]].\n")
    write(vault[0] / "01_People" / "Peter Frei.md", '---\norg: "[[Acme AG]]"\nemail: peter@acme.example\n---\n')
    made = Catalog(tmp_path / "made")
    made.add(vault[0], "vault", {"title_from": "name"})
    indexing.index(made)
    db = made.store("vault")
    (doc,) = db.execute("SELECT id FROM docs WHERE path = '01_People/Peter Frei.md'").fetchone()
    store.keywords(db, [doc], add=False)
    db.execute("DELETE FROM chunks WHERE doc = ?", (doc,))
    db.commit()
    home.mkdir()
    as_made_by_sift_1_7(made, home / "sift.db", 3)
    index = Index(Catalog(home))
    assert "vault/01_People/Peter Frei.md" in ids(index.search("Peter Frei"))
    assert index.links("vault/Prices.md")["out"][0]["id"] == "vault/page.html"


def ids(hits):
    return [h["id"] for h in hits]


def test_a_store_that_does_not_hold_every_row_sent_stops_the_split(old, home, monkeypatch):
    real = upgrade._copy

    def miscounted(held, parts, layout):
        sent = real(held, parts, layout)
        sent[min(sent)]["docs"] += 1
        return sent

    monkeypatch.setattr(upgrade, "_copy", miscounted)
    with pytest.raises(SystemExit, match="vault: its new index holds 4 docs, not 5: the split stopped"):
        Catalog(home)
    assert untouched_but_parts(home)


def untouched_but_parts(home):
    """Whether the old index is as it was, the stores still being made."""
    old = sqlite3.connect(home / "sift.db")
    try:
        return old.execute("PRAGMA user_version").fetchone() == (5,) and not any(
            p.suffix == ".sqlite" for p in (home / "collections").iterdir()
        )
    finally:
        old.close()


def test_a_log_left_beside_the_old_index_stops_the_split_before_the_catalog_takes_its_place(old, home, monkeypatch):
    # SQLite would apply the old index's log to the catalog.
    real = store.alone

    @contextmanager
    def then_another_opens_it(path, name):
        with real(path, name) as held:
            yield held
        Path(f"{path}-wal").write_bytes(b"frames of another sift")

    monkeypatch.setattr(store, "alone", then_another_opens_it)
    with pytest.raises(SystemExit, match=r"another sift has .*sift\.db open: stop it, then start sift again"):
        Catalog(home)
    Path(f"{home / 'sift.db'}-wal").unlink()
    monkeypatch.setattr(store, "alone", real)
    assert len(Catalog(home).collections()) == 2  # the stores made already, the split ends


def test_an_old_index_another_sift_opens_meanwhile_stays_till_it_stops(old, home, monkeypatch):
    real = upgrade.os.replace

    def refused(source, target):
        if str(target).endswith("sift.db"):
            raise PermissionError(13, "The process cannot access the file")
        return real(source, target)

    monkeypatch.setattr(upgrade.os, "replace", refused)
    with pytest.raises(SystemExit, match=r"another sift has .*sift\.db open"):
        Catalog(home)
    monkeypatch.setattr(upgrade.os, "replace", real)
    assert len(Catalog(home).collections()) == 2


def test_the_stores_are_on_the_disk_before_the_old_index_goes(old, home, monkeypatch):
    # Written without a journal, a store is written through to the disk before it takes its name: a power cut after
    # the old index went must not take the stores with it.
    synced, real = [], upgrade.os.fsync

    def fsync(fd):
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(upgrade.os, "fsync", fsync)
    Catalog(home)
    assert len(synced) >= 2  # each store
