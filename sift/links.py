"""Links between Markdown notes, in the index: the links each note holds (see readers.links), stored as written, and the
names each file answers to, both resolved only when asked. So links stay right as notes come and go: a link to a note
that doesn't exist yet resolves once the note is indexed, and one to a note deleted or renamed no longer does, without
the linking note being read again.

A link resolves within its note's collection, as Obsidian resolves it:

- a relative path, `[text](../People/Hans.md)` or `[[./Hans]]`, the file at that path from the note's folder;
- a path, `[[01_People/Hans Müller]]`, the file whose path is it, else the one file whose path ends with it;
- a name, `[[Hans Müller]]`, the one file of that name, or of several, the one in the note's folder;
- and else the one note with that alias (Obsidian's `aliases` property), which Obsidian itself doesn't resolve.

Names compare case-insensitively and composed (readers.fold); a note's name leaves out `.md`, other files' keep their
extension. A link that matches several files is ambiguous, and one that matches none unresolved: a note not written yet.
"""

import json
import posixpath
from collections import Counter

from . import readers

AMBIGUOUS_SHOWN = 5  # candidates an ambiguous link names
UNRESOLVED = ("unresolved", [])


def names(path, meta):
    """The names a file answers to, as (key, kind): its file name, and its aliases (`aliases`, or the older `alias`;
    one or a list)."""
    found = {(readers.key(path.rsplit("/", 1)[-1]), "name")}
    for field in ("aliases", "alias"):
        value = meta.get(field)
        for alias in value if isinstance(value, list) else [value]:
            if isinstance(alias, str) and alias.strip():
                found.add((readers.key(alias.strip()), "alias"))
    return sorted(found)


def save(db, docs):
    """Store the links of documents, and the names of those that are whole files: (id, collection id, path, meta,
    links, whole file?) of each."""
    db.executemany(
        "INSERT INTO names(doc, collection, key, kind) VALUES(?, ?, ?, ?)",
        [(doc, cid, *name) for doc, cid, path, meta, _, whole in docs if whole for name in names(path, meta)],
    )
    db.executemany(
        "INSERT INTO links(doc, collection, field, line, kind, target, key, fragment, shown) VALUES(?,?,?,?,?,?,?,?,?)",
        [(doc, cid, *link) for doc, cid, _, _, found, _ in docs for link in found],
    )


def _stem(path):
    """A path as paths compare: folded, a note's without `.md`."""
    stem, ext = posixpath.splitext(path)
    return readers.fold(stem if ext.lower() in readers.MARKDOWN else path)


def _one(via, docs):
    """How a link resolves to these candidates, (doc id, path) each: to the one, or ambiguously."""
    return (via, [docs[0][0]]) if len(docs) == 1 else ("ambiguous", [doc for doc, _ in docs])


class Resolver:
    """Resolves links to documents, looking each name up once."""

    def __init__(self, db):
        self.db, self._named = db, {}

    def _candidates(self, cid, key):
        """The files of a collection that answer to `key`, by file name and by alias: (doc id, path) each, by path."""
        if (cid, key) not in self._named:
            sql = """SELECT n.kind, n.doc, d.path FROM names n JOIN docs d ON d.id = n.doc
                     WHERE n.collection = ? AND n.key = ? ORDER BY d.path"""
            rows = self.db.execute(sql, (cid, key)).fetchall()
            self._named[cid, key] = tuple(
                [(doc, path) for kind, doc, path in rows if kind == k] for k in ("name", "alias")
            )
        return self._named[cid, key]

    def resolve(self, cid, source, kind, target, key):
        """Where a link of the note at path `source` in collection `cid` leads: ("path" | "name" | "alias", [doc id]),
        ("ambiguous", [doc ids]) or ("unresolved", [])."""
        named, aliased = self._candidates(cid, key)
        relative = target.startswith(("./", "../"))
        if relative or kind == "markdown":
            at = posixpath.normpath(posixpath.join(posixpath.dirname(source), target))
            if found := [d for d in named if _stem(d[1]) == _stem(at)]:
                return _one("path", found)
            if relative:
                return UNRESOLVED
        path = target.lstrip("/")
        if "/" in path:
            found = [d for d in named if _stem(d[1]) == _stem(path)]
            found = found or [d for d in named if _stem(d[1]).endswith("/" + _stem(path))]
            return _one("path", found) if found else UNRESOLVED
        if len(named) > 1:
            here = readers.fold(posixpath.dirname(source))
            found = [d for d in named if readers.fold(posixpath.dirname(d[1])) == here]
            return _one("name", found if len(found) == 1 else named)
        if named:
            return "name", [named[0][0]]
        return _one("alias", aliased) if aliased else UNRESOLVED


LINK = "l.doc, l.collection, d.path, l.field, l.line, l.kind, l.target, l.key, l.fragment, l.shown"
"""What the queries below read of a link and the note holding it."""


def _resolve(resolver, row):
    """Where the link of a row with LINK's columns leads."""
    return resolver.resolve(row[1], row[2], row[5], row[6], row[7])


def outgoing(db, resolver, doc, field=None):
    """The links of a document in order, each as (row with LINK's columns, (how, [doc ids])); with `field`, only those
    in that front-matter field."""
    sql, args = f"SELECT {LINK} FROM links l JOIN docs d ON d.id = l.doc WHERE l.doc = ?", [doc]
    if field is not None:
        sql, args = sql + " AND l.field = ?", [*args, field]
    return [(row, _resolve(resolver, row)) for row in db.execute(sql + " ORDER BY l.id", args)]


def incoming(db, resolver, doc, field=None):
    """The links into a document from the other documents of its collection, by their path and line: rows with LINK's
    columns; with `field`, only those in that front-matter field. Only links that match one of the document's names
    can lead to it, so only those are resolved, found through the index on links' keys."""
    keys = [key for (key,) in db.execute("SELECT DISTINCT key FROM names WHERE doc = ?", (doc,))]
    sql = f"""SELECT {LINK} FROM links l JOIN docs d ON d.id = l.doc JOIN docs t ON t.id = ?
              WHERE l.collection = t.collection AND l.key IN (SELECT value FROM json_each(?)) AND l.doc != t.id"""
    args = [doc, json.dumps(keys)]
    if field is not None:
        sql, args = sql + " AND l.field = ?", [*args, field]
    rows = db.execute(sql + " ORDER BY d.path, l.line, l.id", args)
    return [row for row in rows if _resolve(resolver, row) in (("path", [doc]), ("name", [doc]), ("alias", [doc]))]


def unresolved(db, cid=None):
    """The targets of links in a collection, or in all, that no document answers to, each as (collection id, target as
    most often written, [ids of the documents linking it], number of links). Links to the same path are one target,
    however written: `[[Missing]]`, `[[missing.md]]`."""
    resolver, targets = Resolver(db), {}
    sql, args = f"SELECT {LINK} FROM links l JOIN docs d ON d.id = l.doc", []
    if cid is not None:
        sql, args = sql + " WHERE l.collection = ?", [cid]
    for row in db.execute(sql + " ORDER BY l.id", args):
        if _resolve(resolver, row)[0] != "unresolved":
            continue
        doc, collection, source, _, _, kind, target = row[:7]
        if target.startswith(("./", "../")) or kind == "markdown":
            path = posixpath.normpath(posixpath.join(posixpath.dirname(source), target))
        else:
            path = target.lstrip("/")
        written, docs = targets.setdefault((collection, _stem(path)), (Counter(), {}))
        written[target] += 1
        docs[doc] = None
    return [
        (collection, min(written.items(), key=lambda tc: (-tc[1], tc[0]))[0], list(docs), written.total())
        for (collection, _), (written, docs) in targets.items()
    ]
