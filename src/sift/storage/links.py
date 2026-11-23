import json
from collections.abc import Sequence
from typing import NamedTuple

from sift.domain.text.links import LinkKind, NameKind
from sift.domain.text.resolving import Candidate
from sift.storage.database import Reading, rows

NAMED = """SELECT n.kind, d.id, s.path FROM names n JOIN documents d ON d.id = n.document
           JOIN sources s ON s.id = d.source WHERE n.key = ? ORDER BY s.path"""
OUT = """SELECT l.document, s.path, l.field, l.line, l.kind, l.target, l.key, l.fragment, l.shown
         FROM links l JOIN documents d ON d.id = l.document JOIN sources s ON s.id = d.source
         WHERE l.document = :document AND (:field IS NULL OR l.field = :field) ORDER BY l.ordinal"""
INTO = """SELECT l.document, s.path, l.field, l.line, l.kind, l.target, l.key, l.fragment, l.shown
          FROM links l JOIN documents d ON d.id = l.document JOIN sources s ON s.id = d.source
          WHERE l.key IN (SELECT value FROM json_each(:keys)) AND l.document != :document
            AND (:field IS NULL OR l.field = :field)
          ORDER BY s.path, l.line, l.ordinal"""
KEYED = """SELECT l.document, s.path, l.field, l.line, l.kind, l.target, l.key, l.fragment, l.shown
           FROM links l JOIN documents d ON d.id = l.document JOIN sources s ON s.id = d.source
           WHERE l.key = :key AND (:field IS NULL OR l.field = :field) ORDER BY l.document, l.ordinal"""
KEYS = "SELECT DISTINCT key FROM names WHERE document = ? ORDER BY key"
TITLED = """SELECT d.id, s.path, d.title FROM documents d JOIN sources s ON s.id = d.source
            WHERE d.id IN (SELECT value FROM json_each(?))"""
DOUBTFUL = """SELECT l.document, s.path, l.field, l.line, l.kind, l.target, l.key, l.fragment, l.shown
              FROM links l JOIN documents d ON d.id = l.document JOIN sources s ON s.id = d.source
              WHERE l.kind = 'markdown' OR instr(l.target, '/')
                 OR NOT EXISTS (SELECT 1 FROM names n WHERE n.key = l.key)
              ORDER BY l.document, l.ordinal"""


class Named(NamedTuple):
    kind: str
    document: int
    path: str


class LinkRow(NamedTuple):
    document: int
    path: str
    field: str | None
    line: int
    kind: LinkKind
    target: str
    key: str
    fragment: str | None
    shown: str | None


class Key(NamedTuple):
    key: str


class Titled(NamedTuple):
    document: int
    path: str
    title: str


def named(transaction: Reading, key: str) -> tuple[list[Candidate], list[Candidate]]:
    found = rows(transaction, Named, NAMED, (key,))
    by_name = [Candidate(each.document, each.path) for each in found if each.kind == NameKind.NAME]
    by_alias = [Candidate(each.document, each.path) for each in found if each.kind == NameKind.ALIAS]
    return by_name, by_alias


def _links(found: list[LinkRow]) -> list[LinkRow]:
    return [each._replace(kind=LinkKind(each.kind)) for each in found]


def links_out(transaction: Reading, document: int, field: str | None) -> list[LinkRow]:
    return _links(rows(transaction, LinkRow, OUT, {"document": document, "field": field}))


def links_keyed(transaction: Reading, key: str, field: str | None) -> list[LinkRow]:
    return _links(rows(transaction, LinkRow, KEYED, {"key": key, "field": field}))


def keys_of(transaction: Reading, document: int) -> list[str]:
    return [each.key for each in rows(transaction, Key, KEYS, (document,))]


def links_into(transaction: Reading, keys: Sequence[str], document: int, field: str | None) -> list[LinkRow]:
    asked = {"keys": json.dumps(list(keys)), "document": document, "field": field}
    return _links(rows(transaction, LinkRow, INTO, asked))


def doubtful(transaction: Reading) -> list[LinkRow]:
    return _links(rows(transaction, LinkRow, DOUBTFUL))


def titled(transaction: Reading, documents: Sequence[int]) -> dict[int, Titled]:
    return {each.document: each for each in rows(transaction, Titled, TITLED, (json.dumps(sorted(set(documents))),))}
