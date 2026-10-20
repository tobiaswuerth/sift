"""The three search modes: exact keyword (BM25), semantic (exact vector search) and hybrid with reranking.

The models embed queries and rerank, wherever they run (see `models`). Through a worker, the vectors are searched
as a NumPy matrix and PyTorch never loads: a search starts in a fraction of a second, even on a light notebook."""

import json
import re
import sqlite3

import numpy as np

from . import models as _models
from . import store

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it of on or that the this to "
    "was what when where which who why with".split()
)
RRF_K = 60
SQLITE_INT = 2**63  # SQLite integers are 64-bit
OPERATORS = ("=", "<", "<=", ">", ">=")
FIELD_SAMPLE = 1000  # documents per collection whose front matter tells which fields exist


def _condition(key, op, value):
    """SQL comparing a front-matter value (`value` of json_each) with a filter value, and its arguments."""
    if op not in OPERATORS:
        raise ValueError(f"filter {key!r}: unknown operator {op!r}, use one of {', '.join(OPERATORS)}")
    if not isinstance(value, str | int | float):
        raise ValueError(f"filter {key!r} needs a single string or number, got {value!r}")
    if isinstance(value, int) and not -SQLITE_INT <= value < SQLITE_INT:
        raise ValueError(f"filter {key!r}: {value} is too large")
    if op == "=":
        return "value = ?", [value]
    if isinstance(value, str):  # at the precision given: "2024-12-31" is <= "2024", and not > "2024"
        return f"substr(value, 1, ?) {op} ?", [len(value), value]
    return f"CAST(value AS REAL) {op} ?", [value]  # SQLite reads "2024-06-15" as 2024


def _where(db, collection=None, path=None, filters=None):
    """SQL condition on docs `d`. Filters match front-matter fields: a value, which a list field matches if it
    contains it, or a range such as {">=": "2024-06", "<": "2025"}, which one value or list element must satisfy."""
    sql, args = ["1"], []
    if collection:
        row = db.execute("SELECT id FROM collections WHERE name = ?", (collection,)).fetchone()
        if not row:
            raise LookupError(f"no collection named {collection!r}")
        sql.append("d.collection = ?")
        args.append(row[0])
    if path:
        sql.append("d.path GLOB ?")
        args.append(path)
    for key, value in (filters or {}).items():
        if not re.fullmatch(r"\w+", key):
            raise ValueError(f"invalid filter field {key!r}")
        if value == {}:
            raise ValueError(f"filter {key!r} has no condition")
        conditions = [_condition(key, *c) for c in (value.items() if isinstance(value, dict) else [("=", value)])]
        sql.append(
            f"EXISTS (SELECT 1 FROM json_each(d.meta, '$.{key}') WHERE {' AND '.join(c for c, _ in conditions)})"
        )
        args += [arg for _, condition_args in conditions for arg in condition_args]
    return " AND ".join(sql), args


def _check_fields(fields):
    if fields is not None and (isinstance(fields, str) or not all(isinstance(f, str) and f for f in fields)):
        raise ValueError(f"fields must be a list of front-matter field names, or ['*'] for all, got {fields!r}")
    return fields


def _present(hits, fields, compact):
    """Hits as returned: with the front-matter `fields` asked for, and without text if compact."""
    out = []
    for hit in hits:
        shown = {key: hit[key] for key in ("id", "title", "lines", "score")}
        if fields:
            meta = json.loads(hit["meta"])
            pick = [key for key in meta if key != "title"] if "*" in fields else fields
            shown["fields"] = {key: meta[key] for key in pick if key in meta}
        if not compact:
            shown["text"] = hit["text"]
        out.append(shown)
    return out


def _line_span(lines):
    """First and last line of a range such as "17-26", or of a single line such as "17"."""
    m = isinstance(lines, str) and re.fullmatch(r"([1-9]\d*)(?:-([1-9]\d*))?", lines)
    if not m or int(m[2] or m[1]) < int(m[1]):
        raise ValueError(f"lines must be a line number or a range such as '17-26', got {lines!r}")
    return int(m[1]), int(m[2] or m[1])


def _check_k(k):
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return min(k, SQLITE_INT - 1)


def _phrases(query):
    """Fallback for input that isn't valid FTS5 syntax: every word becomes an exact phrase, all required."""
    return " ".join('"' + w.replace('"', '""') + '"' for w in query.split())


def _any_term(query):
    """The terms of a keyword query joined by OR, for when no chunk holds them all. None for a single term, and
    for queries with operators, which say what they want."""
    terms = [t for t in re.findall(r'"[^"]*"|[^\s"]+', query) if re.search(r"\w", t)]
    if len(terms) < 2 or any(t in ("AND", "OR", "NOT") or t.startswith("NEAR(") for t in terms):
        return None
    return " OR ".join(t if t[0] == '"' else f'"{t[:-1]}" *' if t[-1] == "*" else f'"{t}"' for t in terms)


def _any_word(query):
    """OR query over the meaningful words, used to collect keyword candidates for hybrid search."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def _top(ids, scores, k):
    """The k best (chunk id, score) pairs, best first. A broken (NaN) vector ranks last: PyTorch's topk ranked NaN
    first, so a few broken vectors pushed every other result out."""
    scores = np.where(np.isnan(scores), -np.inf, scores)
    k = min(k, len(scores))
    best = np.argpartition(-scores, k - 1)[:k]
    best = best[np.lexsort((best, -scores[best]))]  # by score, ties by position: the same order every time
    return [(int(ids[i]), float(scores[i])) for i in best if scores[i] > -np.inf]


class Index:
    """Search over an index. Semantic search uses `models` (default: `models.default()`, chosen when first needed,
    so keyword search loads no models)."""

    def __init__(self, db=None, models=None):
        self.db = db or store.connect()
        self._models, self.version, self.vectors = models, None, None

    @property
    def models(self):
        if self._models is None:
            self._models = _models.default()
        return self._models

    # --- candidates as [(chunk_id, score)] -----------------------------------

    def _keyword(self, match, k, where, args):
        if not match.strip():
            return []
        sql = f"""SELECT fts.rowid, -bm25(fts, 2.0, 1.0) FROM fts JOIN chunks c ON c.id = fts.rowid
                  JOIN docs d ON d.id = c.doc WHERE fts MATCH ? AND {where} ORDER BY bm25(fts, 2.0, 1.0) LIMIT ?"""
        for attempt in (match, _phrases(match)):
            try:
                return self.db.execute(sql, [attempt, *args, k]).fetchall()
            except sqlite3.OperationalError as e:
                error = e
        raise ValueError(f"invalid search query {match!r}: {error}")

    def load(self):
        """Load all vectors, where the models score them; reload when the index has changed since the last load."""
        version = store.get_meta(self.db, "version")
        if version is not None and version == self.version:
            return
        n = self.db.execute("SELECT count(*) FROM vectors").fetchone()[0]
        if not n:
            raise LookupError("no embeddings yet, run `sift embed`")
        ids, mat = np.empty(n, np.int64), np.empty((n, int(store.get_meta(self.db, "dim"))), np.float16)
        for row, (chunk, vec) in enumerate(self.db.execute("SELECT chunk, vec FROM vectors ORDER BY chunk")):
            ids[row], mat[row] = chunk, np.frombuffer(vec, np.float16)
        if self.vectors:
            self.vectors.close()  # a GPU copy of the old vectors goes with them
        self.vectors, self.version = self.models.vectors(ids, mat), version

    def _scores(self, query):
        """Cosine similarity of the query with every chunk, as a NumPy array."""
        self.load()
        return self.vectors.scores(self.models.encode([query], query=True)[0])

    def _semantic(self, query, k, where, args):
        if not query.strip():  # the embedding of nothing is close to arbitrary chunks
            return []
        scores = self._scores(query)
        if args:  # restrict to chunks of matching documents
            sql = f"SELECT c.id FROM chunks c JOIN docs d ON d.id = c.doc WHERE {where}"
            allowed = np.fromiter((c for (c,) in self.db.execute(sql, args)), np.int64)
            scores = np.where(np.isin(self.vectors.ids, allowed, assume_unique=True), scores, -np.inf)
        return _top(self.vectors.ids, scores, k)

    def _hits(self, ranked):
        sql = """SELECT co.name, d.path, d.title, d.meta, c.line_start, c.line_end, substr(d.text, c.pos + 1, c.len)
                 FROM chunks c JOIN docs d ON d.id = c.doc JOIN collections co ON co.id = d.collection WHERE c.id = ?"""
        hits = []
        for chunk, score in ranked:
            row = self.db.execute(sql, (chunk,)).fetchone()
            if row is None:  # removed by a concurrent `sift index`
                continue
            name, path, title, meta, start, end, text = row
            hit = {"id": f"{name}/{path}", "title": title, "lines": f"{start}-{end}", "score": round(score, 4)}
            hits.append(hit | {"meta": meta, "text": text})  # meta for _present, text for reranking
        return hits

    # --- public API ----------------------------------------------------------

    # Every search takes `fields`, front-matter fields to add to each hit (["*"] for all), and `compact`, which
    # leaves out the text.

    def search(self, query, k=10, collection=None, path=None, filters=None, fields=None, compact=False):
        """Exact keyword search, ranked by BM25. Accepts FTS5 syntax: "phrases", prefix*, AND, OR, NOT.
        If no chunk holds all terms of a query without operators, chunks with any of them are returned."""
        k, where, fields = _check_k(k), _where(self.db, collection, path, filters), _check_fields(fields)
        ranked = self._keyword(query, k, *where)
        if not ranked and (relaxed := _any_term(query)):
            ranked = self._keyword(relaxed, k, *where)
        return _present(self._hits(ranked), fields, compact)

    def vsearch(self, query, k=10, collection=None, path=None, filters=None, fields=None, compact=False):
        """Semantic search: cosine similarity between the query and every chunk."""
        k, where, fields = _check_k(k), _where(self.db, collection, path, filters), _check_fields(fields)
        return _present(self._hits(self._semantic(query, k, *where)), fields, compact)

    def query(self, query, k=10, collection=None, path=None, filters=None, fields=None, compact=False, candidates=40):
        """Hybrid search for RAG: keyword and semantic candidates, fused by reciprocal rank, then reranked."""
        k, where, fields = _check_k(k), _where(self.db, collection, path, filters), _check_fields(fields)
        candidates = max(candidates, k)
        fused = {}
        for ranked in (self._keyword(_any_word(query), candidates, *where), self._semantic(query, candidates, *where)):
            for rank, (chunk, _) in enumerate(ranked):
                fused[chunk] = fused.get(chunk, 0) + 1 / (RRF_K + rank)
        hits = self._hits(sorted(fused.items(), key=lambda x: -x[1]))
        if not hits:
            return []
        scores = self.models.rerank(query, [f"{h['title']}\n\n{h['text']}" for h in hits])
        for h, s in zip(hits, scores):
            h["score"] = round(float(s), 4)
        return _present(sorted(hits, key=lambda h: -h["score"])[:k], fields, compact)

    def get(self, id, lines=None):
        """Text of an indexed document, by its `collection/path` id: all of it, or `lines` such as "17-26", as
        search hits give them. The path may leave out folders if the file name is unique in the collection.
        The text comes from the index as it was indexed, so it always matches the hits."""
        span = None if lines is None else _line_span(lines)
        name, _, rel = id.partition("/")
        sql = "SELECT d.text, d.path FROM docs d JOIN collections co ON co.id = d.collection WHERE co.name = ? AND "
        rows = self.db.execute(sql + "d.path = ?", (name, rel)).fetchall()
        if not rows and rel:  # e.g. arxiv/2603.29315.md for arxiv/2026/03/2603.29315.md
            rows = self.db.execute(sql + "substr(d.path, -?) = ? LIMIT 6", (name, len(rel) + 1, "/" + rel)).fetchall()
        if len(rows) > 1:
            matches = ", ".join(f"{name}/{path}" for _, path in rows[:5]) + (", ..." if len(rows) > 5 else "")
            raise LookupError(f"{id!r} matches {matches}; give the whole id")
        if not rows:
            raise LookupError(f"unknown document {id!r}")
        text = rows[0][0]
        return text if span is None else "".join(store.LINE.findall(text)[span[0] - 1 : span[1]])

    def fields(self, collection, docs):
        """The front-matter fields of a collection (by id with `docs` documents), from a sample spread over it."""
        sql = """SELECT DISTINCT j.key FROM (SELECT meta FROM docs WHERE collection = ? AND id % ? = 0 LIMIT ?) d,
                 json_each(d.meta) j ORDER BY j.key"""
        return [key for (key,) in self.db.execute(sql, (collection, max(1, docs // FIELD_SAMPLE), FIELD_SAMPLE))]

    def status(self):
        rows = self.db.execute("""
            SELECT co.id, co.name, co.path, count(DISTINCT d.id), count(c.id), count(v.chunk) FROM collections co
            LEFT JOIN docs d ON d.collection = co.id LEFT JOIN chunks c ON c.doc = d.id
            LEFT JOIN vectors v ON v.chunk = c.id GROUP BY co.id ORDER BY co.name""").fetchall()
        return {
            "index": self.db.execute("PRAGMA database_list").fetchone()[2],
            "model": store.get_meta(self.db, "model"),
            "collections": [
                {"name": n, "path": p, "docs": d, "chunks": c, "embedded": e, "fields": self.fields(i, d)}
                for i, n, p, d, c, e in rows
            ],
        }
