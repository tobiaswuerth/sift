"""Search: exact keyword (BM25), semantic (exact vector search), hybrid with reranking, and documents like one.

The models embed queries and rerank, wherever they run (see `models`). Through a worker, the vectors are searched
as a NumPy matrix and PyTorch never loads: a search starts in a fraction of a second, even on a light notebook."""

import json
import re
import sqlite3
from operator import itemgetter

import numpy as np

from . import models as _models
from . import readers, store

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it of on or that the this to "
    "was what when where which who why with".split()
)

SQLITE_INT = 2**63  # SQLite integers are 64-bit
OPERATORS = ("=", "<", "<=", ">", ">=", "not")
FIELD_SAMPLE = 1000  # documents per collection whose front matter tells which fields exist


def _condition(key, op, value):
    """SQL comparing a front-matter value (`value` of json_each) with a filter value, and its arguments. For "=" and
    "not", a list stands for any of its values."""
    if op not in OPERATORS:
        raise ValueError(f"filter {key!r}: unknown operator {op!r}, use one of {', '.join(OPERATORS)}")
    values = value if isinstance(value, list) and op in ("=", "not") else [value]
    if not values:
        raise ValueError(f"filter {key!r} needs at least one value")
    for one in values:
        if not isinstance(one, str | int | float):
            raise ValueError(f"filter {key!r} needs a string or number, or for = and not a list of them, got {one!r}")
        if isinstance(one, int) and not -SQLITE_INT <= one < SQLITE_INT:
            raise ValueError(f"filter {key!r}: {one} is too large")
    if op in ("=", "not"):
        return f"value IN ({', '.join('?' * len(values))})", values
    if isinstance(value, str):  # at the precision given: "2024-12-31" is <= "2024", and not > "2024"
        return f"substr(value, 1, ?) {op} ?", [len(value), value]
    return f"CAST(value AS REAL) {op} ?", [value]  # SQLite reads "2024-06-15" as 2024


def _where(db, collection=None, path=None, filters=None):
    """SQL condition on docs `d`. Filters match front-matter fields: a value, which a list field matches if it
    contains it, a list of values for any of them, or conditions such as {">=": "2024-06", "<": "2025"}, which one
    value or list element must satisfy, and {"not": value or list}, which none may match."""
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
        conditions = value if isinstance(value, dict) else {"=": value}
        some = [_condition(key, op, v) for op, v in conditions.items() if op != "not"]
        values = f"SELECT 1 FROM json_each(d.meta, '$.{key}') WHERE"
        if some:
            sql.append(f"EXISTS ({values} {' AND '.join(c for c, _ in some)})")
            args += [arg for _, condition_args in some for arg in condition_args]
        if "not" in conditions:
            condition, condition_args = _condition(key, "not", conditions["not"])
            sql.append(f"NOT EXISTS ({values} {condition})")
            args += condition_args
    return " AND ".join(sql), args


def _check_fields(fields):
    if fields is not None and (isinstance(fields, str) or not all(isinstance(f, str) and f for f in fields)):
        raise ValueError(f"fields must be a list of front-matter field names, or ['*'] for all, got {fields!r}")
    return fields


def _pick(meta, fields):
    """The front-matter `fields` asked for that a document has, from its meta (JSON); ["*"]: all but the title."""
    meta = json.loads(meta)
    return {
        key: meta[key] for key in ([key for key in meta if key != "title"] if "*" in fields else fields) if key in meta
    }


def _check_sort(sort):
    if sort is not None and not (isinstance(sort, str) and re.fullmatch(r"-?\w+", sort)):
        raise ValueError(
            f"sort must be a front-matter field, with - first to sort descending, e.g. '-created'; got {sort!r}"
        )
    return sort


def _by_field(hits, sort):
    """Hits ordered by the front-matter field `sort` names, descending with "-" first: numbers before text, and hits
    without the field last, in their order."""
    field = sort.removeprefix("-")
    valued = [(json.loads(hit["meta"]).get(field), hit) for hit in hits]
    have = [(value, hit) for value, hit in valued if value is not None]
    have.sort(key=lambda vh: (0, vh[0]) if isinstance(vh[0], int | float) else (1, str(vh[0])), reverse=sort[0] == "-")
    return [hit for _, hit in have] + [hit for value, hit in valued if value is None]


def _present(hits, fields, compact, sort):
    """Hits as returned: ordered by `sort` if given, with the front-matter `fields` asked for, without text if
    compact."""
    out = []
    for hit in _by_field(hits, sort) if sort else hits:
        shown = {key: hit[key] for key in ("id", "title", "lines", "score")}
        if fields:
            shown["fields"] = _pick(hit["meta"], fields)
        if not compact:
            shown["text"] = hit["text"]
        out.append(shown)
    return out


def _line_span(lines):
    """First and last line of a range such as "17-26", or of a single line: "17" or 17."""
    if isinstance(lines, int) and not isinstance(lines, bool):
        lines = str(lines)
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


def _words(text):
    """The meaningful words of a text: lowercase, without stopwords and single letters, in order."""
    return [w for w in re.findall(r"\w+", str(text).lower()) if w not in STOPWORDS and len(w) > 1]


def _any_word(query):
    """OR query over the meaningful words, used to collect keyword candidates for hybrid search."""
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(_words(query)))


def _named(hit, asked):
    """The front-matter values of a hit, but its title, that share a word with the query `asked` (a set of words):
    an author's name, an id."""
    return [
        str(value)
        for key, field in json.loads(hit["meta"]).items()
        if key != "title"
        for value in (field if isinstance(field, list) else [field])
        if asked & set(_words(value))
    ]


def _passage(hit, named):
    """What the reranker reads of a hit: its title, the front-matter values the query names, and its text. Only those:
    all of them (links, licences, long author lists) diluted the passage, and topical queries ranked worse."""
    return f"{hit['title']}\n{' '.join(named)}\n\n{hit['text']}" if named else f"{hit['title']}\n\n{hit['text']}"


COSINE_WEIGHT = 0.5  # how much the cosine counts in `query`, in units of the reranker's spread: see docs/benchmarks.md


def _blend(rerank, cosine, named, weight=COSINE_WEIGHT):
    """`query`'s scores: the reranker's, plus the candidates' cosine with the query, centred and in units of the
    spread of the reranker's scores, so one weight fits any reranker's scale and the scores keep it. A reranker
    trained on web search ranked scientific claims worse than the embedding did. Candidates whose front matter the
    query names keep the best cosine: the embedding never sees front matter."""
    cosine = np.where(named, cosine.max(), cosine)
    spread = rerank.std() / cosine.std() if np.ptp(cosine) > 0 else 0.0  # ptp: exactly 0 when all agree
    return rerank + weight * spread * (cosine - cosine.mean())


def _first_of_each(items, key):
    """The first of the items with the same key, in order."""
    first = {}
    for item in items:
        first.setdefault(key(item), item)
    return list(first.values())


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

    def _keyword(self, match, k, where, args, all_passages=False):
        """The best passage of each of the k documents that best match an FTS5 query, by BM25; or the k best
        passages."""
        if not match.strip():
            return []
        passages = f"""SELECT fts.rowid passage, c.doc doc, -bm25(fts) score FROM fts
                       JOIN chunks c ON c.id = fts.rowid JOIN docs d ON d.id = c.doc WHERE fts MATCH ? AND {where}"""
        if all_passages:
            sql = f"SELECT passage, score FROM ({passages}) ORDER BY score DESC LIMIT ?"
        else:  # MATERIALIZED: SQLite can't take bm25() into an aggregate
            sql = f"""WITH p AS MATERIALIZED ({passages})
                      SELECT passage, max(score) FROM p GROUP BY doc ORDER BY 2 DESC LIMIT ?"""
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

    def _semantic(self, query, k, where, args, all_passages=False):
        if not query.strip():  # the embedding of nothing is close to arbitrary chunks
            return []
        return self._nearest(self._scores(query), k, where, args, all_passages)

    def _nearest(self, scores, k, where, args, all_passages):
        """The best passage of each of the k documents with the best `scores` (one per vector), among the documents
        `where` allows; or the k best passages."""
        if args:  # restrict to chunks of matching documents
            sql = f"SELECT c.id FROM chunks c JOIN docs d ON d.id = c.doc WHERE {where}"
            allowed = np.fromiter((c for (c,) in self.db.execute(sql, args)), np.int64)
            scores = np.where(np.isin(self.vectors.ids, allowed, assume_unique=True), scores, -np.inf)
        if all_passages:
            return _top(self.vectors.ids, scores, k)
        n = k
        while True:  # when the best passages come from a few documents, look further for others
            ranked = _top(self.vectors.ids, scores, n)
            best = self._per_document(ranked)
            if len(best) >= k or len(ranked) < n:
                return best[:k]
            n *= 4

    def _per_document(self, ranked):
        """The first, best, of each document's passages in `ranked`."""
        sql = "SELECT c.id, c.doc FROM json_each(?) j JOIN chunks c ON c.id = j.value"
        docs = dict(self.db.execute(sql, (json.dumps([chunk for chunk, _ in ranked]),)))
        return _first_of_each(
            (r for r in ranked if r[0] in docs), lambda r: docs[r[0]]
        )  # in docs: not removed meanwhile

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
            hits.append(hit | {"meta": meta, "text": text, "chunk": chunk})  # for _present, and `query`'s scores
        return hits

    def _cosines(self, scores, chunks):
        """The cosines of these chunks among all `scores`; one not embedded yet gets their mean, neither a bonus nor a
        penalty."""
        at = np.searchsorted(self.vectors.ids, chunks)  # the vector ids are sorted
        found = (at < len(self.vectors.ids)) & (self.vectors.ids[np.minimum(at, len(self.vectors.ids) - 1)] == chunks)
        cosine = np.where(found, scores[np.minimum(at, len(scores) - 1)], np.nan)
        return np.where(np.isnan(cosine), np.nanmean(cosine) if found.any() else 0.0, cosine)

    # --- public API ----------------------------------------------------------

    # Every search returns the best passage of each document it finds, or with `all_passages` every passage, and
    # takes `fields`, front-matter fields to add to each hit (["*"] for all), `compact`, which leaves out the text,
    # and `sort`, a front-matter field to order the hits by instead of score ("-created": newest first).

    def search(
        self,
        query,
        k=10,
        *,
        collection=None,
        path=None,
        filters=None,
        fields=None,
        compact=False,
        all_passages=False,
        sort=None,
    ):
        """Exact keyword search, ranked by BM25. Accepts FTS5 syntax: "phrases", prefix*, AND, OR, NOT.
        If no chunk holds all terms of a query without operators, chunks with any of them are returned."""
        k, where = _check_k(k), _where(self.db, collection, path, filters)
        fields, sort = _check_fields(fields), _check_sort(sort)
        ranked = self._keyword(query, k, *where, all_passages)
        if not ranked and (relaxed := _any_term(query)):
            ranked = self._keyword(relaxed, k, *where, all_passages)
        return _present(self._hits(ranked), fields, compact, sort)

    def vsearch(
        self,
        query,
        k=10,
        *,
        collection=None,
        path=None,
        filters=None,
        fields=None,
        compact=False,
        all_passages=False,
        sort=None,
    ):
        """Semantic search: cosine similarity between the query and every chunk."""
        k, where = _check_k(k), _where(self.db, collection, path, filters)
        fields, sort = _check_fields(fields), _check_sort(sort)
        return _present(self._hits(self._semantic(query, k, *where, all_passages)), fields, compact, sort)

    def query(
        self,
        query,
        k=10,
        *,
        collection=None,
        path=None,
        filters=None,
        fields=None,
        compact=False,
        all_passages=False,
        sort=None,
        candidates=40,
    ):
        """Hybrid search for RAG: keyword and semantic candidates, all reranked together. The reranker alone decides
        the order: fusing the candidates' ranks into it, e.g. by reciprocal rank, made results worse (MRR@10 0.69 to
        0.64 on a known-item test over 3.2 million arXiv abstracts)."""
        k, where = _check_k(k), _where(self.db, collection, path, filters)
        fields, sort = _check_fields(fields), _check_sort(sort)
        candidates = max(candidates, k)
        keyword = self._keyword(_any_word(query), candidates, *where, all_passages)
        cosines = self._scores(query) if query.strip() else None  # for the semantic candidates and every other
        semantic = self._nearest(cosines, candidates, *where, all_passages) if cosines is not None else []
        hits = self._hits([(chunk, 0.0) for chunk in dict.fromkeys(chunk for chunk, _ in keyword + semantic)])
        if not hits:
            return []
        asked = set(_words(query))
        named = [_named(h, asked) for h in hits]
        rerank = np.asarray(self.models.rerank(query, [_passage(h, n) for h, n in zip(hits, named)]), np.float64)
        scores = _blend(rerank, self._cosines(cosines, [h["chunk"] for h in hits]), np.array([bool(n) for n in named]))
        for h, s in zip(hits, scores):
            h["score"] = round(float(s), 4)
        hits = sorted(hits, key=lambda h: -h["score"])
        hits = hits if all_passages else _first_of_each(hits, itemgetter("id"))
        return _present(hits[:k], fields, compact, sort)

    def get(self, id, lines=None):
        """Text of an indexed document, by its `collection/path` id: all of it, or `lines` such as "17-26", as
        search hits give them, or one line number. The path may leave out folders if the file name is unique in the
        collection. The text comes from the index as it was indexed, so it always matches the hits."""
        return self.document(id, lines)["text"]

    def document(self, id, lines=None, fields=None):
        """A document as {"id", "title", "text"}, the whole id even if asked by file name, and with "fields" if asked
        for, as in hits; `id` and `lines` as for `get`."""
        span, fields = None if lines is None else _line_span(lines), _check_fields(fields)
        _, whole, title, meta, text = self._find(id)
        document = {"id": whole, "title": title} | ({"fields": _pick(meta, fields)} if fields else {})
        return document | {"text": text if span is None else "".join(readers.LINE.findall(text)[span[0] - 1 : span[1]])}

    def similar(
        self,
        id,
        k=10,
        *,
        collection=None,
        path=None,
        filters=None,
        fields=None,
        compact=False,
        all_passages=False,
        sort=None,
    ):
        """Documents closest in meaning to the document `id` (as for `get`), by cosine similarity with the mean of its
        passages' vectors; the document itself is left out."""
        k, where = _check_k(k), _where(self.db, collection, path, filters)
        fields, sort = _check_fields(fields), _check_sort(sort)
        doc = self._find(id)[0]
        self.load()
        sql = "SELECT v.chunk, v.vec FROM vectors v JOIN chunks c ON c.id = v.chunk WHERE c.doc = ?"
        own = self.db.execute(sql, (doc,)).fetchall()
        if not own:
            raise LookupError(f"{id!r} has no embeddings yet, run `sift embed`")
        mean = np.mean([np.frombuffer(vec, np.float16) for _, vec in own], axis=0, dtype=np.float32)
        scores = self.vectors.scores(mean / np.linalg.norm(mean))
        scores = np.where(np.isin(self.vectors.ids, [chunk for chunk, _ in own]), -np.inf, scores)
        return _present(self._hits(self._nearest(scores, k, *where, all_passages)), fields, compact, sort)

    def _find(self, id):
        """The document `id` names (see `get`), as (row id, whole id, title, meta, text)."""
        name, _, rel = id.partition("/")
        sql = """SELECT d.id, d.path, d.title, d.meta, d.text FROM docs d JOIN collections co ON co.id = d.collection
                 WHERE co.name = ? AND """
        rows = self.db.execute(sql + "d.path = ?", (name, rel)).fetchall()
        if not rows and rel:  # e.g. arxiv/2603.29315.md for arxiv/2026/03/2603.29315.md
            rows = self.db.execute(sql + "substr(d.path, -?) = ? LIMIT 6", (name, len(rel) + 1, "/" + rel)).fetchall()
        if len(rows) > 1:
            matches = ", ".join(f"{name}/{row[1]}" for row in rows[:5]) + (", ..." if len(rows) > 5 else "")
            raise LookupError(f"{id!r} matches {matches}; give the whole id")
        if not rows:
            raise LookupError(f"unknown document {id!r}")
        doc, path, title, meta, text = rows[0]
        return doc, f"{name}/{path}", title, meta, text

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
