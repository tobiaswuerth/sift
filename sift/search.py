"""Search: exact keyword (BM25), semantic (exact vector search), hybrid with reranking, and documents like one.

The models embed queries and rerank, wherever they run (see `models`). Through a worker, the vectors are searched
as a NumPy matrix and PyTorch never loads: a search starts in a fraction of a second, even on a light notebook."""

import json
import os
import posixpath
import re
import sqlite3
from collections import Counter
from operator import itemgetter
from pathlib import Path

import numpy as np

from . import console, embed, indexing, links, readers, store
from . import models as _models
from .vectors import patch

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it of on or that the this to "
    "was what when where which who why with".split()
)

SQLITE_INT = 2**63  # SQLite integers are 64-bit
OPERATORS = ("=", "<", "<=", ">", ">=", "not")
FIELD_SAMPLE = 1000  # documents per collection whose front matter tells which fields exist
START = 600  # characters of a document's start that `browse` shows, as its passage
LAST = "\U0010ffff"  # the last character there is: every path starting with p sorts between p and p + LAST


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


def _field(name, what="filter field"):
    """A front-matter field name, as SQLite's JSON functions address it: any name without a double quote."""
    if not (isinstance(name, str) and name and '"' not in name):
        raise ValueError(f"invalid {what} {name!r}")
    return f'$."{name}"'


def _collection_id(db, name):
    row = db.execute("SELECT id FROM collections WHERE name = ?", (name,)).fetchone()
    if not row:
        raise store.no_collection(db, name)
    return row[0]


def _target(spec):
    """The target of a note named as links name it: `Acme AG`, `[[Acme AG|Acme]]`, `[[02_Organizations/Acme AG]]`."""
    spec = spec.strip()
    if readers.WIKILINK_VALUE.fullmatch(spec):
        spec = spec[2:-2].partition("|")[0].removesuffix("\\").partition("#")[0].strip()
    return spec


def _wikilinks(value):
    """The filter values among `value`, one or a list, that are wikilinks."""
    values = value if isinstance(value, list) else [value]
    return [v for v in values if isinstance(v, str) and readers.WIKILINK_VALUE.fullmatch(v.strip())]


def _ids(ids):
    """Document row ids as one SQL argument, for `d.id IN (SELECT value FROM json_each(?))`."""
    return json.dumps(sorted(ids))


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
    if sort is not None:
        try:
            _field(sort.removeprefix("-") if isinstance(sort, str) else sort)
        except ValueError:
            raise ValueError(
                f"sort must be a front-matter field, with - first to sort descending, e.g. '-created'; got {sort!r}"
            ) from None
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


def line_span(lines):
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
    if re.search(r"(?:^|\s)-?(?:title|text|meta|\{[^}]*\})\s*:", query):  # a column filter: "title: ...", "{a b}: ..."
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


VIA = ("path", "name", "alias")  # how a link resolved, when it did
AMBIGUOUS_SHOWN = links.AMBIGUOUS_SHOWN  # notes an ambiguous link or name lists, of those it matches
MENTION_CHARS = 3  # names `mentions` looks for have at least this many characters
UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"})
LINKING_SHOWN = 5  # documents an unresolved target names, of those linking it
LINE_CHARS = 300  # characters of a linking line a link into a document shows, around the link


def _link_out(row, how, through, named):
    """A link out of a document, as `links` returns it: as written, and where it leads."""
    _, _, _, field, line, kind, target, _, fragment, shown = row
    link = {"target": target, "line": line, "kind": kind}
    link |= {key: value for key, value in (("field", field), ("fragment", fragment), ("shown", shown)) if value}
    via, docs = how
    if via in VIA:
        link |= {"id": named[docs[0]][0], "title": named[docs[0]][1], "via": via}
    elif via == "ambiguous":
        link["ambiguous"] = [named[doc][0] for doc in docs[:AMBIGUOUS_SHOWN]]
    else:
        link["unresolved"] = True
    return link | ({"through": named[through][0]} if through else {})


def _by_document(rows):
    """Link rows by the document holding them, in order."""
    found = {}
    for row in rows:
        found.setdefault(row[0], []).append(row)
    return found


def _lines_of(rows, text):
    """The lines of a document holding these links, once each: their number, front-matter field, and text, around the
    link if the line is long."""
    every, found = text.split("\n"), {}
    for _, _, _, field, line, _, target, *_ in rows:
        if line not in found:
            content = every[line - 1].strip()
            if len(content) > LINE_CHARS:
                start = max(0, content.find(target) - LINE_CHARS // 3)
                end = start + LINE_CHARS
                content = ("…" if start else "") + content[start:end] + ("…" if end < len(content) else "")
            found[line] = {"line": line} | ({"field": field} if field else {}) | {"text": content}
    return list(found.values())


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

    # --- which documents a search may find -----------------------------------

    def _where(self, collection=None, path=None, filters=None, links_to=None, linked_from=None):
        """SQL condition on docs `d`, and its arguments. Filters match front-matter fields: a value, which a list field
        matches if it contains it, a list of values for any of them, or conditions such as {">=": "2024-06", "<":
        "2025"}, which one value or list element must satisfy, and {"not": value or list}, which none may match. A
        value that is a wikilink, "[[Acme AG]]", matches a link to the same note however it is written. `links_to`
        and `linked_from` name notes (see _notes): only documents linking one of the first, and linked from one of
        the second."""
        sql, args, resolver = ["1"], [], links.Resolver(self.db)
        if collection:
            sql.append("d.collection = ?")
            args.append(_collection_id(self.db, collection))
        if path:
            sql.append("d.path GLOB ?")
            args.append(path)
        for key, value in (filters or {}).items():
            field = _field(key)
            if value == {}:
                raise ValueError(f"filter {key!r} has no condition")
            conditions = value if isinstance(value, dict) else {"=": value}
            some = [_condition(key, op, v) for op, v in conditions.items() if op != "not"]
            if some:
                condition = f"EXISTS (SELECT 1 FROM json_each(d.meta, ?) WHERE {' AND '.join(c for c, _ in some)})"
                args += [field, *(arg for _, condition_args in some for arg in condition_args)]
                if list(conditions) == ["="] and (linked := self._linking(key, conditions["="], collection, resolver)):
                    condition = f"({condition} OR d.id IN (SELECT value FROM json_each(?)))"
                    args.append(_ids(linked))
                sql.append(condition)
            if "not" in conditions:
                condition, condition_args = _condition(key, "not", conditions["not"])
                sql.append(f"NOT EXISTS (SELECT 1 FROM json_each(d.meta, ?) WHERE {condition})")
                args += [field, *condition_args]
                if linked := self._linking(key, conditions["not"], collection, resolver):
                    sql.append("d.id NOT IN (SELECT value FROM json_each(?))")
                    args.append(_ids(linked))
        allowed = self._linked(links_to, linked_from, collection, resolver)
        if allowed is not None:
            sql.append("d.id IN (SELECT value FROM json_each(?))")
            args.append(_ids(allowed))
        return " AND ".join(sql), args

    def _collections(self, collection):
        """The row id of a collection, or of every collection, as a list."""
        if collection:
            return [_collection_id(self.db, collection)]
        return [cid for (cid,) in self.db.execute("SELECT id FROM collections ORDER BY id")]

    def _resolve(self, spec, collection, resolver):
        """Where a note named as links name it leads, from no note in particular, in `collection` or in each:
        {collection id: (how, [doc ids])}."""
        target = _target(spec)
        key = readers.key(target.rsplit("/", 1)[-1])
        return {cid: resolver.resolve(cid, None, "wikilink", target, key) for cid in self._collections(collection)}

    def _notes(self, specs, what, collection, resolver):
        """The notes that `specs`, one or a list, name: by a document's id, as for `get`, or as links name them,
        "Acme AG" or "[[Acme AG]]", within `collection` if given, else wherever one note answers. As {spec: doc row
        id, or None for a note no document answers to}; a name that more than one note answers to is an error."""
        specs = [specs] if isinstance(specs, str) else specs
        if not isinstance(specs, list | tuple) or not specs:
            raise ValueError(f"{what} needs at least one note: their ids, or link targets, got {specs!r}")
        found = {}
        for spec in specs:
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError(f"{what} takes ids of documents or link targets, got {spec!r}")
            rows = [] if readers.WIKILINK_VALUE.fullmatch(spec.strip()) else self._rows(spec.strip())
            if len(rows) == 1:
                found[spec] = rows[0][0]
                continue
            if len(rows) > 1:
                self._find(spec.strip())  # raises: an ambiguous id
            ways = self._resolve(spec, collection, resolver).values()
            matches = {doc for how, docs in ways if how != "unresolved" for doc in docs}
            if len(matches) > 1:
                named = sorted(whole for whole, _ in self._named(matches).values())
                shown = ", ".join(named[:AMBIGUOUS_SHOWN]) + (", ..." if len(named) > AMBIGUOUS_SHOWN else "")
                raise LookupError(f"{spec!r} matches {shown}; give the whole id, or a collection")
            found[spec] = next(iter(matches), None)
        return found

    def _linked(self, links_to, linked_from, collection, resolver):
        """The documents that link to any of the notes `links_to` names, and that one of `linked_from` links to, by
        row id; None if neither is given. A note no document answers to yet is linked by the links that name it."""
        allowed = None
        if links_to is not None:
            allowed = set()
            for spec, note in self._notes(links_to, "links_to", collection, resolver).items():
                if note is not None:
                    allowed |= {row[0] for row in links.incoming(self.db, resolver, note)}
                elif docs := {
                    d for cid in self._collections(collection) for d in links.written(self.db, cid, _target(spec))
                }:
                    allowed |= docs
                else:
                    raise LookupError(f"no note {spec!r} in the index, and no link to one")
        if linked_from is not None:
            to = set()
            for spec, note in self._notes(linked_from, "linked_from", collection, resolver).items():
                if note is None:
                    raise LookupError(f"no note {spec!r} in the index")
                to |= {docs[0] for _, (how, docs) in links.outgoing(self.db, resolver, note) if how in VIA}
            allowed = to if allowed is None else allowed & to
        return allowed

    def _linking(self, key, value, collection, resolver):
        """The documents whose front-matter field `key` links to a note that a wikilink among the filter values names,
        however the link is written; links to a note no document answers to, or to several, match as written. None if
        no value is a wikilink."""
        specs, docs = _wikilinks(value), set()
        for spec in specs:
            for cid, (how, found) in self._resolve(spec, collection, resolver).items():
                if how in VIA:
                    docs |= {row[0] for row in links.incoming(self.db, resolver, found[0], key)}
                else:
                    docs |= set(links.written(self.db, cid, _target(spec), key))
        return docs if specs else None

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
        links_to=None,
        linked_from=None,
    ):
        """Exact keyword search, ranked by BM25. Accepts FTS5 syntax: "phrases", prefix*, AND, OR, NOT.
        If no chunk holds all terms of a query without operators, chunks with any of them are returned."""
        k, where = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
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
        links_to=None,
        linked_from=None,
    ):
        """Semantic search: cosine similarity between the query and every chunk."""
        k, where = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
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
        links_to=None,
        linked_from=None,
        candidates=40,
    ):
        """Hybrid search for RAG: keyword and semantic candidates, all reranked together, and ordered by the reranker's
        score blended with their cosine (see _blend). Fusing the candidates' first ranks into the order, e.g. by
        reciprocal rank, made results worse (MRR@10 0.69 to 0.64 on a known-item test over 3.2 million arXiv
        abstracts); the blend made them better on six of the seven sets in docs/benchmarks.md, and cost a little on
        those arXiv known items (0.937 to 0.929)."""
        k, where = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
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
        span, fields = None if lines is None else line_span(lines), _check_fields(fields)
        _, whole, title, meta, text = self._find(id)
        document = {"id": whole, "title": title} | ({"fields": _pick(meta, fields)} if fields else {})
        if span:
            every = readers.LINE.findall(text)
            if span[0] > len(every):
                raise ValueError(f"{whole} has {console.count(len(every), 'line')}; line {span[0]} is past its end")
            text = "".join(every[span[0] - 1 : span[1]])
        return document | {"text": text}

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
        links_to=None,
        linked_from=None,
    ):
        """Documents closest in meaning to the document `id` (as for `get`), by cosine similarity with the mean of its
        passages' vectors; the document itself is left out."""
        k, where = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
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

    def _place(self, id):
        """The collection and the path in it that `id` names: "collection/path", or an absolute path in a
        collection's folder, the nearest that holds it; (None, "") for an absolute path in none."""
        if not os.path.isabs(id):
            name, _, rel = id.partition("/")
            return name, rel
        full = os.path.normpath(id)
        for name, root in self.db.execute("SELECT name, path FROM collections ORDER BY length(path) DESC"):
            root = os.path.normpath(root)
            if os.path.normcase(full).startswith(os.path.normcase(os.path.join(root, ""))):
                return name, os.path.relpath(full, root).replace(os.sep, "/")
        return None, ""

    def _rows(self, id):
        """The documents `id` may name (see `get`), at most six: (row id, path, title, meta, text) of each."""
        name, rel = self._place(id)
        sql = """SELECT d.id, d.path, d.title, d.meta, d.text FROM docs d JOIN collections co ON co.id = d.collection
                 WHERE co.name = ? AND """
        rows = self.db.execute(sql + "d.path = ?", (name, rel)).fetchall()
        if rows or not rel:
            return rows
        if os.path.isabs(id):  # in another case, as Windows and macOS name the same file, and `reindex` takes it
            found = self.db.execute(sql + "lower(d.path) = lower(?)", (name, rel)).fetchall()
            same = [row for row in found if readers.fold(row[1]) == readers.fold(rel)]
            return same if len(same) == 1 else []
        # e.g. arxiv/2603.29315.md for arxiv/2026/03/2603.29315.md
        return self.db.execute(sql + "substr(d.path, -?) = ? LIMIT 6", (name, len(rel) + 1, "/" + rel)).fetchall()

    def _note(self, id):
        """The document `id` names, as for `get`, or else the note `id` names as links name it, "Acme AG" or
        "[[Acme AG]]", if one note answers to it in all collections: as _find gives it."""
        try:
            return self._find(id)
        except LookupError as unknown:
            note = self._notes([id], "id", None, links.Resolver(self.db))[id]
            if note is None:
                raise unknown from None
            return self._find(self._named([note])[note][0])

    def _find(self, id):
        """The document `id` names (see `get`), as (row id, whole id, title, meta, text)."""
        name, rows = self._place(id)[0], self._rows(id)
        if len(rows) > 1:
            matches = ", ".join(f"{name}/{row[1]}" for row in rows[:5]) + (", ..." if len(rows) > 5 else "")
            raise LookupError(f"{id!r} matches {matches}; give the whole id")
        if not rows:
            raise LookupError(f"unknown document {id!r}")
        doc, path, title, meta, text = rows[0]
        return doc, f"{name}/{path}", title, meta, text

    def browse(
        self,
        k=50,
        *,
        collection=None,
        path=None,
        filters=None,
        fields=None,
        compact=False,
        sort=None,
        links_to=None,
        linked_from=None,
    ):
        """Documents without a query, by path: what a collection, folder or filter holds. They come as hits without
        lines and score, with the start of their first passage, after any front matter, as their passage; `sort`
        orders them by a front-matter field instead, as in searches."""
        k, (where, args) = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
        fields, sort = _check_fields(fields), _check_sort(sort)
        order, by = "co.name, d.path", []
        if sort:
            value = "json_extract(d.meta, ?)"
            order = f"{value} IS NULL, {value}{' DESC' if sort[0] == '-' else ''}, {order}"
            by = [_field(sort.removeprefix("-"))] * 2
        # The first chunk's start, after any front matter; as a subquery, not a join, so that sorting a whole collection
        # doesn't join every document first (0.4 s, against 5 s, for 3 million).
        first = f"SELECT substr(d.text, c.pos + 1, min(c.len, {START})) FROM chunks c WHERE c.doc = d.id AND c.seq = 0"
        sql = f"""SELECT co.name, d.path, d.title, d.meta, coalesce(({first}), '') FROM docs d
                  JOIN collections co ON co.id = d.collection WHERE {where} ORDER BY {order} LIMIT ?"""
        docs = []
        for name, doc_path, title, meta, text in self.db.execute(sql, [*args, *by, k]):
            doc = {"id": f"{name}/{doc_path}", "title": title}
            if fields:
                doc["fields"] = _pick(meta, fields)
            docs.append(doc if compact else doc | {"text": text})
        return docs

    def folders(self, collection, path=""):
        """The folders directly inside the folder `path` ("" or e.g. "2024/") of a collection, by name. Each folder
        is one jump through the index, so a folder holding millions of documents costs as little as an empty one."""
        path = path.strip("/") + "/" if path.strip("/") else ""  # "2024", "/2024/": the folder 2024/
        cid, found, after = _collection_id(self.db, collection), [], path
        sql = "SELECT path FROM docs WHERE collection = ? AND path > ? AND path < ? ORDER BY path LIMIT 1"
        while row := self.db.execute(sql, (cid, after, path + LAST)).fetchone():
            name, slash, _ = row[0][len(path) :].partition("/")
            if slash:
                found.append(name)
            after = path + name + (slash and "/" + LAST)  # past the folder, or past the document
        return sorted(found)

    def file(self, id):
        """The file a document was read from, by its id as for `get`; None for a JSON Lines record, which is a part
        of its file."""
        sql = """SELECT co.path, s.path, d.path FROM docs d JOIN sources s ON s.id = d.source
                 JOIN collections co ON co.id = d.collection WHERE d.id = ?"""
        root, source, path = self.db.execute(sql, (self._find(id)[0],)).fetchone()
        return Path(root) / source if source == path else None

    # --- links between notes (see `links`) -----------------------------------

    def links(self, id, direction="out", depth=1, field=None, k=50):
        """The links of a document `id` (as for `get`), as {"id", "title", "out", "in"}: "out" the links it holds, in
        order, each with where it leads; "in" the documents linking it, by path, each with the lines that do. Depth 2
        adds those one link further, each with the document it was reached `through`; `field` keeps to links in that
        front-matter field; k caps each way."""
        if direction not in ("out", "in", "both"):
            raise ValueError(f'direction must be "out", "in" or "both", got {direction!r}')
        if depth not in (1, 2) or isinstance(depth, bool):
            raise ValueError(f"depth must be 1 or 2, got {depth!r}")
        if field is not None and not isinstance(field, str):
            raise ValueError(f"field must be the name of a front-matter field, got {field!r}")
        k = _check_k(k)
        doc, whole, title, _, _ = self._note(id)
        resolver, found = links.Resolver(self.db), {"id": whole, "title": title}
        if direction != "in":
            found["out"] = self._out(resolver, doc, depth, field, k)
        if direction != "out":
            found["in"] = self._in(resolver, doc, depth, field, k)
        return found

    def _out(self, resolver, doc, depth, field, k):
        """The links out of a document, and with depth 2, the documents they lead to link: (row, how, through)."""
        first = links.outgoing(self.db, resolver, doc, field)
        found = [(row, how, None) for row, how in first]
        if depth == 2:
            reached = [docs[0] for _, (via, docs) in first if via in VIA]
            seen = {doc, *reached}
            for step in dict.fromkeys(reached):
                for row, (via, docs) in links.outgoing(self.db, resolver, step, field):
                    if via in VIA and docs[0] not in seen:
                        seen.add(docs[0])
                        found.append((row, (via, docs), step))
        found = found[:k]
        named = self._named([d for _, (_, docs), through in found for d in [*docs, through] if d])
        return [_link_out(row, how, through, named) for row, how, through in found]

    def _in(self, resolver, doc, depth, field, k):
        """The documents linking a document, and with depth 2, those linking them: (doc, its links, through)."""
        first = _by_document(links.incoming(self.db, resolver, doc, field))
        found = [(source, rows, None) for source, rows in first.items()]
        if depth == 2:
            seen = {doc, *first}
            for step in first:
                for source, rows in _by_document(links.incoming(self.db, resolver, step, field)).items():
                    if source not in seen:
                        seen.add(source)
                        found.append((source, rows, step))
        found = found[:k]
        named = self._named([d for source, _, through in found for d in (source, through) if d])
        sql = "SELECT id, text FROM docs WHERE id IN (SELECT value FROM json_each(?))"
        texts = dict(self.db.execute(sql, (json.dumps([source for source, _, _ in found]),)))
        return [
            {"id": named[source][0], "title": named[source][1], "links": _lines_of(rows, texts[source])}
            | ({"through": named[through][0]} if through else {})
            for source, rows, through in found
        ]

    def link_count(self, id):
        """How many documents link to the document `id`, and how many documents it links to: (in, out)."""
        doc, resolver = self._note(id)[0], links.Resolver(self.db)
        into = {row[0] for row in links.incoming(self.db, resolver, doc)}
        out = {docs[0] for _, (via, docs) in links.outgoing(self.db, resolver, doc) if via in VIA}
        return len(into), len(out)

    def _named(self, docs):
        """The whole id and title of documents, by row id."""
        sql = """SELECT d.id, co.name || '/' || d.path, d.title FROM docs d JOIN collections co ON co.id = d.collection
                 WHERE d.id IN (SELECT value FROM json_each(?))"""
        return {doc: (whole, title) for doc, whole, title in self.db.execute(sql, (json.dumps(list(set(docs))),))}

    def unresolved(self, collection=None, k=50):
        """Link targets no document answers to, in a collection or in all: notes not written yet. The most linked
        first, each with how many documents link it and the ids of a few of them."""
        k, cid = _check_k(k), _collection_id(self.db, collection) if collection else None
        found = sorted(links.unresolved(self.db, cid), key=lambda t: (-len(t[2]), t[1]))[:k]
        named = self._named([doc for _, _, docs, _ in found for doc in docs])
        return [
            {"target": target, "docs": len(docs), "ids": sorted(named[doc][0] for doc in docs)[:LINKING_SHOWN]}
            for _, target, docs, _ in found
        ]

    def mentions(self, id, k=10, *, collection=None, also=None, fields=None, compact=False):
        """Documents that name the note `id` without linking to it, found as search finds them, by its names as phrases:
        its file name, its `title`, its aliases, and the values of the front-matter fields `also` names, e.g.
        ["emails"]. The note itself and the documents linking it are left out, and names shorter than MENTION_CHARS,
        which would be everywhere. A name with ä, ö, ü or ß is also looked for as written with ae, oe, ue or ss: the
        keyword index finds Müller as Muller, but not as Mueller."""
        k, fields = _check_k(k), _check_fields(fields)
        also = [also] if isinstance(also, str) else also
        if also is not None and not (isinstance(also, list | tuple) and all(isinstance(f, str) and f for f in also)):
            raise ValueError(f'also must name front-matter fields, e.g. ["emails"], got {also!r}')
        doc, whole, _, meta, _ = self._note(id)
        meta = json.loads(meta)
        names = [posixpath.splitext(whole.rsplit("/", 1)[-1])[0], *links.aliases(meta)]
        for value in [meta.get("title"), *(meta.get(field) for field in also or [])]:
            names += [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
        phrases = {}
        for name in names:
            for variant in (name, name.translate(UMLAUTS)):
                variant = " ".join(variant.split())
                if len(variant) >= MENTION_CHARS and re.search(r"\w", variant) and not _wikilinks(variant):
                    phrases.setdefault(readers.fold(variant), variant)
        if not phrases:
            return []
        where, args = self._where(collection)
        linking = {row[0] for row in links.incoming(self.db, links.Resolver(self.db), doc)} | {doc}
        where, args = where + " AND d.id NOT IN (SELECT value FROM json_each(?))", [*args, _ids(linking)]
        match = " OR ".join('"' + phrase.replace('"', '""') + '"' for phrase in phrases.values())
        return _present(self._hits(self._keyword(match, k, where, args)), fields, compact, None)

    # --- reading files again (see indexing.reindex) ---------------------------

    def reindex(self, paths):
        """Read files again now, after they were written, changed or deleted, and embed their new passages if the index
        is embedded: the next search finds them as they are, its vectors in memory following without a full load. Per
        file: {"id", "status"}, see indexing.reindex, with "chunks", its documents' passages, and "embedded", how many
        of them have vectors; and a "note" on anything that went wrong. Scans are left for `sift index` to read with
        OCR: a server runs this on its event loop, which reading one would hold up for minutes."""
        done = indexing.reindex(self.db, paths, scans=False)
        self._patch(done.before, done.after, removed=done.removed)
        note = self._embed_new([doc for _, docs in done.files for doc in docs])
        sql = """SELECT count(c.id), count(v.chunk) FROM chunks c LEFT JOIN vectors v ON v.chunk = c.id
                 WHERE c.doc IN (SELECT value FROM json_each(?))"""
        results = []
        for entry, docs in done.files:
            if entry["status"] in ("added", "changed", "unchanged"):
                chunks, embedded = self.db.execute(sql, (json.dumps(docs),)).fetchone()
                entry |= {"chunks": chunks, "embedded": embedded}
                if note and embedded < chunks:
                    entry["note"] = "; ".join(filter(None, [entry.get("note"), note]))
            results.append(entry)
        return results

    def _embed_new(self, docs):
        """Embed the passages of these documents that have no vector yet, if the index is embedded, and add them to the
        vectors in memory. Returns why some weren't, or None."""
        model = store.get_meta(self.db, "model")
        if model is None:  # never embedded: keyword search alone, and no model to load
            return None
        sql = """SELECT c.id, c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM chunks c JOIN docs d ON d.id = c.doc
                 WHERE c.doc IN (SELECT value FROM json_each(?))
                 AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id) ORDER BY c.id"""
        rows = self.db.execute(sql, (json.dumps(docs),)).fetchall()
        if not rows:
            return None
        try:
            if (using := self.models.name) != model:
                return f"not embedded: the index was embedded with {model}; run `sift embed --reset` for {using}"
            matrix = embed.encode(self.models, [embed.text_of(*row[1:]) for row in rows])
        except ConnectionError as e:  # a worker that doesn't answer: keyword search finds them meanwhile
            return f"not embedded: {e}"
        intact = np.isfinite(matrix).all(axis=1)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            before = store.get_meta(self.db, "version")
            sql = "SELECT id FROM chunks WHERE id IN (SELECT value FROM json_each(?))"  # not removed meanwhile
            still = {c for (c,) in self.db.execute(sql, (json.dumps([row[0] for row in rows]),))}
            keep = [i for i, row in enumerate(rows) if intact[i] and row[0] in still]
            self.db.executemany(
                "INSERT OR REPLACE INTO vectors(chunk, vec) VALUES(?, ?)",
                [(rows[i][0], matrix[i].tobytes()) for i in keep],
            )
            store.touch(self.db)
            after = store.get_meta(self.db, "version")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._patch(before, after, ids=[rows[i][0] for i in keep], matrix=matrix[keep])
        return None if intact.all() else "some passages could not be embedded (the model returned NaN); try again"

    def _patch(self, before, after, removed=(), ids=(), matrix=None):
        """Bring the vectors in memory from the index's version `before` to `after`, a change this index made itself,
        without loading them all again. Vectors of another version are loaded anew by the next search anyway."""
        if self.vectors is None or self.version != before or before == after:
            return
        if matrix is None:
            matrix = np.empty((0, int(store.get_meta(self.db, "dim"))), np.float16)
        patched = patch(self.vectors, list(removed), list(ids), matrix)
        self.vectors, self.version = (self.vectors, None) if patched is None else (patched, after)

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
        linked, broken = self.link_counts()
        return {
            "index": self.db.execute("PRAGMA database_list").fetchone()[2],
            "model": store.get_meta(self.db, "model"),
            "collections": [
                {"name": n, "path": p, "docs": d, "chunks": c, "embedded": e, "fields": self.fields(i, d)}
                | {"links": linked.get(i, 0), "unresolved": broken.get(i, 0)}
                for i, n, p, d, c, e in rows
            ],
        }

    def link_counts(self):
        """How many links the notes of each collection hold, and how many of them are unresolved, by collection id."""
        linked = dict(self.db.execute("SELECT collection, count(*) FROM links GROUP BY collection"))
        broken = Counter()
        for cid, _, _, n in links.unresolved(self.db):
            broken[cid] += n
        return linked, broken
