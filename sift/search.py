"""Search: exact keyword (BM25), semantic (codes of the vectors, the nearest scored exactly), hybrid with reranking,
and documents like one.

The models embed queries and rerank, wherever they run (see `models`). Through a worker, PyTorch never loads: a search
starts in a fraction of a second, even on a light notebook."""

import json
import math
import os
import posixpath
import re
import sqlite3
from operator import itemgetter
from pathlib import Path

import numpy as np

from . import console, embed, expression, indexing, links, postings, readers, store, topk
from . import models as _models
from .catalog import Catalog
from .vectors import Codes, Filter, codes, top

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
    compact, the other copies of the same work under "also", and "partial" if found by a search out of time."""
    out = []
    for hit in _by_field(hits, sort) if sort else hits:
        shown = {key: hit[key] for key in ("id", "title", "lines", "score")}
        if fields:
            shown["fields"] = _pick(hit["meta"], fields)
        if not compact:
            shown["text"] = hit["text"]
        if hit.get("also"):
            shown["also"] = hit["also"]
        if hit.get("partial"):
            shown["partial"] = True
        out.append(shown)
    return out


DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:")


def _doi(meta):
    """A document's DOI, from its front-matter field `doi`, as DOIs compare: in any case, without the prefix of a link
    or "doi:"; None if it has none, or one that is no DOI (they all start with "10.")."""
    value = json.loads(meta).get("doi") if meta else None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    for prefix in DOI_PREFIXES:
        value = value.removeprefix(prefix)
    return value if value.startswith("10.") else None


def _one_per_work(hits):
    """Ranked hits, one per work: a hit with the DOI of a better ranked one goes, as several collections hold the same
    paper, and the better ranked one names it under "also"."""
    kept, first = [], {}
    for hit in hits:
        doi = _doi(hit["meta"])
        if doi is None or doi not in first:
            kept.append(hit)
            if doi is not None:
                first[doi] = hit
        else:
            first[doi].setdefault("also", []).append(hit["id"])
    return kept


def line_span(lines):
    """First and last line of a range such as "17-26", or of a single line: "17" or 17."""
    if isinstance(lines, int) and not isinstance(lines, bool):
        lines = str(lines)
    m = isinstance(lines, str) and re.fullmatch(r"([1-9]\d*)(?:-([1-9]\d*))?", lines)
    if not m or int(m[2] or m[1]) < int(m[1]):
        raise ValueError(f"lines must be a line number or a range such as '17-26', got {lines!r}")
    return int(m[1]), int(m[2] or m[1])


ADAPTIVE_SHARE = 0.9  # of the gaps between neighbouring scores, the first, where the largest is looked for
ADAPTIVE_BUFFER = 5  # hits kept after the largest gap, as the paper keeps: of 10 or 50 vsearch hits for 2,000 arXiv
# titles, 6 on average, with the paper sought in 99.6% of the cases it was among them (docs/benchmarks.md)


def _adaptive(hits):
    """Hits, best first, up to the largest drop in score among the first ADAPTIVE_SHARE of their gaps, and
    ADAPTIVE_BUFFER more: Adaptive-k (Taguchi et al. 2025, arXiv 2506.08479), the few that stand out instead of k."""
    if len(hits) < 3:
        return hits
    scores = np.array([hit["score"] for hit in hits], np.float64)
    gaps = scores[:-1] - scores[1:]
    return hits[: int(np.argmax(gaps[: math.ceil(ADAPTIVE_SHARE * len(gaps))])) + 1 + ADAPTIVE_BUFFER]


def _check_budget(budget):
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int | float) or not budget > 0):
        raise ValueError(f"budget must be a positive number of seconds, got {budget!r}")
    return budget


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


QUOTED = r'"(?:[^"]|"")*"'
PHRASE = re.compile(rf"(?:{QUOTED}|[^\s()\"*]+)(?:\s*\*)?")  # a quoted phrase or a bare word, prefix or not
PLACED = re.compile(r"NEAR\s*\(|[{}+^:]")  # operators that join phrases or tie them to columns or places
MARK = "\x01"  # what highlight() puts before each phrase it finds, to count them
MORE_CANDIDATES = 20  # each collection gives this many more of its best, or as many again, to be scored again


def _phrases_of(match):
    """The phrases of an FTS5 query that is phrases and AND, OR, NOT and parentheses alone, each as a query of its own;
    None for another, such as one with NEAR or a column filter."""
    if PLACED.search(re.sub(QUOTED, " ", match)):
        return None
    return [phrase for phrase in PHRASE.findall(match) if phrase not in ("AND", "OR", "NOT")]


def _totals(db):
    """The rows of a store's keyword index, and the tokens they hold in all: FTS5's record of them."""
    row = db.execute("SELECT block FROM fts_data WHERE id = 1").fetchone()
    numbers = postings.varints(row[0])[0].tolist() if row else [0]
    return numbers[0], sum(numbers[1:])


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
    _, _, field, line, kind, target, _, fragment, shown = row
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
    for _, _, field, line, _, target, *_ in rows:
        if line not in found:
            content = every[line - 1].strip()
            if len(content) > LINE_CHARS:
                start = max(0, content.find(target) - LINE_CHARS // 3)
                end = start + LINE_CHARS
                content = ("…" if start else "") + content[start:end] + ("…" if end < len(content) else "")
            found[line] = {"line": line} | ({"field": field} if field else {}) | {"text": content}
    return list(found.values())


def _by_value(rows, descending):
    """Rows of several stores, (collection, path, ..., value), each store's in SQLite's order by value and path,
    merged in that order: numbers before text, ascending or descending, and rows without a value last; then by
    collection and path."""
    rows = sorted(rows, key=itemgetter(0, 1))
    valued = [row for row in rows if row[-1] is not None]
    valued.sort(key=lambda row: (0, row[-1]) if isinstance(row[-1], int | float) else (1, row[-1]), reverse=descending)
    return valued + [row for row in rows if row[-1] is None]


def _codes(db):
    """The codes of a store's vectors (see store.read_codes) to search; None if it has none."""
    found = store.read_codes(db)
    return Codes(*found) if found else None


def _size(path):
    """The bytes a store takes on disk, its write-ahead log included."""
    return sum(os.path.getsize(file) for file in (path, f"{path}-wal") if os.path.isfile(file))


def _postings_state(db):
    """How a collection's keyword postings stand, for a status: "ready", "not built" or "N chunks behind" (searches
    read FTS5 meanwhile, slower; `sift index` builds or updates them)."""
    if not postings.built(db):
        return "not built"
    behind = postings.behind(db)
    return f"{console.count(behind, 'chunk')} behind" if behind else "ready"


class _Filter(Filter):
    """Which chunks of the collections of `scope` a search may find: those of the documents (`d`) that `where`, with
    `args`, lets through. What `check` was asked is remembered, as a semantic search asks again when it looks further;
    `passes` asks the stores each time, for a keyword search, which never asks twice."""

    CHUNKS = "SELECT c.id FROM chunks c JOIN docs d ON d.id = c.doc WHERE "

    def __init__(self, catalog, scope, where, args):
        self.stores, self.where, self.args = {c.id: catalog.store(c) for c in scope}, where, args
        self.seen, self.passed = np.empty(0, np.int64), np.empty(0, bool)  # the chunks checked, by id, and which pass
        self._listed = None

    def check(self, ids):
        ids = np.asarray(ids, np.int64)
        if len(new := np.setdiff1d(ids, self.seen)):
            seen = np.concatenate([self.seen, new])
            order = np.argsort(seen, kind="stable")
            self.seen, self.passed = seen[order], np.concatenate([self.passed, self.passes(new)])[order]
        return self.passed[np.searchsorted(self.seen, ids)]

    def passes(self, ids):
        """Which of these chunks, of the collections of the scope, pass, as their stores tell."""
        sql = f"{self.CHUNKS}c.id IN (SELECT value FROM json_each(?)) AND {self.where}"
        owners, passed = ids >> store.SHIFT, []
        for cid in np.unique(owners).tolist():
            rows = self.stores[cid].execute(sql, [json.dumps(ids[owners == cid].tolist()), *self.args])
            passed += [chunk for (chunk,) in rows]
        return np.isin(ids, np.array(passed, np.int64))

    def listed(self):
        if self._listed is None:
            rows = (i for db in self.stores.values() for (i,) in db.execute(self.CHUNKS + self.where, self.args))
            self._listed = np.fromiter(rows, np.int64)
        return self._listed


class _Resolvers(dict):
    """A links.Resolver for each collection, made when first needed."""

    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    def __missing__(self, collection):
        self[collection] = resolver = links.Resolver(self.catalog.store(collection))
        return resolver


class Index:
    """Search over the active collections of a catalog (default: SIFT_HOME's). Semantic search uses `models` (default:
    `models.default()`, chosen when first needed, so keyword search loads no models)."""

    def __init__(self, catalog=None, models=None):
        self.catalog = catalog or Catalog()
        self._models = models
        self._wrap = None
        self.vectors = {}  # collection id: (its store's version, the Codes of its vectors)
        self._postings = {}  # collection id: (the state of its keyword postings, their postings.Reader)
        self._said = set()  # warnings given, each once

    @property
    def models(self):
        if self._models is None:
            self._models = _models.default()
            if self._wrap:
                self._models = self._wrap(self._models)
        return self._models

    def models_through(self, wrap):
        """Have the models used through wrap(models), e.g. jobs.OnMainThread: now if chosen, else once they are."""
        self._wrap = wrap
        if self._models is not None:
            self._models = wrap(self._models)

    def _warn(self, text):
        if text not in self._said:
            self._said.add(text)
            console.say(f"warning: {text}")

    # --- which collections and documents a search may find -----------------------

    def active(self):
        """The active collections whose stores are there, by id; the codes of others are let go."""
        found = {}
        for c in self.catalog.collections():
            if not c.active:
                self.catalog.release(c.id)
            elif os.path.isfile(c.store):
                found[c.id] = c
            else:
                self._warn(f"{c.name}: its index is missing: {c.store}; searches pass it over")
        for cid in self.vectors.keys() - found.keys():
            del self.vectors[cid]
        return found

    def _scope(self, collection):
        """The collections a search covers: the one named, or every active one."""
        if not collection:
            return list(self.active().values())
        c = self.catalog.get(collection)
        self._readable(c)
        return [c]

    def _readable(self, collection):
        """The store of a collection to read from; an error if the collection is inactive, or its store missing."""
        if not collection.active:
            raise LookupError(
                f"collection {collection.name!r} is inactive: "
                f"`{console.command('sift', 'activate', collection.name)}` to search it"
            )
        return self.catalog.store(collection)

    def _owner(self, row_id):
        """The collection a document or chunk belongs to, by its row id."""
        cid = store.owner(row_id)
        if found := next((c for c in self.catalog.collections() if c.id == cid), None):
            return found
        raise LookupError("its collection was removed meanwhile")

    def _db(self, row_id):
        """The store holding a document or chunk, by its row id."""
        return self.catalog.store(self._owner(row_id))

    def _by_store(self, ids):
        """Row ids of documents or chunks by the collection they belong to, of those still there: (collection, its
        store, [ids]) each."""
        known, groups = {c.id: c for c in self.catalog.collections()}, {}
        for i in ids:
            groups.setdefault(store.owner(i), []).append(int(i))
        return [(known[cid], self.catalog.store(known[cid]), group) for cid, group in groups.items() if cid in known]

    def _where(self, collection=None, path=None, filters=None, links_to=None, linked_from=None):
        """The collections to search, and an SQL condition on their docs `d` with its arguments. Filters match
        front-matter fields: a value, which a list field matches if it contains it, a list of values for any of them,
        or conditions such as {">=": "2024-06", "<": "2025"}, which one value or list element must satisfy, and
        {"not": value or list}, which none may match. A value that is a wikilink, "[[Acme AG]]", matches a link to the
        same note however it is written. `links_to` and `linked_from` name notes (see _notes): only documents linking
        one of the first, and linked from one of the second."""
        scope, sql, args, resolvers = self._scope(collection), ["1"], [], _Resolvers(self.catalog)
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
                if list(conditions) == ["="] and (linked := self._linking(key, conditions["="], scope, resolvers)):
                    condition = f"({condition} OR d.id IN (SELECT value FROM json_each(?)))"
                    args.append(_ids(linked))
                sql.append(condition)
            if "not" in conditions:
                condition, condition_args = _condition(key, "not", conditions["not"])
                sql.append(f"NOT EXISTS (SELECT 1 FROM json_each(d.meta, ?) WHERE {condition})")
                args += [field, *condition_args]
                if linked := self._linking(key, conditions["not"], scope, resolvers):
                    sql.append("d.id NOT IN (SELECT value FROM json_each(?))")
                    args.append(_ids(linked))
        allowed = self._linked(links_to, linked_from, scope, resolvers)
        if allowed is not None:
            sql.append("d.id IN (SELECT value FROM json_each(?))")
            args.append(_ids(allowed))
        return scope, " AND ".join(sql), args

    def _resolve(self, spec, scope, resolvers):
        """Where a note named as links name it leads, from no note in particular, in each collection of `scope`:
        {collection: (how, [doc ids])}."""
        target = _target(spec)
        key = readers.key(target.rsplit("/", 1)[-1])
        return {c: resolvers[c].resolve(None, "wikilink", target, key) for c in scope}

    def _notes(self, specs, what, scope, resolvers):
        """The notes that `specs`, one or a list, name: by a document's id, as for `get`, or as links name them,
        "Acme AG" or "[[Acme AG]]", in the collections of `scope`, wherever one note answers. As {spec: doc row id, or
        None for a note no document answers to}; a name that more than one note answers to is an error."""
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
            ways = self._resolve(spec, scope, resolvers).values()
            matches = {doc for how, docs in ways if how != "unresolved" for doc in docs}
            if len(matches) > 1:
                named = sorted(whole for whole, _ in self._named(matches).values())
                shown = ", ".join(named[:AMBIGUOUS_SHOWN]) + (", ..." if len(named) > AMBIGUOUS_SHOWN else "")
                raise LookupError(f"{spec!r} matches {shown}; give the whole id, or a collection")
            found[spec] = next(iter(matches), None)
        return found

    def _linked(self, links_to, linked_from, scope, resolvers):
        """The documents that link to any of the notes `links_to` names, and that one of `linked_from` links to, by
        row id; None if neither is given. A note no document answers to yet is linked by the links that name it."""
        allowed = None
        if links_to is not None:
            allowed = set()
            for spec, note in self._notes(links_to, "links_to", scope, resolvers).items():
                if note is not None:
                    c = self._owner(note)
                    allowed |= {row[0] for row in links.incoming(self.catalog.store(c), resolvers[c], note)}
                elif docs := {d for c in scope for d in links.written(self.catalog.store(c), _target(spec))}:
                    allowed |= docs
                else:
                    raise LookupError(f"no note {spec!r} in the index, and no link to one")
        if linked_from is not None:
            to = set()
            for spec, note in self._notes(linked_from, "linked_from", scope, resolvers).items():
                if note is None:
                    raise LookupError(f"no note {spec!r} in the index")
                c = self._owner(note)
                outgoing = links.outgoing(self.catalog.store(c), resolvers[c], note)
                to |= {docs[0] for _, (how, docs) in outgoing if how in VIA}
            allowed = to if allowed is None else allowed & to
        return allowed

    def _linking(self, key, value, scope, resolvers):
        """The documents whose front-matter field `key` links to a note that a wikilink among the filter values names,
        however the link is written; links to a note no document answers to, or to several, match as written. None if
        no value is a wikilink."""
        specs, docs = _wikilinks(value), set()
        for spec in specs:
            for c, (how, found) in self._resolve(spec, scope, resolvers).items():
                db = self.catalog.store(c)
                if how in VIA:
                    docs |= {row[0] for row in links.incoming(db, resolvers[c], found[0], key)}
                else:
                    docs |= set(links.written(db, _target(spec), key))
        return docs if specs else None

    # --- candidates as [(chunk_id, score)] -----------------------------------

    def _keyword(self, match, k, scope, where, args, all_passages=False, deadline=None):
        """The best passage of each of the k documents that best match an FTS5 query, by BM25, or the k best
        passages, in the collections of `scope`, scored with the word counts of all of them. A query of terms, AND, OR
        and NOT is answered from the keyword postings (see `topk`), until the topk.Deadline if given, where every
        collection has them ready; others by FTS5, each collection's best scored again together (see _as_one), unless
        the query places its phrases (NEAR, column filters)."""
        if not match.strip():
            return []
        if (query := expression.parse(match)) is not None and (held := self._ready(scope)) is not None:
            allowed = _Filter(self.catalog, scope, where, args).passes if args else None
            return topk.search(held, query, k, by_document=not all_passages, allowed=allowed, deadline=deadline)
        passages = f"""SELECT fts.rowid passage, c.doc doc, -bm25(fts) score FROM fts
                       JOIN chunks c ON c.id = fts.rowid JOIN docs d ON d.id = c.doc WHERE fts MATCH ? AND {where}"""
        if all_passages:
            sql = f"SELECT passage, doc, score FROM ({passages}) ORDER BY score DESC LIMIT ?"
        else:  # MATERIALIZED: SQLite can't take bm25() into an aggregate
            sql = f"""WITH p AS MATERIALIZED ({passages})
                      SELECT passage, doc, max(score) FROM p GROUP BY doc ORDER BY 3 DESC LIMIT ?"""
        # Every passage of a collection's best documents: which is a document's best depends on the word counts too.
        every = f"""WITH p AS MATERIALIZED ({passages}),
                    best AS (SELECT doc FROM p GROUP BY doc ORDER BY max(score) DESC LIMIT ?)
                    SELECT passage, doc, score FROM p WHERE doc IN (SELECT doc FROM best)"""
        for attempt in (match, _phrases(match)):
            phrases = _phrases_of(attempt) if len(scope) > 1 else None
            each = k if not phrases else k + max(k, MORE_CANDIDATES)  # a collection's best may rank otherwise
            query = sql if all_passages or not phrases else every
            try:
                found = {c: self.catalog.store(c).execute(query, [attempt, *args, each]).fetchall() for c in scope}
            except sqlite3.OperationalError as e:
                error = e
                continue
            if phrases:
                found = self._as_one(found, phrases)
            ranked = sorted((row for rows in found.values() for row in rows), key=lambda row: -row[2])
            ranked = ranked if all_passages else _first_of_each(ranked, itemgetter(1))
            return [(passage, score) for passage, _, score in ranked[:k]]
        raise ValueError(f"invalid search query {match!r}: {error}")

    def _ready(self, scope):
        """The keyword postings of the collections of `scope`, one postings.Reader each, or None if any has none
        ready: never built, or behind FTS5 since a change."""
        readers = []
        for c in scope:
            db = self.catalog.store(c)
            if not postings.ready(db):
                return None
            state = postings.Reader.state(db)
            held = self._postings.get(c.id)
            if held is None or held[0] != state or held[1].db is not db:  # changed, or the store opened anew
                held = self._postings[c.id] = (state, postings.Reader(db))
            readers.append(held[1])
        return readers

    def _as_one(self, found, phrases):
        """The passages found in each collection, {collection: [(passage, doc, score)]}, scored again by BM25 as FTS5
        scores them, with the word counts of all these collections together, as one index would. Each collection's own
        would take a word common in a small collection for a common word: a page alone in its collection, about
        benchmarks, sank below the passages of others that only mention them."""
        rows = tokens = 0
        holding = [0] * len(phrases)  # the rows that hold each phrase
        for c in found:
            db = self.catalog.store(c)
            n, t = _totals(db)
            rows, tokens = rows + n, tokens + t
            for i, phrase in enumerate(phrases):
                holding[i] += db.execute("SELECT count(*) FROM fts WHERE fts MATCH ?", (phrase,)).fetchone()[0]
        idf = [math.log((rows - n + 0.5) / (n + 0.5)) for n in holding]
        idf = [value if value > 0 else 1e-6 for value in idf]  # as FTS5 takes them
        average = tokens / rows if rows else 1.0
        highlighted = ", ".join(f"highlight(fts, {column}, '{MARK}', '')" for column in range(3))
        hits = f"SELECT rowid, {highlighted} FROM fts WHERE fts MATCH ? AND rowid IN (SELECT value FROM json_each(?))"
        scored = {}
        for c, passages in found.items():
            db, ids = self.catalog.store(c), json.dumps([passage for passage, _, _ in passages])
            sizes = "SELECT id, sz FROM fts_docsize WHERE id IN (SELECT value FROM json_each(?))"
            size = {i: int(postings.varints(sz)[0].sum()) for i, sz in db.execute(sizes, (ids,))}
            score = dict.fromkeys(size, 0.0)
            for weight, phrase in zip(idf, phrases):
                for passage, *columns in db.execute(hits, (phrase, ids)):
                    n = sum(text.count(MARK) for text in columns if text)
                    score[passage] += weight * float(postings.term_scores(n, size[passage], average))
            scored[c] = [(passage, doc, score.get(passage, 0.0)) for passage, doc, _ in passages]
        return scored

    def load(self):
        """Load the codes of the active collections' vectors (see `vectors`), each store's batch by batch; again when a
        store has changed since."""
        active = self.active()
        if not active and self.catalog.collections():
            raise LookupError("no collection is active: `sift activate NAME` takes one back into searches")
        for c in active.values():
            db = self.catalog.store(c)
            version, held = store.get_meta(db, "version"), self.vectors.get(c.id)
            if held is not None and version is not None and held[0] == version:
                continue
            if (codes_held := _codes(db)) is None:
                self.vectors.pop(c.id, None)
            else:
                self.vectors[c.id] = (version, codes_held)
        if not self.vectors:
            raise LookupError("no embeddings yet, run `sift embed`")

    def _exact(self, chunks, dim):
        """The vectors the stores hold of these chunks, of `dim` dimensions, as (their ids, fp16 rows): those deleted
        meanwhile, and any of another length, are left out."""
        sql = "SELECT chunk, vec FROM vectors WHERE chunk IN (SELECT value FROM json_each(?))"
        rows = [
            row
            for _, db, group in self._by_store(np.asarray(chunks).tolist())
            for row in db.execute(sql, (json.dumps(group),))
            if len(row[1]) == 2 * dim
        ]
        matrix = np.frombuffer(b"".join(vec for _, vec in rows), np.float16).reshape(len(rows), dim)
        return np.array([chunk for chunk, _ in rows], np.int64), matrix

    def _vector(self, query):
        """The query's vector, with the codes loaded to search with it."""
        self.load()
        return self.models.encode([query], query=True)[0].astype(np.float32)

    def _nearest(self, query, k, scope, where, args, all_passages, excluded=None):
        """The best passage of each of the k documents nearest the query's vector, among the documents of the
        collections of `scope` that `where` allows, but the chunks `excluded`; or the k best passages."""
        allowed = _Filter(self.catalog, scope, where, args) if args else None
        held, width = [], codes(query[None, :]).shape[1]
        for c in scope:
            if c.id in self.vectors and (theirs := self.vectors[c.id][1]).width != width:
                self._warn(f"{c.name} was embedded with another model: `sift embed --reset` embeds it with this one")
            elif c.id in self.vectors:
                held.append(theirs)

        def exact(ids):
            return self._exact(ids, len(query))

        if all_passages:
            return top(held, query, k, exact, allowed, excluded)
        n = k
        while True:  # when the best passages come from a few documents, look further for others
            ranked = top(held, query, n, exact, allowed, excluded)
            best = self._per_document(ranked)
            if len(best) >= k or len(ranked) < n:
                return best[:k]
            n *= 4

    def _per_document(self, ranked):
        """The first, best, of each document's passages in `ranked`."""
        sql, docs = "SELECT c.id, c.doc FROM json_each(?) j JOIN chunks c ON c.id = j.value", {}
        for _, db, group in self._by_store(chunk for chunk, _ in ranked):
            docs |= dict(db.execute(sql, (json.dumps(group),)))
        return _first_of_each((r for r in ranked if r[0] in docs), lambda r: docs[r[0]])  # in docs: not removed

    def _works(self, rank, k, all_passages):
        """The hits of the best k works by rank(n), the best n documents (or with `all_passages`, passages, of which
        all are kept): one hit per work, as _one_per_work leaves them, asking for more while copies leave fewer."""
        if all_passages:
            return self._hits(rank(k))
        n = k
        while True:
            ranked = rank(n)
            hits = _one_per_work(self._hits(ranked))
            if len(hits) >= k or len(ranked) < n:
                return hits[:k]
            n *= 2

    def _hits(self, ranked):
        sql = """SELECT d.path, d.title, d.meta, c.line_start, c.line_end, substr(d.text, c.pos + 1, c.len)
                 FROM chunks c JOIN docs d ON d.id = c.doc WHERE c.id = ?"""
        known, hits = {c.id: c for c in self.catalog.collections()}, []
        for chunk, score in ranked:
            c = known.get(store.owner(chunk))
            row = c and self.catalog.store(c).execute(sql, (chunk,)).fetchone()
            if not row:  # removed by a concurrent `sift index`
                continue
            path, title, meta, start, end, text = row
            hit = {"id": f"{c.name}/{path}", "title": title, "lines": f"{start}-{end}", "score": round(score, 4)}
            hits.append(hit | {"meta": meta, "text": text, "chunk": chunk})  # for _present, and `query`'s scores
        return hits

    def _cosines(self, query, chunks):
        """The cosines of these chunks with the query's vector, exactly; one not embedded yet, or broken, gets the
        others' mean, neither a bonus nor a penalty."""
        found, rows = self._exact(np.array(chunks, np.int64), len(query))
        exact = dict(zip(found.tolist(), (rows.astype(np.float32) @ query).tolist()))
        cosine = np.array([exact.get(chunk, np.nan) for chunk in chunks], np.float64)
        cosine[~np.isfinite(cosine)] = np.nan
        known = np.isfinite(cosine)
        return np.where(known, cosine, np.nanmean(cosine) if known.any() else 0.0)

    # --- public API ----------------------------------------------------------

    # Every search returns the best passage of each document it finds, or with `all_passages` every passage, and
    # takes `fields`, front-matter fields to add to each hit (["*"] for all), `compact`, which leaves out the text,
    # and `sort`, a front-matter field to order the hits by instead of score ("-created": newest first). With
    # `adaptive`, k is the most hits returned: those before the largest drop in score, and a few more (see _adaptive).
    # Without a `collection`, a search covers every active one.

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
        adaptive=False,
        budget=None,
    ):
        """Exact keyword search, ranked by BM25. Accepts FTS5 syntax: "phrases", prefix*, AND, OR, NOT.
        If no chunk holds all terms of a query without operators, chunks with any of them are returned. With a
        `budget` in seconds, a search the keyword postings answer stops once it is spent and has found any, and returns
        the best found so far, each hit marked "partial"."""
        k, (scope, where, args) = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
        fields, sort, budget = _check_fields(fields), _check_sort(sort), _check_budget(budget)
        deadline = topk.Deadline(budget) if budget is not None else None

        def rank(n):
            ranked = self._keyword(query, n, scope, where, args, all_passages, deadline)
            if not ranked and (relaxed := _any_term(query)):
                ranked = self._keyword(relaxed, n, scope, where, args, all_passages, deadline)
            return ranked

        hits = self._works(rank, k, all_passages)
        for hit in hits if deadline is not None and deadline.cut else ():
            hit["partial"] = True
        return _present(_adaptive(hits) if adaptive else hits, fields, compact, sort)

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
        adaptive=False,
    ):
        """Semantic search: the passages nearest the query's embedding, by cosine similarity (see `vectors`)."""
        k, (scope, where, args) = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
        fields, sort = _check_fields(fields), _check_sort(sort)
        if not query.strip():  # the embedding of nothing is close to arbitrary chunks
            return []
        vector = self._vector(query)
        hits = self._works(lambda n: self._nearest(vector, n, scope, where, args, all_passages), k, all_passages)
        return _present(_adaptive(hits) if adaptive else hits, fields, compact, sort)

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
        adaptive=False,
        candidates=40,
    ):
        """Hybrid search for RAG: keyword and semantic candidates, all reranked together, and ordered by the reranker's
        score blended with their cosine (see _blend). Fusing the candidates' first ranks into the order, e.g. by
        reciprocal rank, made results worse (MRR@10 0.69 to 0.64 on a known-item test over 3.2 million arXiv
        abstracts); the blend made them better on six of the seven sets in docs/benchmarks.md, and cost a little on
        those arXiv known items (0.937 to 0.929)."""
        k, (scope, where, args) = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
        fields, sort = _check_fields(fields), _check_sort(sort)
        candidates = max(candidates, k)
        keyword = self._keyword(_any_word(query), candidates, scope, where, args, all_passages)
        vector = self._vector(query) if query.strip() else None  # for the semantic candidates and every cosine
        semantic = self._nearest(vector, candidates, scope, where, args, all_passages) if vector is not None else []
        hits = self._hits([(chunk, 0.0) for chunk in dict.fromkeys(chunk for chunk, _ in keyword + semantic)])
        if not hits:
            return []
        asked = set(_words(query))
        named = [_named(h, asked) for h in hits]
        rerank = np.asarray(self.models.rerank(query, [_passage(h, n) for h, n in zip(hits, named)]), np.float64)
        scores = _blend(rerank, self._cosines(vector, [h["chunk"] for h in hits]), np.array([bool(n) for n in named]))
        for h, s in zip(hits, scores):
            h["score"] = round(float(s), 4)
        hits = sorted(hits, key=lambda h: -h["score"])
        hits = (hits if all_passages else _one_per_work(_first_of_each(hits, itemgetter("id"))))[:k]
        return _present(_adaptive(hits) if adaptive else hits, fields, compact, sort)

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
        adaptive=False,
    ):
        """Documents closest in meaning to the document `id` (as for `get`), by cosine similarity with the mean of its
        passages' vectors; the document itself is left out."""
        k, (scope, where, args) = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
        fields, sort = _check_fields(fields), _check_sort(sort)
        doc = self._find(id)[0]
        self.load()
        sql = "SELECT v.chunk, v.vec FROM vectors v JOIN chunks c ON c.id = v.chunk WHERE c.doc = ?"
        own = self._db(doc).execute(sql, (doc,)).fetchall()
        if not own:
            raise LookupError(f"{id!r} has no embeddings yet, run `sift embed`")
        mean = np.mean([np.frombuffer(vec, np.float16) for _, vec in own], axis=0, dtype=np.float32)
        excluded, vector = np.array([chunk for chunk, _ in own], np.int64), mean / np.linalg.norm(mean)
        hits = self._works(
            lambda n: self._nearest(vector, n, scope, where, args, all_passages, excluded), k, all_passages
        )
        return _present(_adaptive(hits) if adaptive else hits, fields, compact, sort)

    def _place(self, id):
        """The collection, and the path in it, that `id` names: "collection/path", or an absolute path in a
        collection's folder, the nearest that holds it; (None, path) for a collection there isn't, (None, "") for an
        absolute path in none."""
        collections = self.catalog.collections()
        if not os.path.isabs(id):
            name, _, rel = id.partition("/")
            return next((c for c in collections if c.name == name), None), rel
        full = os.path.normpath(id)
        for c in sorted(collections, key=lambda c: -len(c.path)):
            root = os.path.normpath(c.path)
            if os.path.normcase(full).startswith(os.path.normcase(os.path.join(root, ""))):
                return c, os.path.relpath(full, root).replace(os.sep, "/")
        return None, ""

    def _rows(self, id):
        """The documents `id` may name (see `get`), at most six: (row id, path, title, meta, text) of each."""
        collection, rel = self._place(id)
        if collection is None:
            return []
        db, sql = self._readable(collection), "SELECT id, path, title, meta, text FROM docs WHERE "
        rows = db.execute(sql + "path = ?", (rel,)).fetchall()
        if rows or not rel:
            return rows
        if os.path.isabs(id):  # in another case, as Windows and macOS name the same file, and `reindex` takes it
            found = db.execute(sql + "lower(path) = lower(?)", (rel,)).fetchall()
            same = [row for row in found if readers.fold(row[1]) == readers.fold(rel)]
            return same if len(same) == 1 else []
        # e.g. arxiv/2603.29315.md for arxiv/2026/03/2603.29315.md
        return db.execute(sql + "substr(path, -?) = ? ORDER BY path LIMIT 6", (len(rel) + 1, "/" + rel)).fetchall()

    def _note(self, id):
        """The document `id` names, as for `get`, or else the note `id` names as links name it, "Acme AG" or
        "[[Acme AG]]", if one note answers to it in all active collections: as _find gives it."""
        try:
            return self._find(id)
        except LookupError as unknown:
            note = self._notes([id], "id", self._scope(None), _Resolvers(self.catalog))[id]
            if note is None:
                raise unknown from None
            return self._find(self._named([note])[note][0])

    def _find(self, id):
        """The document `id` names (see `get`), as (row id, whole id, title, meta, text)."""
        collection, rows = self._place(id)[0], self._rows(id)
        if len(rows) > 1:
            matches = ", ".join(f"{collection.name}/{row[1]}" for row in rows[:5]) + (", ..." if len(rows) > 5 else "")
            raise LookupError(f"{id!r} matches {matches}; give the whole id")
        if not rows:
            raise LookupError(f"unknown document {id!r}")
        doc, path, title, meta, text = rows[0]
        return doc, f"{collection.name}/{path}", title, meta, text

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
        """Documents without a query, by collection and path: what a collection, folder or filter holds. They come as
        hits without lines and score, with the start of their first passage, after any front matter, as their
        passage; `sort` orders them by a front-matter field instead, as in searches."""
        k, (scope, where, args) = _check_k(k), self._where(collection, path, filters, links_to, linked_from)
        fields, sort = _check_fields(fields), _check_sort(sort)
        value, order, by = "NULL", "d.path", []
        if sort:
            value = "json_extract(d.meta, ?)"
            order = f"{value} IS NULL, {value}{' DESC' if sort[0] == '-' else ''}, {order}"
            by = [_field(sort.removeprefix("-"))]
        # The first chunk's start, after any front matter; as a subquery, not a join, so that sorting a whole collection
        # doesn't join every document first (0.4 s, against 5 s, for 3 million).
        first = f"SELECT substr(d.text, c.pos + 1, min(c.len, {START})) FROM chunks c WHERE c.doc = d.id AND c.seq = 0"
        sql = f"""SELECT d.path, d.title, d.meta, coalesce(({first}), ''), {value} FROM docs d WHERE {where}
                  ORDER BY {order} LIMIT ?"""
        rows = [(c.name, *row) for c in scope for row in self.catalog.store(c).execute(sql, [*by, *args, *by * 2, k])]
        rows = _by_value(rows, sort[0] == "-") if sort else rows  # else in order: the collections by name, each by path
        docs = []
        for name, doc_path, title, meta, text, _ in rows[:k]:
            doc = {"id": f"{name}/{doc_path}", "title": title}
            if fields:
                doc["fields"] = _pick(meta, fields)
            docs.append(doc if compact else doc | {"text": text})
        return docs

    def folders(self, collection, path=""):
        """The folders directly inside the folder `path` ("" or e.g. "2024/") of a collection, by name. Each folder
        is one jump through the index, so a folder holding millions of documents costs as little as an empty one."""
        path = path.strip("/") + "/" if path.strip("/") else ""  # "2024", "/2024/": the folder 2024/
        db, found, after = self._readable(self.catalog.get(collection)), [], path
        sql = "SELECT path FROM docs WHERE path > ? AND path < ? ORDER BY path LIMIT 1"
        while row := db.execute(sql, (after, path + LAST)).fetchone():
            name, slash, _ = row[0][len(path) :].partition("/")
            if slash:
                found.append(name)
            after = path + name + (slash and "/" + LAST)  # past the folder, or past the document
        return sorted(found)

    def file(self, id):
        """The file a document was read from, by its id as for `get`; None for a JSON Lines record, which is a part
        of its file."""
        doc = self._find(id)[0]
        collection = self._owner(doc)
        sql = "SELECT s.path, d.path FROM docs d JOIN sources s ON s.id = d.source WHERE d.id = ?"
        source, path = self.catalog.store(collection).execute(sql, (doc,)).fetchone()
        return Path(collection.path) / source if source == path else None

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
        db = self._db(doc)
        resolver, found = links.Resolver(db), {"id": whole, "title": title}
        if direction != "in":
            found["out"] = self._out(db, resolver, doc, depth, field, k)
        if direction != "out":
            found["in"] = self._in(db, resolver, doc, depth, field, k)
        return found

    def _out(self, db, resolver, doc, depth, field, k):
        """The links out of a document, and with depth 2, the documents they lead to link: (row, how, through)."""
        first = links.outgoing(db, resolver, doc, field)
        found = [(row, how, None) for row, how in first]
        if depth == 2:
            reached = [docs[0] for _, (via, docs) in first if via in VIA]
            seen = {doc, *reached}
            for step in dict.fromkeys(reached):
                for row, (via, docs) in links.outgoing(db, resolver, step, field):
                    if via in VIA and docs[0] not in seen:
                        seen.add(docs[0])
                        found.append((row, (via, docs), step))
        found = found[:k]
        named = self._named([d for _, (_, docs), through in found for d in [*docs, through] if d])
        return [_link_out(row, how, through, named) for row, how, through in found]

    def _in(self, db, resolver, doc, depth, field, k):
        """The documents linking a document, and with depth 2, those linking them: (doc, its links, through)."""
        first = _by_document(links.incoming(db, resolver, doc, field))
        found = [(source, rows, None) for source, rows in first.items()]
        if depth == 2:
            seen = {doc, *first}
            for step in first:
                for source, rows in _by_document(links.incoming(db, resolver, step, field)).items():
                    if source not in seen:
                        seen.add(source)
                        found.append((source, rows, step))
        found = found[:k]
        named = self._named([d for source, _, through in found for d in (source, through) if d])
        sql = "SELECT id, text FROM docs WHERE id IN (SELECT value FROM json_each(?))"
        texts = dict(db.execute(sql, (json.dumps([source for source, _, _ in found]),)))
        return [
            {"id": named[source][0], "title": named[source][1], "links": _lines_of(rows, texts[source])}
            | ({"through": named[through][0]} if through else {})
            for source, rows, through in found
        ]

    def link_count(self, id):
        """How many documents link to the document `id`, and how many documents it links to: (in, out)."""
        doc = self._note(id)[0]
        db = self._db(doc)
        resolver = links.Resolver(db)
        into = {row[0] for row in links.incoming(db, resolver, doc)}
        out = {docs[0] for _, (via, docs) in links.outgoing(db, resolver, doc) if via in VIA}
        return len(into), len(out)

    def _named(self, docs):
        """The whole id and title of documents, by row id."""
        sql, named = "SELECT id, path, title FROM docs WHERE id IN (SELECT value FROM json_each(?))", {}
        for c, db, group in self._by_store(set(docs)):
            named |= {doc: (f"{c.name}/{path}", title) for doc, path, title in db.execute(sql, (json.dumps(group),))}
        return named

    def unresolved(self, collection=None, k=50):
        """Link targets no document answers to, in a collection or in every active one: notes not written yet. The
        most linked first, each with how many documents link it and the ids of a few of them."""
        k, scope = _check_k(k), self._scope(collection)
        found = [target for c in scope for target in links.unresolved(self.catalog.store(c))]
        found = sorted(found, key=lambda t: (-len(t[1]), t[0]))[:k]
        named = self._named([doc for _, docs, _ in found for doc in docs])
        return [
            {"target": target, "docs": len(docs), "ids": sorted(named[doc][0] for doc in docs)[:LINKING_SHOWN]}
            for target, docs, _ in found
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
        scope, where, args = self._where(collection)
        db = self._db(doc)
        linking = {row[0] for row in links.incoming(db, links.Resolver(db), doc)} | {doc}
        where, args = where + " AND d.id NOT IN (SELECT value FROM json_each(?))", [*args, _ids(linking)]
        match = " OR ".join('"' + phrase.replace('"', '""') + '"' for phrase in phrases.values())
        return _present(self._hits(self._keyword(match, k, scope, where, args)), fields, compact, None)

    # --- reading files again (see indexing.reindex) ---------------------------

    def reindex(self, paths):
        """Read files again now, after they were written, changed or deleted, and embed their new passages if their
        collection is embedded: the next search finds them as they are, the codes in memory following without a full
        load. Per file: {"id", "status"}, see indexing.reindex, with "chunks", its documents' passages, and
        "embedded", how many of them have vectors; and a "note" on anything that went wrong. Scans are left for `sift
        index` to read with OCR: a server runs one call at a time, which reading one would hold up for minutes."""
        done = indexing.reindex(self.catalog, paths, scans=False)
        for cid, (before, after) in done.versions.items():
            removed = [chunk for chunk in done.removed if store.owner(chunk) == cid]
            self._patch(cid, before, after, removed=removed)
        note = self._embed_new([doc for _, docs in done.files for doc in docs])
        sql = """SELECT count(c.id), count(v.chunk) FROM chunks c LEFT JOIN vectors v ON v.chunk = c.id
                 WHERE c.doc IN (SELECT value FROM json_each(?))"""
        results = []
        for entry, docs in done.files:
            if entry["status"] in ("added", "changed", "unchanged"):
                chunks, embedded = self._db(docs[0]).execute(sql, (json.dumps(docs),)).fetchone() if docs else (0, 0)
                entry |= {"chunks": chunks, "embedded": embedded}
                if note and embedded < chunks:
                    entry["note"] = "; ".join(filter(None, [entry.get("note"), note]))
            results.append(entry)
        return results

    def _embed_new(self, docs):
        """Embed the passages of these documents that have no vector yet, in each collection already embedded, and add
        their codes to those in memory. Returns why some weren't, or None."""
        notes = [self._embed_new_in(c, db, group) for c, db, group in self._by_store(docs)]
        return "; ".join(dict.fromkeys(filter(None, notes))) or None

    def _embed_new_in(self, collection, db, docs):
        model = store.get_meta(db, "model")
        if model is None:  # never embedded: keyword search alone, and no model to load
            return None
        sql = """SELECT c.id, c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM chunks c JOIN docs d ON d.id = c.doc
                 WHERE c.doc IN (SELECT value FROM json_each(?))
                 AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id) ORDER BY c.id"""
        rows = db.execute(sql, (json.dumps(docs),)).fetchall()
        if not rows:
            return None
        try:
            if (using := self.models.name) != model:
                return (
                    f"not embedded: {collection.name} was embedded with {model}; run `sift embed --reset` for {using}"
                )
            matrix = embed.encode(self.models, [embed.text_of(*row[1:]) for row in rows])
        except ConnectionError as e:  # a worker that doesn't answer: keyword search finds them meanwhile
            return f"not embedded: {e}"
        intact = np.isfinite(matrix).all(axis=1)
        db.execute("BEGIN IMMEDIATE")
        try:
            before = store.get_meta(db, "version")
            sql = "SELECT id FROM chunks WHERE id IN (SELECT value FROM json_each(?))"  # not removed meanwhile
            still = {c for (c,) in db.execute(sql, (json.dumps([row[0] for row in rows]),))}
            keep = [i for i, row in enumerate(rows) if intact[i] and row[0] in still]
            store.put_vectors(db, [rows[i][0] for i in keep], matrix[keep])
            store.touch(db)
            after = store.get_meta(db, "version")
            db.commit()
        except BaseException:
            db.rollback()
            raise
        self._patch(collection.id, before, after, ids=[rows[i][0] for i in keep], matrix=matrix[keep])
        return None if intact.all() else "some passages could not be embedded (the model returned NaN); try again"

    def _patch(self, cid, before, after, removed=(), ids=(), matrix=None):
        """Bring the codes in memory of the collection `cid` from its store's version `before` to `after`, a change this
        index made itself, without loading them all again. Codes of another version are loaded anew by the next
        search anyway."""
        held = self.vectors.get(cid)
        if held is None or held[0] != before or before == after:
            return
        bits = codes(matrix) if matrix is not None else np.empty((0, held[1].width), np.uint8)
        self.vectors[cid] = (after, held[1].patched(list(removed), list(ids), bits))

    def fields(self, collection):
        """The front-matter fields of a collection, from FIELD_SAMPLE documents spread from its first to its last,
        each looked up by id: reading every document for its front matter took a large collection minutes."""
        db = self.catalog.store(collection)
        first, last = db.execute("SELECT (SELECT min(id) FROM docs), (SELECT max(id) FROM docs)").fetchone()
        if first is None:
            return []
        steps = max(FIELD_SAMPLE - 1, 1)
        ids = sorted({first + (last - first) * i // steps for i in range(steps + 1)})
        sql = """SELECT DISTINCT j.key FROM json_each(?) p,
                 json_each((SELECT meta FROM docs WHERE id >= p.value ORDER BY id LIMIT 1)) j ORDER BY j.key"""
        return [key for (key,) in db.execute(sql, (json.dumps(ids),))]

    def status(self):
        """The catalog, the model the active collections are embedded with, and every collection: its folder, whether
        it is active, its store and the bytes it takes, or that it is missing; and of an active one, its documents,
        chunks, those embedded, how its keyword postings stand, front-matter fields and links."""
        collections, models = [], set()
        for c in self.catalog.collections():
            entry = {"name": c.name, "path": c.path, "active": c.active, "store": c.store}
            if not os.path.isfile(c.store):
                collections.append(entry | {"missing": True})
                continue
            entry["size"] = _size(c.store)
            if c.active:
                db = self.catalog.store(c)
                counted = store.counts(db)
                linked, unresolved = self.link_counts(c)
                entry |= {"docs": counted["docs"], "chunks": counted["chunks"], "embedded": counted["vectors"]}
                entry["keywords"] = _postings_state(db)
                entry["fields"] = self.fields(c)
                entry |= {"links": linked, "unresolved": unresolved}
                models.add(store.get_meta(db, "model"))
            collections.append(entry)
        return {
            "index": str(self.catalog.path),
            "model": ", ".join(sorted(filter(None, models))) or None,
            "collections": collections,
        }

    def link_counts(self, collection):
        """How many links the notes of a collection hold, and how many of them are unresolved."""
        db = self.catalog.store(collection)
        return db.execute("SELECT count(*) FROM links").fetchone()[0], sum(n for *_, n in links.unresolved(db))
