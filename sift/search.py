"""Search: exact keyword (BM25), semantic (codes of the vectors, the nearest scored exactly), hybrid with reranking,
and documents like one.

The models embed queries and rerank, wherever they run (see `models`). Through a worker, PyTorch never loads: a search
starts in a fraction of a second, even on a light notebook."""

import copy
import json
import math
import os
import posixpath
import re
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Sequence
from concurrent.futures import Future, wait
from functools import partial, wraps
from operator import itemgetter
from pathlib import Path
from typing import Any, Concatenate, Literal, NamedTuple, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from . import console, embed, expression, indexing, links, postings, queries, readers, store, topk
from . import models as _models
from .catalog import Catalog, Collection
from .filters import comparison, conditions, each, json_path
from .hits import (
    MOST_HITS,
    Hit,
    Options,
    check_budget,
    check_fields,
    check_k,
    check_sort,
    doi_of,
    line_span,
    one_per_work,
    pick,
    present,
)
from .vectors import Codes, Filter, Floats, IdList, Ids, Mask, codes, top

_T = TypeVar("_T")
Db = store.Connection
Row = links.Row
Resolution = links.Resolution
Resolver = links.Resolver
Scope = list[Collection]  # the collections a search covers
Where = tuple[Scope, str, list[Any]]  # them, an SQL condition on their documents `d`, and its arguments
Ranked = list[tuple[int, float]]  # (chunk id, score), best first
Passages = list[tuple[int, int, float]]  # (chunk id, document id, score) of each
Named = dict[int, tuple[str, str]]  # documents' whole ids and titles, by row id
Doc = tuple[int, str, str, str, str]  # a document: (row id, whole id, title, meta, text)
State = tuple[tuple[int, int], tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]  # see Index._state

FIELD_SAMPLE = 1000  # documents per collection whose front matter tells which fields exist
START = 600  # characters of a document's start that `browse` shows, as its passage
QUERY_VECTORS = 256  # questions whose embeddings an index keeps, 3 KB each: an agent asks one again, with other options
RESULTS = 32  # searches whose results an index keeps, while what they read stays as it was
KEPT_HITS = 1000  # hits of a search at most for its results to be kept: 10,000 with their passages take about 20 MB


MOST_QUERIES = 20  # queries one search call takes


def _reachable(db: Db) -> bool:
    """Whether an open store still answers: not when its drive went away, or a network share dropped."""
    try:
        store.changes(db)
    except sqlite3.Error:
        return False
    return True


def _target(spec: str) -> str:
    """The target of a note named as links name it: `Acme AG`, `[[Acme AG|Acme]]`, `[[02_Organizations/Acme AG]]`."""
    spec = spec.strip()
    if readers.WIKILINK_VALUE.fullmatch(spec):
        spec = spec[2:-2].partition("|")[0].removesuffix("\\").partition("#")[0].strip()
    return spec


def _wikilinks(value: Any) -> list[str]:
    """The filter values among `value`, one or a list, that are wikilinks."""
    return [v for v in each(value) if isinstance(v, str) and readers.WIKILINK_VALUE.fullmatch(v.strip())]


def _ids(ids: Iterable[int]) -> str:
    """Document row ids as one SQL argument, for `d.id IN (SELECT value FROM json_each(?))`."""
    return json.dumps(sorted(ids))


MORE_CANDIDATES = 20  # each collection gives this many more of its best, or as many again, to be scored again


def _totals(db: Db) -> tuple[int, int]:
    """The rows of a store's keyword index, and the tokens they hold in all: FTS5's record of them."""
    row = db.execute("SELECT block FROM fts_data WHERE id = 1").fetchone()
    numbers: list[int] = postings.varints(row[0])[0].tolist() if row else []
    return (numbers[0], sum(numbers[1:])) if numbers else (0, 0)  # none: an index that never held a row


def _first_of_each(items: Iterable[_T], key: Callable[[_T], Hashable]) -> list[_T]:
    """The first of the items with the same key, in order."""
    first: dict[Hashable, _T] = {}
    for item in items:
        first.setdefault(key(item), item)
    return list(first.values())


VIA = ("path", "name", "alias")  # how a link resolved, when it did
AMBIGUOUS_SHOWN = links.AMBIGUOUS_SHOWN  # notes an ambiguous link or name lists, of those it matches
MENTION_CHARS = 3  # names `mentions` looks for have at least this many characters
UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"})
LINKING_SHOWN = 5  # documents an unresolved target names, of those linking it
LINE_CHARS = 300  # characters of a linking line a link into a document shows, around the link


def _link_out(row: Row, how: Resolution, through: int | None, named: Named) -> dict[str, Any]:
    """A link out of a document, as `links` returns it: as written, and where it leads."""
    _, _, field, line, kind, target, _, fragment, shown = row
    link: dict[str, Any] = {"target": target, "line": line, "kind": kind}
    link |= {key: value for key, value in (("field", field), ("fragment", fragment), ("shown", shown)) if value}
    via, docs = how
    if via in VIA:
        link |= {"id": named[docs[0]][0], "title": named[docs[0]][1], "via": via}
    elif via == "ambiguous":
        link["ambiguous"] = [named[doc][0] for doc in docs[:AMBIGUOUS_SHOWN]]
    else:
        link["unresolved"] = True
    return link | ({"through": named[through][0]} if through else {})


def _by_document(rows: Iterable[Row]) -> dict[int, list[Row]]:
    """Link rows by the document holding them, in order."""
    found: dict[int, list[Row]] = {}
    for row in rows:
        found.setdefault(row[0], []).append(row)
    return found


def _lines_of(rows: Iterable[Row], text: str) -> list[dict[str, Any]]:
    """The lines of a document holding these links, once each: their number, front-matter field, and text, around the
    link if the line is long."""
    every = text.split("\n")
    found: dict[int, dict[str, Any]] = {}
    for row in rows:
        field, target = row[2], row[5]
        line: int = row[3]
        if line not in found:
            content = every[line - 1].strip()
            if len(content) > LINE_CHARS:
                start = max(0, content.find(target) - LINE_CHARS // 3)
                end = start + LINE_CHARS
                content = ("…" if start else "") + content[start:end] + ("…" if end < len(content) else "")
            found[line] = {"line": line} | ({"field": field} if field else {}) | {"text": content}
    return list(found.values())


def _by_value(rows: list[Row], descending: bool) -> list[Row]:
    """Rows of several stores, (collection, path, ..., value), each store's in SQLite's order by value and path,
    merged in that order: numbers before text ascending, text before numbers descending, and rows without a value
    last; then by collection and path."""
    rows = sorted(rows, key=itemgetter(0, 1))
    valued = [row for row in rows if row[-1] is not None]
    valued.sort(key=lambda row: (0, row[-1]) if isinstance(row[-1], int | float) else (1, row[-1]), reverse=descending)
    return valued + [row for row in rows if row[-1] is None]


class _Held(NamedTuple):
    """The codes of a collection's vectors in memory, and what its store said when they were read: its version, the
    epoch of its codes (see store.codes_changed), how many vectors it counted (None: unknown, as after a patch), and
    the last chunk read."""

    version: str | None
    codes: Codes
    epoch: str | None = None
    stored: int | None = None
    last: int = -1


def _appended_to(held: _Held | None, epoch: str | None, count: int) -> int | None:
    """How many codes `held` counted as its store did, if the store has only added codes since, `count` now, in the
    same `epoch`; None if they are to be read anew."""
    if held is None or held.stored is None or held.epoch != epoch or count < held.stored:
        return None
    return held.stored


def _held(db: Db) -> _Held | None:
    """The codes of a store's vectors (see store.read_codes) to search, read in one snapshot with what the store says
    of them; None if it has none."""
    with store.snapshot([db]):
        version, epoch, count = store.get_meta(db, "version"), store.get_meta(db, "codes_epoch"), store.counts(db)
        found = store.read_codes(db)
    return _Held(version, Codes(*found), epoch, count["vectors"], int(found[0][-1])) if found else None


def _read_anew(path: str) -> _Held | None:
    """_held of the store at `path`, through a connection of its own, which a call cancelled meanwhile can't stop."""
    db = store.read_only(path)
    try:
        return _held(db)
    finally:
        db.close()


def _in_a_thread(work: Callable[[], _T]) -> "Future[_T]":
    """A Future of work(), done in a thread of its own, which never holds up the process from ending."""
    done: Future[_T] = Future()

    def run() -> None:
        try:
            done.set_result(work())
        except BaseException as error:  # handed to whoever takes the result
            done.set_exception(error)

    threading.Thread(target=run, name="sift-codes", daemon=True).start()
    return done


def _size(path: str) -> int:
    """The bytes a store takes on disk, its write-ahead log included."""
    return sum(os.path.getsize(file) for file in (path, f"{path}-wal") if os.path.isfile(file))


def _postings_state(db: Db) -> str:
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

    def __init__(self, catalog: Catalog, scope: Scope, where: str, args: Sequence[Any]) -> None:
        self.stores: dict[int, Db] = {c.id: catalog.store(c) for c in scope}
        self.where, self.args = where, args
        self.seen: Ids = np.empty(0, np.int64)  # the chunks checked, by id, and which pass
        self.passed: Mask = np.empty(0, bool)
        self._listed: Ids | None = None

    def check(self, ids: Ids) -> Mask:
        ids = np.asarray(ids, np.int64)
        if len(new := np.setdiff1d(ids, self.seen)):
            seen = np.concatenate([self.seen, new])
            order = np.argsort(seen, kind="stable")
            self.seen, self.passed = seen[order], np.concatenate([self.passed, self.passes(new)])[order]
        return self.passed[np.searchsorted(self.seen, ids)]

    def passes(self, ids: Ids) -> Mask:
        """Which of these chunks, of the collections of the scope, pass, as their stores tell."""
        sql = f"{self.CHUNKS}c.id IN (SELECT value FROM json_each(?)) AND {self.where}"
        owners = ids >> store.SHIFT
        passed: list[int] = []
        for cid in np.unique(owners).tolist():
            rows = self.stores[cid].execute(sql, [json.dumps(ids[owners == cid].tolist()), *self.args])
            passed += [chunk for (chunk,) in rows]
        return np.isin(ids, np.array(passed, np.int64))

    def listed(self) -> Ids:
        if self._listed is None:
            rows = (i for db in self.stores.values() for (i,) in db.execute(self.CHUNKS + self.where, self.args))
            self._listed = np.fromiter(rows, np.int64)
        return self._listed


class _Resolvers(dict[Collection, Resolver]):
    """A links.Resolver for each collection, made when first needed."""

    def __init__(self, catalog: Catalog) -> None:
        super().__init__()
        self.catalog = catalog

    def __missing__(self, collection: Collection) -> Resolver:
        self[collection] = resolver = Resolver(self.catalog.store(collection))
        return resolver


class Index:
    """Search over the active collections of a catalog (default: SIFT_HOME's). Semantic search uses `models` (default:
    `models.default()`, chosen when first needed, so keyword search loads no models)."""

    def __init__(self, catalog: Catalog | None = None, models: _models.Models | None = None) -> None:
        self.catalog = catalog or Catalog()
        self._models = models
        self._wrap: Callable[[_models.Models], _models.Models] | None = None
        self.vectors: dict[int, _Held] = {}  # collection id: the codes of its vectors
        self._reading: dict[int, Future[_Held | None]] = {}  # collection id: its codes read anew, in a thread
        # collection id: (the state of its keyword postings, their postings.Reader)
        self._postings: dict[int, tuple[tuple[str | None, str | None, str | None], postings.Reader]] = {}
        self._said: set[str] = set()  # warnings given, each once
        self._query_vectors: OrderedDict[str, NDArray[np.float32]] = OrderedDict()  # question: its embedding
        # a search and its options: (the state it read, its results), see _remembered
        self._results: OrderedDict[str, tuple[State, list[Hit]]] = OrderedDict()
        self._asked: list[str] = []  # the questions of the call of several being answered, embedded together
        self._passed: set[int] = set()  # active collections a search passed over, stores missing or unusable: not kept

    @staticmethod
    def _several(
        method: Callable[Concatenate["Index", str, ...], list[Hit]],
    ) -> Callable[Concatenate["Index", str | list[Any], ...], list[Hit] | list[list[Hit]]]:
        """A search that takes a list of queries too, and answers each as if asked alone, in order: one call, and one
        turn of an agent, instead of several. Their questions are embedded together, in one pass of the model, once the
        first is checked (see Index._vector); they return at most MOST_HITS hits in all, and share a time `budget`."""

        @wraps(method)
        def several(self: "Index", query: str | list[Any], *args: Any, **kwargs: Any) -> list[Hit] | list[list[Hit]]:
            if not isinstance(query, list):
                return method(self, query, *args, **kwargs)
            if not query:
                raise ValueError("a search takes at least one query")
            if len(query) > MOST_QUERIES:
                raise ValueError(f"a search takes at most {MOST_QUERIES} queries at once, got {len(query)}")
            if wrong := [q for q in query if not isinstance(q, str) or not q.strip()]:
                raise ValueError(f"each query of several is a text that isn't blank, got {wrong[0]!r}")
            k = args[0] if args else kwargs.get("k", 10)
            if isinstance(k, int) and min(k, MOST_HITS) * len(query) > MOST_HITS:  # one asked for more gets MOST_HITS
                raise ValueError(
                    f"several queries return at most {MOST_HITS:,} hits in all; {len(query)} of k={k} would be "
                    f"{k * len(query):,}: ask fewer at a time, or a smaller k"
                )
            budget = kwargs.get("budget")
            if isinstance(budget, int | float) and not isinstance(budget, bool) and budget > 0:  # else said as it was
                kwargs = kwargs | {"budget": budget / len(query)}
            self._asked = list(dict.fromkeys(query))
            try:
                return [method(self, q, *args, **kwargs) for q in query]
            finally:
                self._asked = []

        return several

    @staticmethod
    def _remembered(
        method: Callable[Concatenate["Index", ...], list[Hit]],
    ) -> Callable[Concatenate["Index", ...], list[Hit]]:
        """A search whose results the index keeps, the last RESULTS of them, as long as what it reads stays as it was
        (Index._state): an agent asking again, or the UI showing a page again, gets them without reading the stores."""

        @wraps(method)
        def remembered(self: "Index", *args: Any, **kwargs: Any) -> list[Hit]:
            if kwargs.get("budget") is not None:  # what a search finds in time depends on the time it had
                return method(self, *args, **kwargs)
            try:
                key = json.dumps([method.__name__, args, kwargs], sort_keys=True, default=str)
            except TypeError:  # filters of keys JSON can't sort: no key, and the search says what is wrong
                return method(self, *args, **kwargs)
            self._settle()  # the codes read anew meanwhile are those the state holds
            state = self._state()
            if (kept := self._results.get(key)) is not None and kept[0] == state:
                self._results.move_to_end(key)
                return copy.deepcopy(kept[1])
            self._passed.clear()
            hits = method(self, *args, **kwargs)
            # Not with codes being read anew, which take over later; nor a collection passed over, whose store may come
            # back unsaid; nor results so large that keeping them costs more.
            if not (self._reading or self._passed) and len(hits) <= KEPT_HITS and self._state() == state:
                self._results[key] = (state, copy.deepcopy(hits))
                while len(self._results) > RESULTS:
                    self._results.popitem(last=False)
            return hits

        return remembered

    @property
    def models(self) -> _models.Models:
        if self._models is None:
            self._models = _models.default()
            if self._wrap:
                self._models = self._wrap(self._models)
        return self._models

    def models_through(self, wrap: Callable[[_models.Models], _models.Models]) -> None:
        """Have the models used through wrap(models), e.g. jobs.OnMainThread: now if chosen, else once they are."""
        self._wrap = wrap
        if self._models is not None:
            self._models = wrap(self._models)

    def _state(self) -> State:
        """What searches read, as a value that changes with it: the catalog and the stores opened so far as SQLite
        counts their changes, those other connections commit (data_version) and those of the connection itself
        (total_changes, as the server's own `reindex` makes them), and the codes held. A store not opened yet was read
        by no search kept, and a search that opens it changes this, so it isn't kept that time."""
        stores: list[tuple[Any, ...]] = []
        for cid, path, db in self.catalog.open_stores():
            try:
                stores.append((cid, path, *store.changes(db)))
            except sqlite3.Error:  # out of reach, as on a drive unplugged: equal to no state, so nothing kept or used
                stores.append((cid, path, object()))
        held = tuple((cid, h.version, h.epoch, h.stored, h.last) for cid, h in self.vectors.items())
        return store.changes(self.catalog.db), tuple(stores), held

    def forget(self) -> None:
        """Let go of the questions' embeddings and the searches' results kept: as `sift bench` times each mode."""
        self._query_vectors.clear()
        self._results.clear()

    def _warn(self, text: str) -> None:
        if text not in self._said:
            self._said.add(text)
            console.say(f"warning: {text}")

    # --- which collections and documents a search may find -----------------------

    def active(self) -> dict[int, Collection]:
        """The active collections whose stores open, by id; what is held of others is let go (see _settle)."""
        self._settle()
        found: dict[int, Collection] = {}
        for c in self.catalog.collections():
            if not c.active:
                continue
            if not os.path.isfile(c.store):
                self._passed.add(c.id)
                self._warn(f"{c.name}: its index is missing: {c.store}; searches pass it over")
            else:
                try:
                    self.catalog.opened(c)
                except LookupError as unusable:  # damaged, a newer sift's, locked or out of reach
                    self._passed.add(c.id)
                    self._warn(f"{unusable}; searches pass it over")
                else:
                    found[c.id] = c
        for held in (self.vectors, self._reading, self._postings):
            for cid in held.keys() - found.keys():
                del held[cid]
        return found

    def _scope(self, collection: str | None) -> Scope:
        """The collections a search covers: the one named, or every active one. None active, or none at all, is an
        error that says what to do: "no matches" told nobody that nothing was searched. Those whose stores are
        missing are active, and passed over as said (see active)."""
        if not collection:
            found = list(self.active().values())
            if not found and (nothing := self._nothing_active()):
                raise LookupError(nothing)
            return found
        c = self.catalog.get(collection)
        self._readable(c)
        return [c]

    def _nothing_active(self) -> str | None:
        """What to say when no collection is active, and to do about it; None if one is."""
        collections = self.catalog.collections()
        if any(c.active for c in collections):
            return None
        if collections:
            return "no collection is active: `sift activate NAME` takes one back into searches"
        return "no collection yet: `sift add FOLDER`, then `sift index`"

    def _readable(self, collection: Collection) -> Db:
        """The store of a collection to read from; an error if the collection is inactive, or its store missing."""
        if not collection.active:
            raise LookupError(
                f"collection {collection.name!r} is inactive: "
                f"`{console.command('sift', 'activate', collection.name)}` to search it"
            )
        return self.catalog.store(collection)

    def _owner(self, row_id: int) -> Collection:
        """The collection a document or chunk belongs to, by its row id."""
        cid = store.owner(row_id)
        if found := next((c for c in self.catalog.collections() if c.id == cid), None):
            return found
        raise LookupError("its collection was removed meanwhile")

    def _db(self, row_id: int) -> Db:
        """The store holding a document or chunk, by its row id."""
        return self.catalog.store(self._owner(row_id))

    def _by_store(self, ids: Iterable[int]) -> list[tuple[Collection, Db, list[int]]]:
        """Row ids of documents or chunks by the collection they belong to, of those still there: (collection, its
        store, [ids]) each."""
        known = {c.id: c for c in self.catalog.collections()}
        groups: dict[int, list[int]] = {}
        for i in ids:
            groups.setdefault(store.owner(i), []).append(int(i))
        return [(known[cid], self.catalog.store(known[cid]), group) for cid, group in groups.items() if cid in known]

    def _where(
        self,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
        links_to: Any = None,
        linked_from: Any = None,
    ) -> Where:
        """The collections to search, and an SQL condition on their docs `d` with its arguments. Filters match
        front-matter fields: a value, which a list field matches if it contains it, a list of values for any of them,
        or conditions such as {">=": "2024-06", "<": "2025"}, which one value or list element must satisfy, and
        {"not": value or list}, which none may match. A value that is a wikilink, "[[Acme AG]]", matches a link to the
        same note however it is written. `links_to` and `linked_from` name notes (see _notes): only documents linking
        one of the first, and linked from one of the second."""
        scope, resolvers = self._scope(collection), _Resolvers(self.catalog)
        sql = ["1"]
        args: list[Any] = []
        if path:
            sql.append("d.path GLOB ?")
            args.append(path)
        for key, value in (filters or {}).items():
            field = json_path(key)
            if value == {}:
                raise ValueError(f"filter {key!r} has no condition")
            given = conditions(value)
            some = [comparison(key, op, v) for op, v in given.items() if op != "not"]
            if some:
                condition = f"EXISTS (SELECT 1 FROM json_each(d.meta, ?) WHERE {' AND '.join(c for c, _ in some)})"
                args += [field, *(arg for _, condition_args in some for arg in condition_args)]
                if list(given) == ["="] and (linked := self._linking(key, given["="], scope, resolvers)):
                    condition = f"({condition} OR d.id IN (SELECT value FROM json_each(?)))"
                    args.append(_ids(linked))
                sql.append(condition)
            if "not" in given:
                condition, condition_args = comparison(key, "not", given["not"])
                sql.append(f"NOT EXISTS (SELECT 1 FROM json_each(d.meta, ?) WHERE {condition})")
                args += [field, *condition_args]
                if linked := self._linking(key, given["not"], scope, resolvers):
                    sql.append("d.id NOT IN (SELECT value FROM json_each(?))")
                    args.append(_ids(linked))
        allowed = self._linked(links_to, linked_from, scope, resolvers)
        if allowed is not None:
            sql.append("d.id IN (SELECT value FROM json_each(?))")
            args.append(_ids(allowed))
        return scope, " AND ".join(sql), args

    def _resolve(self, spec: str, scope: Scope, resolvers: _Resolvers) -> dict[Collection, Resolution]:
        """Where a note named as links name it leads, from no note in particular, in each collection of `scope`:
        {collection: (how, [doc ids])}."""
        target = _target(spec)
        key = readers.key(target.rsplit("/", 1)[-1])
        return {c: resolvers[c].resolve(None, "wikilink", target, key) for c in scope}

    def _notes(
        self, specs: str | Sequence[Any], what: str, scope: Scope, resolvers: _Resolvers
    ) -> dict[str, int | None]:
        """The notes that `specs`, one or a list, name: by a document's id, as for `get`, or as links name them,
        "Acme AG" or "[[Acme AG]]", in the collections of `scope`, wherever one note answers. As {spec: doc row id, or
        None for a note no document answers to}; a name that more than one note answers to is an error."""
        specs = [specs] if isinstance(specs, str) else specs
        if not isinstance(specs, list | tuple) or not specs:
            raise ValueError(f"{what} needs at least one note: their ids, or link targets, got {specs!r}")
        found: dict[str, int | None] = {}
        for spec in specs:
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError(f"{what} takes ids of documents or link targets, got {spec!r}")
            rows: list[Row] = [] if readers.WIKILINK_VALUE.fullmatch(spec.strip()) else self._rows(spec.strip())
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

    def _linked(self, links_to: Any, linked_from: Any, scope: Scope, resolvers: _Resolvers) -> set[int] | None:
        """The documents that link to any of the notes `links_to` names, and that one of `linked_from` links to, by
        row id; None if neither is given. A note no document answers to yet is linked by the links that name it."""
        allowed: set[int] | None = None
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
            to: set[int] = set()
            for spec, note in self._notes(linked_from, "linked_from", scope, resolvers).items():
                if note is None:
                    raise LookupError(f"no note {spec!r} in the index")
                c = self._owner(note)
                outgoing = links.outgoing(self.catalog.store(c), resolvers[c], note)
                to |= {docs[0] for _, (how, docs) in outgoing if how in VIA}
            allowed = to if allowed is None else allowed & to
        return allowed

    def _linking(self, key: str, value: Any, scope: Scope, resolvers: _Resolvers) -> set[int] | None:
        """The documents whose front-matter field `key` links to a note that a wikilink among the filter values names,
        however the link is written; links to a note no document answers to, or to several, match as written. None if
        no value is a wikilink."""
        specs = _wikilinks(value)
        docs: set[int] = set()
        for spec in specs:
            for c, (how, found) in self._resolve(spec, scope, resolvers).items():
                db = self.catalog.store(c)
                if how in VIA:
                    docs |= {row[0] for row in links.incoming(db, resolvers[c], found[0], key)}
                else:
                    docs |= set(links.written(db, _target(spec), key))
        return docs if specs else None

    # --- candidates as [(chunk_id, score)] -----------------------------------

    def _keyword(
        self,
        match: str,
        k: int,
        scope: Scope,
        where: str,
        args: Sequence[Any],
        all_passages: bool = False,
        deadline: topk.Deadline | None = None,
    ) -> Ranked:
        """The best passage of each of the k documents that best match an FTS5 query, by BM25, or the k best
        passages, in the collections of `scope`, scored with the word counts of all of them. A query of words and
        phrases, AND, OR and NOT is answered from the keyword postings (see `topk`), until the topk.Deadline if given,
        where every collection has them ready; others by FTS5, each collection's best scored again together (see
        _as_one), unless the query places its phrases (NEAR, column filters)."""
        if not match.strip():
            return []
        query = expression.parse(match)
        if query is None and expression.rejected(match):
            query = expression.parse(queries.as_phrases(match))  # as FTS5 below reads it: its words as phrases
        if query is not None:
            with store.snapshot(self.catalog.store(c) for c in scope):  # postings rebuilt meanwhile are another state's
                if (held := self._ready(scope)) is not None:
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
        errors: list[sqlite3.OperationalError] = []
        for attempt in (match, queries.as_phrases(match)):
            phrases = queries.phrases_of(attempt) if len(scope) > 1 else None
            each = k if not phrases else k + max(k, MORE_CANDIDATES)  # a collection's best may rank otherwise
            query = sql if all_passages or not phrases else every
            try:
                found = {c: self.catalog.store(c).execute(query, [attempt, *args, each]).fetchall() for c in scope}
            except sqlite3.OperationalError as e:
                if not expression.rejected(attempt):  # the store's trouble, or a call cancelled: not the query's
                    raise
                errors.append(e)
                continue
            if phrases:
                found = self._as_one(found, phrases)
            ranked = sorted((row for rows in found.values() for row in rows), key=lambda row: -row[2])
            ranked = ranked if all_passages else _first_of_each(ranked, itemgetter(1))
            return [(passage, score) for passage, _, score in ranked[:k]]
        raise ValueError(f"invalid search query {match!r}: {errors[-1]}")

    def _ready(self, scope: Scope) -> list[postings.Reader] | None:
        """The keyword postings of the collections of `scope`, one postings.Reader each, or None if any has none
        ready: never built, or behind FTS5 since a change. A collection without a chunk has nothing to find, and no
        postings to wait for: added and not indexed yet, it sent the searches of every collection to FTS5."""
        readers: list[postings.Reader] = []
        for c in scope:
            db = self.catalog.store(c)
            if not store.counts(db)["chunks"]:
                continue
            if not postings.ready(db):
                return None
            state = postings.Reader.state(db)
            held = self._postings.get(c.id)
            if held is None or held[0] != state or held[1].db is not db:  # changed, or the store opened anew
                held = self._postings[c.id] = (state, postings.Reader(db))
            readers.append(held[1])
        return readers

    def _as_one(self, found: dict[Collection, Passages], phrases: list[str]) -> dict[Collection, Passages]:
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
        scored: dict[Collection, Passages] = {}
        for c, passages in found.items():
            db, ids = self.catalog.store(c), [passage for passage, _, _ in passages]
            size = postings.sizes(db, ids)
            score = dict.fromkeys(size, 0.0)
            for weight, phrase in zip(idf, phrases):
                for passage, n in postings.matches(db, phrase, ids).items():
                    score[passage] += weight * float(postings.term_scores(n, size[passage], average))
            scored[c] = [(passage, doc, score.get(passage, 0.0)) for passage, doc, _ in passages]
        return scored

    def load(self, scope: Scope | None = None, required: bool = True) -> None:
        """Load the codes of the vectors (see `vectors`) of the collections of `scope`, those a search covers, or of
        every active one, as a server does when it starts: each store's batch by batch. Follow a store that changed
        since: the codes written past the last held are read alone, as `sift embed` writes them; after another change,
        all are read again in a thread of their own, while searches go on with those held, which the codes read then
        replace. An error if none of them has codes, unless not `required`: `query` reranks its keyword candidates
        then."""
        if scope is None:
            scope = list(self.active().values())
        else:
            self._settle()
        if not scope and self.catalog.collections() and (nothing := self._nothing_active()):
            raise LookupError(nothing)
        damaged: dict[str, str] = {}
        for c in scope:
            try:
                self._follow(c)
            except sqlite3.DatabaseError as error:  # one store damaged keeps no server from starting
                damaged[c.name] = store.damage(c.name, error)
                self._warn(f"{damaged[c.name]}; semantic searches pass it over")
                self._passed.add(c.id)
        if required and not any(c.id in self.vectors for c in scope):
            bare = [c.name for c in scope if c.name not in damaged]  # what to do about each: embed, or forget
            where = f" in {', '.join(bare)}" if bare else ""
            said = [f"no embeddings yet{where}, run `sift embed`"] if bare or not damaged else []
            raise LookupError("; ".join(said + list(damaged.values())))

    def _settle(self) -> None:
        """Let go of what is held of collections no longer active, as another sift deactivated or removed them, and
        close their stores (the space of a store `sift rm` deleted is freed once closed); and take the codes read anew
        that are ready. Without opening every active store, as a search of one collection doesn't."""
        collections = {c.id: c for c in self.catalog.collections() if c.active}
        for held in (self.vectors, self._reading, self._postings):
            for cid in held.keys() - collections.keys():
                del held[cid]
        for cid, _, db in self.catalog.open_stores():
            if cid not in collections or not _reachable(db):  # one out of reach is opened again once it is back
                self.catalog.release(cid)
                # A store opened anew counts its changes afresh, as the one before began: no state tells them apart
                self._results.clear()
        for cid in [cid for cid, reading in self._reading.items() if reading.done()]:
            self._take_read(collections[cid])

    def _take_read(self, collection: Collection) -> None:
        """Take the codes of a collection read anew, once they are."""
        cid = collection.id
        if (reading := self._reading.get(cid)) is not None and reading.done():
            del self._reading[cid]
            try:
                found = reading.result()
            except Exception as error:  # a disk that failed, no room for a second copy: read again for the next search
                self._warn(f"{collection.name}: its vectors' codes could not be read again: {error}")
            else:
                if found is None:
                    self.vectors.pop(cid, None)
                else:
                    self.vectors[cid] = found

    def _follow(self, collection: Collection) -> None:
        """Bring the codes held of a collection up to its store (see load)."""
        cid, db = collection.id, self.catalog.store(collection)
        self._take_read(collection)
        held = self.vectors.get(cid)
        with store.snapshot([db]):
            version = store.get_meta(db, "version")
            if held is not None and version is not None and held.version == version:
                return
            count, epoch = store.counts(db)["vectors"], store.get_meta(db, "codes_epoch")
            stored = _appended_to(held, epoch, count)
            most = count - stored + 1 if stored is not None else count
            tail = store.read_codes(db, after=held.last, most=most) if held is not None and count else None
        if tail and held is not None and tail[1].shape[1] != held.codes.width:  # another model's, after a reset
            tail, stored = None, None  # all read anew
        if not count:  # no vectors at all, as after `sift embed --reset`: no search can find any
            self.vectors.pop(cid, None)
            self._reading.pop(cid, None)  # of the vectors gone
        elif held is None:
            self._reading.pop(cid, None)  # older than what is read now
            if (found := _held(db)) is not None:
                self.vectors[cid] = found
        elif stored is not None and len(tail[0] if tail else ()) == count - stored:
            codes, last = (held.codes.appended(*tail), int(tail[0][-1])) if tail else (held.codes, held.last)
            self.vectors[cid] = _Held(version, codes, epoch, count, last)
        else:  # read anew; meanwhile the codes past the last held join them, so that a note just embedded is found
            if tail:
                self.vectors[cid] = held._replace(codes=held.codes.appended(*tail), stored=None, last=int(tail[0][-1]))
            if cid not in self._reading:
                self._reading[cid] = _in_a_thread(partial(_read_anew, collection.store))

    def settle(self, timeout: float | None = None) -> None:
        """Wait until the codes being read anew are read, at most `timeout` seconds each; the next search takes them."""
        for reading in list(self._reading.values()):
            wait([reading], timeout)

    def _exact(self, chunks: IdList, dim: int) -> tuple[Ids, NDArray[np.float16]]:
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

    @overload
    def _vector(self, query: str, scope: Scope) -> NDArray[np.float32]: ...

    @overload
    def _vector(self, query: str, scope: Scope, required: Literal[True]) -> NDArray[np.float32]: ...

    @overload
    def _vector(self, query: str, scope: Scope, required: bool) -> NDArray[np.float32] | None: ...

    def _vector(self, query: str, scope: Scope, required: bool = True) -> NDArray[np.float32] | None:
        """The query's vector, with the codes of the collections searched loaded (see load): kept for the next
        QUERY_VECTORS questions, and embedded together with the others of a call of several queries, the first time one
        is needed, after the search checked what it was asked."""
        self.load(scope, required)
        if not any(c.id in self.vectors for c in scope):  # for `query`: nothing embedded to compare it with
            return None
        self._embed_questions([query, *self._asked])
        return self._query_vectors[query]

    def _embed_questions(self, questions: list[str]) -> None:
        """Embed those of the questions not kept, together, keeping all of them as the latest."""
        new = [q for q in dict.fromkeys(questions) if q not in self._query_vectors]
        for question, vector in zip(new, self.models.encode(new, query=True) if new else ()):
            vector = vector.astype(np.float32)
            vector.setflags(write=False)  # shared by every search asking it again
            self._query_vectors[question] = vector
        for question in questions:
            self._query_vectors.move_to_end(question)
        while len(self._query_vectors) > QUERY_VECTORS:
            self._query_vectors.popitem(last=False)

    def _filter(self, scope: Scope, where: str, args: Sequence[Any]) -> _Filter | None:
        """A search's filter of the chunks `where` allows, or None for all: one per search, as it remembers what it was
        asked (see _Filter)."""
        return _Filter(self.catalog, scope, where, args) if args else None

    def _nearest(
        self,
        query: Floats,
        k: int,
        scope: Scope,
        allowed: Filter | None,
        all_passages: bool,
        excluded: IdList | None = None,
    ) -> Ranked:
        """The best passage of each of the k documents nearest the query's vector, among the chunks of the collections
        of `scope` the filter `allowed` lets through (see _filter), but those `excluded`; or the k best passages."""
        held: list[Codes] = []
        width = codes(query[None, :]).shape[1]
        for c in scope:
            if (theirs := self.vectors.get(c.id)) is None:
                continue
            if theirs.codes.width != width:
                self._warn(f"{c.name} was embedded with another model: `sift embed --reset` embeds it with this one")
            else:
                held.append(theirs.codes)

        def exact(ids: Ids) -> tuple[Ids, NDArray[np.float16]]:
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

    def _per_document(self, ranked: Ranked) -> Ranked:
        """The first, best, of each document's passages in `ranked`."""
        sql = "SELECT c.id, c.doc FROM json_each(?) j JOIN chunks c ON c.id = j.value"
        docs: dict[int, int] = {}
        for _, db, group in self._by_store(chunk for chunk, _ in ranked):
            docs |= dict(db.execute(sql, (json.dumps(group),)))
        return _first_of_each((r for r in ranked if r[0] in docs), lambda r: docs[r[0]])  # in docs: not removed

    def _works(
        self,
        rank: Callable[[int], Ranked],
        k: int,
        all_passages: bool,
        more: Callable[[], bool] = lambda: True,
        keep: Callable[[Hit], bool] = lambda hit: True,
    ) -> list[Hit]:
        """The hits of the best k works by rank(n), the best n documents, of those keep(hit) keeps: one hit per work, as
        _one_per_work leaves them, or with `all_passages` the best k passages; asking for more while fewer are left,
        and more() allows."""
        n = k
        while True:
            ranked = rank(n)
            hits = [hit for hit in self._hits(ranked) if keep(hit)]
            hits = hits if all_passages else one_per_work(hits)
            if len(hits) >= k or len(ranked) < n or not more():
                return hits[:k]
            n *= 2

    def _hits(self, ranked: Ranked) -> list[Hit]:
        sql = """SELECT d.path, d.title, d.meta, c.line_start, c.line_end, c.id, c.doc, c.pos, c.len
                 FROM chunks c JOIN docs d ON d.id = c.doc WHERE c.id IN (SELECT value FROM json_each(?))"""
        found: dict[int, tuple[Any, ...]] = {}
        hits: list[Hit] = []
        for c, db, chunks in self._by_store(chunk for chunk, _ in ranked):
            with store.snapshot([db]):  # the chunks and their texts as they are at one moment
                rows = db.execute(sql, (json.dumps(chunks),)).fetchall()
                texts = store.texts(db, (row[5:] for row in rows))  # each document's text read once
            for *shown, chunk, _, _, _ in rows:
                found[chunk] = (c.name, *shown, texts[chunk])
        for chunk, score in ranked:
            if chunk in found:  # else removed by a concurrent `sift index`
                name, path, title, meta, start, end, text = found[chunk]
                hit: Hit = {"id": f"{name}/{path}", "title": title, "lines": f"{start}-{end}", "score": round(score, 4)}
                hits.append(hit | {"meta": meta, "text": text, "chunk": chunk})  # for _present, and `query`'s scores
        return hits

    def _cosines(self, query: Floats | None, chunks: Sequence[int]) -> NDArray[np.float64]:
        """The cosines of these chunks with the query's vector, exactly; one not embedded yet, or broken, gets the
        others' mean, neither a bonus nor a penalty; all of them 0 without a vector (see _vector)."""
        if query is None:
            return np.zeros(len(chunks))
        found, rows = self._exact(np.array(chunks, np.int64), len(query))
        exact = dict(zip(found.tolist(), (rows.astype(np.float32) @ query).tolist()))
        cosine = np.array([exact.get(chunk, np.nan) for chunk in chunks], np.float64)
        cosine[~np.isfinite(cosine)] = np.nan
        known = np.isfinite(cosine)
        return np.where(known, cosine, np.nanmean(cosine) if known.any() else 0.0)

    # --- public API ----------------------------------------------------------

    # Every search takes, beside what it looks for and k, the options of hits.Options, by name: where it looks, what
    # it lets through, and how its hits come.

    def _options(self, k: object, options: dict[str, Any]) -> tuple[int, Options, Scope, str, list[Any]]:
        """What a search was asked beside what it looks for, checked: (k, its Options, the collections it covers, and
        the SQL condition on their documents with its arguments, see _where)."""
        asked = Options(**options)
        k, (scope, where, args) = check_k(k), self._where(*asked.where)
        return k, asked.checked(), scope, where, args

    @_several
    @_remembered
    def search(self, query: str, k: int = 10, *, budget: float | None = None, **options: Any) -> list[Hit]:
        """Exact keyword search, ranked by BM25. Accepts FTS5 syntax: "phrases", prefix*, AND, OR, NOT.
        If no chunk holds all terms of a query without operators, chunks with any of them are returned. With a
        `budget` in seconds, a search the keyword postings answer stops once it is spent and has found any, and returns
        the best found so far, each hit marked "partial"."""
        k, asked, scope, where, args = self._options(k, options)
        budget = check_budget(budget)
        deadline = topk.Deadline(budget) if budget is not None else None

        def rank(n: int) -> Ranked:
            ranked = self._keyword(query, n, scope, where, args, asked.all_passages, deadline)
            if not ranked and (relaxed := queries.any_term(query)):
                ranked = self._keyword(relaxed, n, scope, where, args, asked.all_passages, deadline)
            return ranked

        stopped: list[bool] = []

        def in_time() -> bool:  # asked again out of time, a search scores less than the first ask did
            if deadline is not None and (deadline.cut or deadline.over()):
                stopped.append(True)
            return not stopped

        hits = self._works(rank, k, asked.all_passages, more=in_time)
        for hit in hits if stopped or (deadline is not None and deadline.cut) else ():
            hit["partial"] = True
        return asked.shown(hits)

    @_several
    @_remembered
    def vsearch(self, query: str, k: int = 10, **options: Any) -> list[Hit]:
        """Semantic search: the passages nearest the query's embedding, by cosine similarity (see `vectors`)."""
        k, asked, scope, where, args = self._options(k, options)
        if not query.strip():  # the embedding of nothing is close to arbitrary chunks
            return []
        vector, allowed = self._vector(query, scope), self._filter(scope, where, args)
        hits = self._works(
            lambda n: self._nearest(vector, n, scope, allowed, asked.all_passages), k, asked.all_passages
        )
        return asked.shown(hits)

    @_several
    @_remembered
    def query(self, query: str, k: int = 10, *, candidates: int = 40, **options: Any) -> list[Hit]:
        """Hybrid search for RAG: keyword and semantic candidates, all reranked together, and ordered by the reranker's
        score blended with their cosine (see _blend). Fusing the candidates' first ranks into the order, e.g. by
        reciprocal rank, made results worse (MRR@10 0.69 to 0.64 on a known-item test over 3.2 million arXiv
        abstracts); the blend made them better on six of the seven sets in docs/benchmarks.md, and cost a little on
        those arXiv known items (0.937 to 0.929)."""
        k, asked, scope, where, args = self._options(k, options)
        candidates = min(max(candidates, k), MOST_HITS)
        keyword = self._keyword(queries.any_word(query), candidates, scope, where, args, asked.all_passages)
        vector = self._vector(query, scope, required=False) if query.strip() else None  # semantic candidates, cosines
        allowed = self._filter(scope, where, args)
        semantic = self._nearest(vector, candidates, scope, allowed, asked.all_passages) if vector is not None else []
        hits = self._hits([(chunk, 0.0) for chunk in dict.fromkeys(chunk for chunk, _ in keyword + semantic)])
        if not hits:
            return []
        words = set(queries.words(query))
        named = [queries.named(h, words) for h in hits]
        rerank = np.asarray(self.models.rerank(query, [queries.passage(h, n) for h, n in zip(hits, named)]), np.float64)
        scores = queries.blend(
            rerank, self._cosines(vector, [h["chunk"] for h in hits]), np.array([bool(n) for n in named])
        )
        for h, s in zip(hits, scores):
            h["score"] = round(float(s), 4)
        hits = sorted(hits, key=lambda h: -h["score"])
        hits = (hits if asked.all_passages else one_per_work(_first_of_each(hits, itemgetter("id"))))[:k]
        return asked.shown(hits)

    def get(self, id: str, lines: object = None) -> str:
        """Text of an indexed document, by its `collection/path` id: all of it, or `lines` such as "17-26", as
        search hits give them, or one line number. The path may leave out folders if the file name is unique in the
        collection. The text comes from the index as it was indexed, so it always matches the hits."""
        return self.document(id, lines)["text"]

    def document(self, id: str, lines: object = None, fields: Any = None) -> dict[str, Any]:
        """A document as {"id", "title", "text"}, the whole id even if asked by file name, and with "fields" if asked
        for, as in hits; `id` and `lines` as for `get`."""
        span, fields = None if lines is None else line_span(lines), check_fields(fields)
        _, whole, title, meta, text = self._find(id)
        document: dict[str, Any] = {"id": whole, "title": title} | ({"fields": pick(meta, fields)} if fields else {})
        if span:
            every = readers.LINE.findall(text)
            if span[0] > len(every):
                raise ValueError(f"{whole} has {console.count(len(every), 'line')}; line {span[0]} is past its end")
            text = "".join(every[span[0] - 1 : span[1]])
        return document | {"text": text}

    def documents(self, asked: Iterable[tuple[str, object]], fields: Any = None) -> list[dict[str, Any]]:
        """Several documents, each (id, lines) of `asked` as `document` gives it, in order; one that can't be read is
        {"id", "error"}, and the others still come."""
        check_fields(fields)
        found: list[dict[str, Any]] = []
        for id, lines in asked:
            try:
                found.append(self.document(id, lines, fields))
            except (KeyError, IndexError):  # LookupErrors too, but a bug's, not a document that can't be read
                raise
            except (LookupError, ValueError) as error:
                found.append({"id": id, "error": str(error)})
            except sqlite3.Error as error:  # its drive gone; but a cancelled call's statements end so too: it stops
                if store.ABORT.is_set():
                    raise
                found.append({"id": id, "error": f"its store can't be read ({error}): is its drive there?"})
        return found

    @_remembered
    def similar(self, id: str, k: int = 10, **options: Any) -> list[Hit]:
        """Documents closest in meaning to the document `id` (as for `get`), by cosine similarity with the mean of its
        passages' vectors; the document itself is left out, and so are its copies, documents with its DOI."""
        k, asked, scope, where, args = self._options(k, options)
        doc, _, _, meta, _ = self._find(id)
        self.load(scope)
        sql = "SELECT v.chunk, v.vec FROM vectors v JOIN chunks c ON c.id = v.chunk WHERE c.doc = ?"
        own = self._db(doc).execute(sql, (doc,)).fetchall()
        if not own:
            raise LookupError(f"{id!r} has no embeddings yet, run `sift embed`")
        mean = np.mean([np.frombuffer(vec, np.float16) for _, vec in own], axis=0, dtype=np.float32)
        excluded, vector = np.array([chunk for chunk, _ in own], np.int64), mean / np.linalg.norm(mean)
        doi, allowed = doi_of(meta), self._filter(scope, where, args)

        def other(hit: Hit) -> bool:  # its copies in other collections are the same paper
            return doi is None or doi_of(hit["meta"]) != doi

        hits = self._works(
            lambda n: self._nearest(vector, n, scope, allowed, asked.all_passages, excluded),
            k,
            asked.all_passages,
            keep=other,
        )
        return asked.shown(hits)

    def _place(self, id: str) -> tuple[Collection | None, str]:
        """The collection, and the path in it, that `id` names: "collection/path", or an absolute path in a
        collection's folder, the nearest that holds it (see indexing.holding); (None, path) for a collection there
        isn't, (None, "") for an absolute path in none."""
        collections = self.catalog.collections()
        if not os.path.isabs(id):
            name, _, rel = id.partition("/")
            return next((c for c in collections if c.name == name), None), rel
        return indexing.holding(collections, os.path.normpath(id)) or (None, "")

    def _rows(self, id: str) -> list[Row]:
        """The documents `id` may name (see `get`), at most six: (row id, path, title, meta, text) of each. For an id
        that is no document's path, only the documents whose paths end in its last part are read, as the store lists
        them (store.BASENAMES); every path of an older store, until `sift index` has listed them."""
        collection, rel = self._place(id)
        if collection is None:
            return []
        db, sql = self._readable(collection), "SELECT id, path, title, meta, text FROM docs WHERE "
        rows = db.execute(sql + "path = ?", (rel,)).fetchall()
        if rows or not rel:
            return rows
        listed = store.basenames_listed(db)  # CROSS JOIN: the documents of that last part first, then each one's path
        among = "basenames b CROSS JOIN docs d ON d.id = b.doc WHERE b.name = ? AND " if listed else "docs d WHERE "
        name = [rel.rsplit("/", 1)[-1]] if listed else []

        def where(condition: str, *args: Any) -> list[Row]:
            """The documents among those whose path `condition` holds for, by path. Their ids first, found by their
            paths alone: no text is read but of the documents returned."""
            ids = f"SELECT d.id FROM {among}{condition}"
            return db.execute(f"{sql}id IN ({ids}) ORDER BY path", (*name, *args)).fetchall()

        if os.path.isabs(id):  # in another case, as Windows and macOS name the same file, and `reindex` takes it
            same = [row for row in where("lower(d.path) = lower(?)", rel) if readers.fold(row[1]) == readers.fold(rel)]
            return same if len(same) == 1 else []
        # e.g. arxiv/2603.29315.md for arxiv/2026/03/2603.29315.md
        return where("substr(d.path, -?) = ? ORDER BY d.path LIMIT 6", len(rel) + 1, "/" + rel)

    def _note(self, id: str) -> Doc:
        """The document `id` names, as for `get`, or else the note `id` names as links name it, "Acme AG" or
        "[[Acme AG]]", if one note answers to it in all active collections: as _find gives it."""
        try:
            return self._find(id)
        except LookupError as unknown:
            note = self._notes([id], "id", self._scope(None), _Resolvers(self.catalog))[id]
            if note is None:
                raise unknown from None
            return self._find(self._named([note])[note][0])

    def _find(self, id: str) -> Doc:
        """The document `id` names (see `get`), as (row id, whole id, title, meta, text)."""
        collection, rows = self._place(id)[0], self._rows(id)
        if collection is None or not rows:  # no collection: no rows
            raise LookupError(f"unknown document {id!r}")
        if len(rows) > 1:
            matches = ", ".join(f"{collection.name}/{row[1]}" for row in rows[:5]) + (", ..." if len(rows) > 5 else "")
            raise LookupError(f"{id!r} matches {matches}; give the whole id")
        doc, path, title, meta, text = rows[0]
        return doc, f"{collection.name}/{path}", title, meta, text

    def browse(
        self,
        k: int = 50,
        *,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
        fields: Any = None,
        compact: bool = False,
        sort: Any = None,
        links_to: Any = None,
        linked_from: Any = None,
    ) -> list[dict[str, Any]]:
        """Documents without a query, by collection and path: what a collection, folder or filter holds. They come as
        hits without lines and score, with the start of their first passage, after any front matter, as their
        passage; `sort` orders them by a front-matter field instead, as in searches."""
        k, (scope, where, args) = check_k(k), self._where(collection, path, filters, links_to, linked_from)
        fields, sort = check_fields(fields), check_sort(sort)
        value, order = "NULL", "d.path"
        by: list[str] = []
        if sort:
            value = "json_extract(d.meta, ?)"
            order = f"{value} IS NULL, {value}{' DESC' if sort[0] == '-' else ''}, {order}"
            by = [json_path(sort.removeprefix("-"))]
        # The first chunk's start, after any front matter; as a subquery, not a join, so that sorting a whole collection
        # doesn't join every document first (0.4 s, against 5 s, for 3 million).
        first = f"SELECT substr(d.text, c.pos + 1, min(c.len, {START})) FROM chunks c WHERE c.doc = d.id AND c.seq = 0"
        sql = f"""SELECT d.path, d.title, d.meta, coalesce(({first}), ''), {value} FROM docs d WHERE {where}
                  ORDER BY {order} LIMIT ?"""
        rows = [(c.name, *row) for c in scope for row in self.catalog.store(c).execute(sql, [*by, *args, *by * 2, k])]
        rows = _by_value(rows, sort[0] == "-") if sort else rows  # else in order: the collections by name, each by path
        docs: list[dict[str, Any]] = []
        for name, doc_path, title, meta, text, _ in rows[:k]:
            doc: dict[str, Any] = {"id": f"{name}/{doc_path}", "title": title}
            if fields:
                doc["fields"] = pick(meta, fields)
            docs.append(doc if compact else doc | {"text": text})
        return docs

    def folders(self, collection: str, path: str = "") -> list[str]:
        """The folders directly inside the folder `path` ("" or e.g. "2024/") of a collection, by name. Each folder
        is one jump through the index, past the documents in it, so a folder holding millions of documents in its
        folders costs as little as an empty one; documents lying in `path` itself are passed over within a query."""
        path = path.strip("/") + "/" if path.strip("/") else ""  # "2024", "/2024/": the folder 2024/
        db, after = self._readable(self.catalog.get(collection)), path
        found: list[str] = []
        within, bound = (" AND path < ?", [path[:-1] + "0"]) if path else ("", [])  # "0" follows "/"
        sql = f"SELECT path FROM docs WHERE path > ?{within} AND instr(substr(path, ?), '/') ORDER BY path LIMIT 1"
        while row := db.execute(sql, (after, *bound, len(path) + 1)).fetchone():
            name = row[0][len(path) :].partition("/")[0]
            if name:  # a record whose id starts with "/" lies in a folder of no name, which nothing could open
                found.append(name)
            after = path + name + "0"  # past the folder: whatever starts with `name/` sorts before
        return sorted(found)

    def file(self, id: str) -> Path | None:
        """The file a document was read from, by its id as for `get`; None for a JSON Lines record, which is a part
        of its file, and for a file outside the collection's folder, as an older sift's reindex stored "D:/x.md"."""
        doc = self._find(id)[0]
        collection = self._owner(doc)
        sql = "SELECT s.path, d.path FROM docs d JOIN sources s ON s.id = d.source WHERE d.id = ?"
        source, path = self.catalog.store(collection).execute(sql, (doc,)).fetchone()
        full = os.path.normpath(os.path.join(collection.path, *source.split("/")))
        return Path(full) if source == path and indexing.inside(collection.path, full) else None

    # --- links between notes (see `links`) -----------------------------------

    def links(self, id: str, direction: str = "out", depth: Any = 1, field: Any = None, k: int = 50) -> dict[str, Any]:
        """The links of a document `id` (as for `get`), as {"id", "title", "out", "in"}: "out" the links it holds, in
        order, each with where it leads; "in" the documents linking it, by path, each with the lines that do. Depth 2
        adds those one link further, each with the document it was reached `through`; `field` keeps to links in that
        front-matter field; k caps each way."""
        if direction not in ("out", "in", "both"):
            raise ValueError(f'direction must be "out", "in" or "both", got {direction!r}')
        if isinstance(depth, bool) or depth not in (1, 2):
            raise ValueError(f"depth must be 1 or 2, got {depth!r}")
        if field is not None and not isinstance(field, str):
            raise ValueError(f"field must be the name of a front-matter field, got {field!r}")
        k = check_k(k)
        doc, whole, title, _, _ = self._note(id)
        db = self._db(doc)
        resolver = Resolver(db)
        found: dict[str, Any] = {"id": whole, "title": title}
        with store.snapshot([db]):  # the links, and the notes they name, as they are at one moment
            if direction != "in":
                found["out"] = self._out(db, resolver, doc, depth, field, k)
            if direction != "out":
                found["in"] = self._in(db, resolver, doc, depth, field, k)
        return found

    def _out(self, db: Db, resolver: Resolver, doc: int, depth: int, field: str | None, k: int) -> list[dict[str, Any]]:
        """The links out of a document, and with depth 2, the documents they lead to link: (row, how, through)."""
        first = links.outgoing(db, resolver, doc, field)
        found: list[tuple[Row, Resolution, int | None]] = [(row, how, None) for row, how in first]
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

    def _in(self, db: Db, resolver: Resolver, doc: int, depth: int, field: str | None, k: int) -> list[dict[str, Any]]:
        """The documents linking a document, and with depth 2, those linking them: (doc, its links, through)."""
        first = _by_document(links.incoming(db, resolver, doc, field))
        found: list[tuple[int, list[Row], int | None]] = [(source, rows, None) for source, rows in first.items()]
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

    def link_count(self, id: str) -> tuple[int, int]:
        """How many documents link to the document `id`, and how many documents it links to: (in, out)."""
        doc = self._note(id)[0]
        db = self._db(doc)
        resolver = links.Resolver(db)
        into = {row[0] for row in links.incoming(db, resolver, doc)}
        out = {docs[0] for _, (via, docs) in links.outgoing(db, resolver, doc) if via in VIA}
        return len(into), len(out)

    def _named(self, docs: Iterable[int]) -> Named:
        """The whole id and title of documents, by row id."""
        sql = "SELECT id, path, title FROM docs WHERE id IN (SELECT value FROM json_each(?))"
        named: Named = {}
        for c, db, group in self._by_store(set(docs)):
            named |= {doc: (f"{c.name}/{path}", title) for doc, path, title in db.execute(sql, (json.dumps(group),))}
        return named

    def unresolved(self, collection: str | None = None, k: int = 50) -> list[dict[str, Any]]:
        """Link targets no document answers to, in a collection or in every active one: notes not written yet. The
        most linked first, each with how many documents link it and the ids of a few of them."""
        k, scope = check_k(k), self._scope(collection)
        found = [target for c in scope for target in links.unresolved(self.catalog.store(c))]
        found = sorted(found, key=lambda t: (-len(t[1]), t[0]))[:k]
        named = self._named([doc for _, docs, _ in found for doc in docs])
        return [
            {"target": target, "docs": len(docs), "ids": sorted(named[doc][0] for doc in docs)[:LINKING_SHOWN]}
            for target, docs, _ in found
        ]

    def mentions(
        self,
        id: str,
        k: int = 10,
        *,
        collection: str | None = None,
        also: str | Sequence[Any] | None = None,
        fields: Any = None,
        compact: bool = False,
    ) -> list[Hit]:
        """Documents that name the note `id` without linking to it, found as search finds them, by its names as phrases:
        its file name, its `title`, its aliases, and the values of the front-matter fields `also` names, e.g.
        ["emails"]. The note itself and the documents linking it are left out, and names shorter than MENTION_CHARS,
        which would be everywhere. A name with ä, ö, ü or ß is also looked for as written with ae, oe, ue or ss: the
        keyword index finds Müller as Muller, but not as Mueller."""
        k, fields = check_k(k), check_fields(fields)
        also = [also] if isinstance(also, str) else also
        if also is not None and not (isinstance(also, list | tuple) and all(isinstance(f, str) and f for f in also)):
            raise ValueError(f'also must name front-matter fields, e.g. ["emails"], got {also!r}')
        doc, whole, _, meta, _ = self._note(id)
        meta = json.loads(meta)
        names = [posixpath.splitext(whole.rsplit("/", 1)[-1])[0], *links.aliases(meta)]
        for value in [meta.get("title"), *(meta.get(field) for field in also or [])]:
            names += [v for v in each(value) if isinstance(v, str)]
        phrases: dict[str, str] = {}
        for name in names:
            for variant in (name, name.translate(UMLAUTS)):
                variant = " ".join(variant.split())
                if len(variant) >= MENTION_CHARS and re.search(r"\w", variant) and not _wikilinks(variant):
                    phrases.setdefault(readers.fold(variant), variant)
        if not phrases:
            return []
        scope, where, args = self._where(collection)
        db = self._db(doc)
        linking = {row[0] for row in links.incoming(db, Resolver(db), doc)} | {doc}
        where, args = where + " AND d.id NOT IN (SELECT value FROM json_each(?))", [*args, _ids(linking)]
        match = " OR ".join('"' + phrase.replace('"', '""') + '"' for phrase in phrases.values())
        return present(self._hits(self._keyword(match, k, scope, where, args)), fields, compact, None)

    # --- reading files again (see indexing.reindex) ---------------------------

    def reindex(self, paths: Sequence[str]) -> list[dict[str, Any]]:
        """Read files again now, after they were written, changed or deleted, and embed their new passages if their
        collection is embedded: the next search finds them as they are, the codes in memory following without a full
        load. Per file: {"id", "status"}, see indexing.reindex, with "chunks", its documents' passages, and
        "embedded", how many of them have vectors; and a "note" on anything that went wrong. Scans are left for `sift
        index` to read with OCR: a server runs one call at a time, which reading one would hold up for minutes."""
        done = indexing.reindex(self.catalog, paths, scans=False)
        for cid, (before, *after) in done.versions.items():
            removed = [chunk for chunk in done.removed if store.owner(chunk) == cid]
            self._patch(cid, before, after, removed=removed)
        note = self._embed_new([doc for _, docs in done.files for doc in docs])
        sql = """SELECT count(c.id), count(v.chunk) FROM chunks c LEFT JOIN vectors v ON v.chunk = c.id
                 WHERE c.doc IN (SELECT value FROM json_each(?))"""
        results: list[dict[str, Any]] = []
        for entry, docs in done.files:
            shown: dict[str, Any] = dict(entry)
            if entry["status"] in ("added", "changed", "unchanged"):
                chunks, embedded = self._db(docs[0]).execute(sql, (json.dumps(docs),)).fetchone() if docs else (0, 0)
                shown |= {"chunks": chunks, "embedded": embedded}
                if note and embedded < chunks:
                    shown["note"] = "; ".join(filter(None, [entry.get("note"), note]))
            results.append(shown)
        return results

    def _embed_new(self, docs: Iterable[int]) -> str | None:
        """Embed the passages of these documents that have no vector yet, in each collection already embedded, and add
        their codes to those in memory. Returns why some weren't, or None."""
        notes = [self._embed_new_in(c, db, group) for c, db, group in self._by_store(docs)]
        return "; ".join(dict.fromkeys(filter(None, notes))) or None

    def _embed_new_in(self, collection: Collection, db: Db, docs: list[int]) -> str | None:
        model = store.get_meta(db, "model")
        if model is None:  # never embedded: keyword search alone, and no model to load
            return None
        new = """chunks c JOIN docs d ON d.id = c.doc WHERE c.doc IN (SELECT value FROM json_each(?))
                 AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id) ORDER BY c.id"""
        rows = embed.passages(db, new, (json.dumps(docs),))
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
            after = store.vectors_state(db)
            db.commit()
        except BaseException:
            db.rollback()
            raise
        self._patch(collection.id, before, after, ids=[rows[i][0] for i in keep], matrix=matrix[keep])
        return None if intact.all() else "some passages could not be embedded (the model returned NaN); try again"

    def _patch(
        self,
        cid: int,
        before: str | None,
        after: Sequence[Any],
        removed: Sequence[int] = (),
        ids: Sequence[int] = (),
        matrix: Floats | None = None,
    ) -> None:
        """Bring the codes in memory of the collection `cid` from its store's version `before` to the state `after`
        (version, epoch, count of vectors: store.vectors_state, read in the change's transaction), a change this index
        made itself, without loading them all again. Codes of another version are followed by the next search anyway."""
        held, (version, epoch, count) = self.vectors.get(cid), after
        if held is None or held.version != before or before == version or cid in self._reading:
            return
        bits = codes(matrix) if matrix is not None else np.empty((0, held.codes.width), np.uint8)
        last = max([held.last, *map(int, ids)])
        self.vectors[cid] = _Held(version, held.codes.patched(list(removed), list(ids), bits), epoch, count, last)

    def fields(self, collection: Collection) -> list[str]:
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

    def status(self) -> dict[str, Any]:
        """The catalog, the model the active collections are embedded with, and every collection: its folder, whether
        it is active, its store and the bytes it takes, or that it is missing; and of an active one, its documents,
        chunks, those embedded, how its keyword postings stand, front-matter fields and links, or why its store can't
        be opened."""
        collections: list[dict[str, Any]] = []
        models: set[str | None] = set()
        for c in self.catalog.collections():
            entry: dict[str, Any] = {"name": c.name, "path": c.path, "active": c.active, "store": c.store}
            if not os.path.isfile(c.store):
                collections.append(entry | {"missing": True})
                continue
            entry["size"] = _size(c.store)
            if c.active:
                try:
                    db = self.catalog.opened(c)
                except LookupError as unusable:  # damaged, a newer sift's, locked or out of reach
                    collections.append(entry | {"error": str(unusable)})
                    continue
                try:
                    counted = store.counts(db)
                    linked, unresolved = self.link_counts(c)
                    entry |= {"docs": counted["docs"], "chunks": counted["chunks"], "embedded": counted["vectors"]}
                    entry["keywords"] = _postings_state(db)
                    entry["fields"] = self.fields(c)
                    entry |= {"links": linked, "unresolved": unresolved}
                    models.add(store.get_meta(db, "model"))
                except sqlite3.DatabaseError as error:  # damaged where it opened fine: said, the others still come
                    entry = {k: entry[k] for k in ("name", "path", "active", "store", "size")}
                    entry["error"] = store.damage(c.name, error)
            collections.append(entry)
        return {
            "index": str(self.catalog.path),
            "model": ", ".join(sorted(filter(None, models))) or None,
            "collections": collections,
        }

    def link_counts(self, collection: Collection) -> tuple[int, int]:
        """How many links the notes of a collection hold, and how many of them are unresolved."""
        db = self.catalog.store(collection)
        return db.execute("SELECT count(*) FROM links").fetchone()[0], sum(n for *_, n in links.unresolved(db))
