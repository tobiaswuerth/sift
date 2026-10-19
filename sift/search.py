"""The three search modes: exact keyword (BM25), semantic (exact GPU vector search) and hybrid with reranking."""

import re
import sqlite3
from pathlib import Path

import numpy as np

from . import store

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it of on or that the this to "
    "was what when where which who why with".split()
)
RRF_K = 60


def _where(db, collection=None, path=None, filters=None):
    """SQL condition on docs `d`. Filters match front-matter fields; list fields match if they contain the value."""
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
        if not isinstance(value, str | int | float):
            raise ValueError(f"filter {key!r} needs a single string or number, got {value!r}")
        sql.append(f"EXISTS (SELECT 1 FROM json_each(d.meta, '$.{key}') WHERE value = ?)")
        args.append(value)
    return " AND ".join(sql), args


def _check_k(k):
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _phrases(query):
    """Fallback for input that isn't valid FTS5 syntax: every word becomes an exact phrase, all required."""
    return " ".join('"' + w.replace('"', '""') + '"' for w in query.split())


def _any_word(query):
    """OR query over the meaningful words, used to collect keyword candidates for hybrid search."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


class Index:
    def __init__(self, db=None):
        self.db = db or store.connect()
        self.version, self.mat = None, None
        self._key = ("vectors", id(self))

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

    def _load(self):
        """Load all vectors into RAM; reload when the index has changed since the last load."""
        import torch

        from . import embed

        version = store.get_meta(self.db, "version")
        if version is not None and version == self.version:
            return
        n = self.db.execute("SELECT count(*) FROM vectors").fetchone()[0]
        if not n:
            raise LookupError("no embeddings yet, run `sift embed`")
        ids, mat = np.empty(n, np.int64), np.empty((n, int(store.get_meta(self.db, "dim"))), np.float16)
        for row, (chunk, vec) in enumerate(self.db.execute("SELECT chunk, vec FROM vectors ORDER BY chunk")):
            ids[row], mat[row] = chunk, np.frombuffer(vec, np.float16)
        host = torch.from_numpy(mat)
        self.ids, self.host = ids, host if embed.fp16() else host.float()  # CPUs are slow at fp16 matmul
        embed.discard(self._key)  # a GPU copy of the old vectors goes with them
        self.mat = None if embed.gpu() else self.host
        self.version = version

    def _vectors(self):
        """All vectors as one matrix where they are searched: on the GPU while in use, otherwise in RAM."""
        from . import embed

        self._load()
        embed.acquire(self._key, lambda: setattr(self, "mat", self.host.to(embed.DEVICE)), self._release)
        return self.ids, self.mat

    def _release(self):
        self.mat = None  # the copy in RAM stays

    def _semantic(self, query, k, where, args):
        import torch

        from .embed import encode

        ids, mat = self._vectors()
        q = torch.from_numpy(encode([query], query=True)[0]).to(mat.device, mat.dtype)
        scores = mat @ q
        if args:  # restrict to chunks of matching documents
            sql = f"SELECT c.id FROM chunks c JOIN docs d ON d.id = c.doc WHERE {where}"
            allowed = np.array([c for (c,) in self.db.execute(sql, args)], np.int64)
            pos = np.searchsorted(ids, allowed)
            pos = pos[(pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == allowed)]
            keep = torch.zeros(len(ids), dtype=torch.bool, device=mat.device)
            keep[torch.from_numpy(pos).to(mat.device)] = True
            scores = scores.masked_fill(~keep, float("-inf"))
        top = torch.topk(scores.float(), min(k, len(ids)))
        return [(int(ids[i]), s) for s, i in zip(top.values.tolist(), top.indices.tolist()) if s > float("-inf")]

    def _hits(self, ranked):
        sql = """SELECT co.name, d.path, d.title, c.line_start, c.line_end, fts.text FROM chunks c
                 JOIN docs d ON d.id = c.doc JOIN collections co ON co.id = d.collection JOIN fts ON fts.rowid = c.id
                 WHERE c.id = ?"""
        hits = []
        for chunk, score in ranked:
            row = self.db.execute(sql, (chunk,)).fetchone()
            if row is None:  # removed by a concurrent `sift index`
                continue
            name, path, title, start, end, text = row
            hit = {"id": f"{name}/{path}", "title": title, "lines": f"{start}-{end}", "score": round(score, 4)}
            hits.append(hit | {"text": text})
        return hits

    # --- public API ----------------------------------------------------------

    def search(self, query, k=10, collection=None, path=None, filters=None):
        """Exact keyword search, ranked by BM25. Accepts FTS5 syntax: "phrases", prefix*, AND, OR, NOT."""
        return self._hits(self._keyword(query, _check_k(k), *_where(self.db, collection, path, filters)))

    def vsearch(self, query, k=10, collection=None, path=None, filters=None):
        """Semantic search: cosine similarity between the query and every chunk."""
        return self._hits(self._semantic(query, _check_k(k), *_where(self.db, collection, path, filters)))

    def query(self, query, k=10, collection=None, path=None, filters=None, candidates=40):
        """Hybrid search for RAG: keyword and semantic candidates, fused by reciprocal rank, then reranked."""
        from .embed import rerank

        k, where = _check_k(k), _where(self.db, collection, path, filters)
        fused = {}
        for ranked in (self._keyword(_any_word(query), candidates, *where), self._semantic(query, candidates, *where)):
            for rank, (chunk, _) in enumerate(ranked):
                fused[chunk] = fused.get(chunk, 0) + 1 / (RRF_K + rank)
        hits = self._hits(sorted(fused.items(), key=lambda x: -x[1]))
        if not hits:
            return []
        scores = rerank(query, [f"{h['title']}\n\n{h['text']}" for h in hits])
        for h, s in zip(hits, scores):
            h["score"] = round(float(s), 4)
        return sorted(hits, key=lambda h: -h["score"])[:k]

    def get(self, id):
        """Full text of a document, by its `collection/path` id."""
        name, _, rel = id.partition("/")
        row = self.db.execute("SELECT path FROM collections WHERE name = ?", (name,)).fetchone()
        root = Path(row[0]).resolve() if row else None
        path = (root / rel).resolve() if root and rel else None
        if path is None or not path.is_relative_to(root):
            raise LookupError(f"unknown document {id!r}")
        try:
            return path.read_text("utf-8-sig", "replace")
        except OSError as e:
            raise LookupError(f"cannot read {id!r}: {e.strerror or e}") from None

    def status(self):
        rows = self.db.execute("""
            SELECT co.name, co.path, count(DISTINCT d.id), count(c.id), count(v.chunk) FROM collections co
            LEFT JOIN docs d ON d.collection = co.id LEFT JOIN chunks c ON c.doc = d.id
            LEFT JOIN vectors v ON v.chunk = c.id GROUP BY co.id ORDER BY co.name""").fetchall()
        return {
            "index": self.db.execute("PRAGMA database_list").fetchone()[2],
            "model": store.get_meta(self.db, "model"),
            "collections": [{"name": n, "path": p, "docs": d, "chunks": c, "embedded": e} for n, p, d, c, e in rows],
        }
