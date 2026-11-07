"""Keyword search's own postings, built from each store's FTS5 index, which stays the reference: per term, the chunks
holding it and how often, as arrays a search ranks with NumPy (see `topk`). FTS5 scores every match of a query, about
a microsecond each, so a common word took seconds on tens of millions of chunks; these let a search skip whatever
can't reach its best results (Block-Max Pruning, Mallia et al. 2024), at about a third of the size of FTS5's index.

- A segment holds the postings of a set of chunks: the base, read out of FTS5's index in one pass (in worker
  processes for a large store), and deltas, the chunks changed since, tokenized by FTS5 in a table of their own.
- A term's postings in a segment: its chunks, ascending, each with how often the term occurs in it (all columns, as
  FTS5's bm25 counts). A short list is one blob (zlib) in kw_terms. A long one, in the base, comes in blocks of BLOCK
  postings in kw_blocks, after a blob of its ranges: for each RANGE consecutive chunk ids holding the term, how many
  postings it has there and its best term score there, rounded up to a byte. Summed over a query's terms, these bound
  the score of every chunk in the range.
- Per segment, each chunk's length in tokens (all columns) and its document, in blocks (kw_chunks).
- A chunk whose postings changed after a segment was built is masked: its postings there count no more.
- Statistics (chunks, tokens, document frequencies) are the segments', masked postings included, as Lucene's are: the
  same as FTS5's until something changes, and the same again after the next rebuild.

Updates follow FTS5: store.keywords marks every chunk whose entries change dirty, numbered by the change. `update`
adds a delta of those chunks, or builds anew once deltas would grow large; a search over a store with dirty chunks
reads FTS5, as before.
"""

import itertools
import json
import multiprocessing
import sqlite3
import struct
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import console, store

RANGE_SHIFT = 7  # a range is 128 consecutive chunk ids: on PubMed, quicker and smaller than 64 (benchmarks.md); at
# most 8, as a range's postings are counted in a byte
BLOCK = 4096  # postings per block of a long list: a few kilobytes read for a few postings
LONG = 4096  # postings from which a base list comes in blocks, with ranges; shorter ones are read whole
CHUNK_BLOCK = 65536  # chunks per block of lengths and documents
K1, B = 1.2, 0.75  # the parameters of FTS5's bm25()
TOP = K1 + 1  # the most a term score can be, idf aside
MOST_SEGMENTS = 8  # the base and its deltas; one more, and the postings are built anew
REBUILD_SHARE = 0.05  # changed chunks of all, from which the postings are built anew rather than given a delta
SMALL = 100_000  # chunks; a store this small is built anew on every update, in a moment
PARALLEL = 1_000_000  # chunks from which a build reads FTS5's index in worker processes
FETCH = 1_000_000  # rows fetched at a time
LEVEL = 6  # zlib's

_BLOCK = struct.Struct("<qIB")  # first chunk id, postings, bytes per count (1, 2 or 4)
_WIDTHS = {1: np.uint8, 2: np.uint16, 4: np.uint32}
_RANGES = struct.Struct("<qI")  # first range id, ranges
_CHUNKS = struct.Struct("<qqI")  # first chunk id, its document, chunks


# --- encoding ------------------------------------------------------------------


def _gaps(values):
    """Ascending ids as the gaps after the first, which fit 32 bits: a collection hands out ids in order."""
    gaps = np.diff(values)
    if len(gaps) and gaps.max() >= 1 << 32:
        raise ValueError("ids more than 2**32 apart: too many for keyword postings")
    return gaps.astype(np.uint32)


def _ascending(first, gaps, n):
    values = np.empty(n, np.int64)
    values[0] = first
    np.cumsum(gaps, out=values[1:])
    values[1:] += first
    return values


def encode_block(ids, tf):
    """Postings as stored: chunk ids (ascending, at least one) and how often the term occurs in each (1 or more)."""
    most = int(tf.max())
    width = 1 if most < 1 << 8 else 2 if most < 1 << 16 else 4
    counts = tf.astype(_WIDTHS[width])
    return zlib.compress(_BLOCK.pack(int(ids[0]), len(ids), width) + _gaps(ids).tobytes() + counts.tobytes(), LEVEL)


def decode_block(data):
    """(chunk ids, counts), as int64, of an encoded block."""
    raw = zlib.decompress(data)
    first, n, width = _BLOCK.unpack_from(raw)
    at = _BLOCK.size
    ids = _ascending(first, np.frombuffer(raw, np.uint32, n - 1, at), n)
    tf = np.frombuffer(raw, _WIDTHS[width], n, at + 4 * (n - 1)).astype(np.int64)
    return ids, tf


def encode_ranges(rid, counts, best):
    """A long list's ranges: their ids (ascending), postings in each (1 to RANGE), and best term score, as bytes."""
    return zlib.compress(
        _RANGES.pack(int(rid[0]), len(rid))
        + _gaps(rid).tobytes()
        + (counts - 1).astype(np.uint8).tobytes()
        + best.astype(np.uint8).tobytes(),
        LEVEL,
    )


def decode_ranges(data):
    """(range ids, where each range's postings start and end in the list, its best term score) of encoded ranges."""
    raw = zlib.decompress(data)
    first, n = _RANGES.unpack_from(raw)
    at = _RANGES.size
    rid = _ascending(first, np.frombuffer(raw, np.uint32, n - 1, at), n)
    at += 4 * (n - 1)
    ends = np.cumsum(np.frombuffer(raw, np.uint8, n, at).astype(np.int64) + 1)
    best = np.frombuffer(raw, np.uint8, n, at + n).astype(np.float64) * (TOP / 255)
    return rid, np.r_[0, ends[:-1]], ends, best


def quantized(scores):
    """Term scores as bytes, rounded up, so that a bound stays one."""
    return np.minimum(np.ceil(scores * (255 / TOP)), 255).astype(np.uint8)


def encode_chunks(ids, lengths, docs):
    """Chunks as stored: ids (ascending, at least one), lengths in tokens, documents."""
    return zlib.compress(
        _CHUNKS.pack(int(ids[0]), int(docs[0]), len(ids))
        + _gaps(ids).tobytes()
        + lengths.astype(np.uint32).tobytes()
        + np.diff(docs).astype(np.int64).tobytes(),
        LEVEL,
    )


def decode_chunks(data):
    """(ids, lengths, documents), as int64, of encoded chunks."""
    raw = zlib.decompress(data)
    first, doc, n = _CHUNKS.unpack_from(raw)
    at = _CHUNKS.size
    ids = _ascending(first, np.frombuffer(raw, np.uint32, n - 1, at), n)
    at += 4 * (n - 1)
    lengths = np.frombuffer(raw, np.uint32, n, at).astype(np.int64)
    docs = _ascending(doc, np.frombuffer(raw, np.int64, n - 1, at + 4 * n), n)
    return ids, lengths, docs


def term_scores(tf, lengths, average):
    """BM25's term scores as FTS5's bm25 computes them, without the idf: from counts, lengths, the average length."""
    tf = np.asarray(tf, np.float64)
    return tf * TOP / (tf + K1 * (1 - B + B * np.asarray(lengths) / average))


# --- reading FTS5 --------------------------------------------------------------


def varints(blob):
    """The numbers of a blob of SQLite varints, as FTS5 stores sizes, and the offset each starts at: 7 bits a byte, the
    high bit set on all but the last. Vectorised, for a blob of many."""
    raw = np.frombuffer(blob, np.uint8)
    last = (raw & 0x80) == 0
    starts = np.flatnonzero(np.r_[True, last[:-1]])
    which = np.cumsum(np.r_[True, last[:-1]]) - 1  # the number each byte belongs to
    place = np.arange(len(raw)) - starts[which]
    size = np.diff(np.r_[starts, len(raw)])
    weights = (raw & 0x7F).astype(np.float64) * np.exp2(7.0 * (size[which] - 1 - place))
    return np.bincount(which, weights=weights, minlength=len(starts)).astype(np.int64), starts


def _lengths(db, table):
    """(chunk ids, lengths) as an FTS5 docsize table has them, all columns' tokens, a batch at a time."""
    ids, lengths = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    rows = db.execute(f"SELECT id, sz FROM {table} ORDER BY id")
    for batch in iter(lambda: rows.fetchmany(FETCH), []):
        blobs = [row[1] for row in batch]
        numbers, starts = varints(b"".join(blobs))
        ends = np.cumsum(np.fromiter(map(len, blobs), np.int64, len(blobs)))
        row = np.searchsorted(ends, starts, side="right")
        ids.append(np.fromiter((r[0] for r in batch), np.int64, len(batch)))
        lengths.append(np.bincount(row, weights=numbers, minlength=len(batch)).astype(np.int64))
    return np.concatenate(ids), np.concatenate(lengths)


def _documents(db, ids):
    """The documents of these chunks (ascending), a batch of the chunks table at a time."""
    found, owners = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    rows = db.execute("SELECT id, doc FROM chunks ORDER BY id")
    for batch in iter(lambda: rows.fetchmany(FETCH), []):
        found.append(np.fromiter((r[0] for r in batch), np.int64, len(batch)))
        owners.append(np.fromiter((r[1] for r in batch), np.int64, len(batch)))
    found, owners = np.concatenate(found), np.concatenate(owners)
    at = np.minimum(np.searchsorted(found, ids), max(len(found) - 1, 0))
    return np.where(found[at] == ids, owners[at], -1) if len(found) else np.full(len(ids), -1, np.int64)


def _postings(rows):
    """(term, chunk ids, counts) per term of FTS5 vocabulary rows (term, chunk), one per occurrence, in its order:
    by term, then chunk."""
    term, parts = None, []
    for batch in iter(lambda: rows.fetchmany(FETCH), []):
        names = np.array([row[0] for row in batch], dtype=object)
        chunks = np.fromiter((row[1] for row in batch), np.int64, len(batch))
        cuts = np.r_[0, np.flatnonzero(names[1:] != names[:-1]) + 1, len(batch)]
        for start, end in zip(cuts[:-1].tolist(), cuts[1:].tolist()):
            if names[start] != term:
                if term is not None:
                    yield term, *_counted(np.concatenate(parts))
                term, parts = names[start], []
            parts.append(chunks[start:end])
    if term is not None:
        yield term, *_counted(np.concatenate(parts))


def _counted(chunks):
    """Each chunk once, with how often it was listed."""
    cuts = np.r_[0, np.flatnonzero(np.diff(chunks)) + 1]
    return chunks[cuts], np.diff(np.r_[cuts, len(chunks)])


# --- building ------------------------------------------------------------------


class _Lengths:
    """The lengths of the chunks a segment holds, by id."""

    def __init__(self, ids, lengths):
        self.ids, self.lengths = ids, lengths

    def of(self, chunks):
        """The lengths of these chunks, and which of them the segment holds: one added after the lengths were read is
        dirty, and a delta has it once it is updated."""
        if not len(self.ids):
            return np.zeros(len(chunks), np.int64), np.zeros(len(chunks), bool)
        at = np.minimum(np.searchsorted(self.ids, chunks), len(self.ids) - 1)
        return np.asarray(self.lengths[at]), np.asarray(self.ids[at] == chunks)


def _encoded(term, chunks, tf, held, average, ranged):
    """A term's postings, encoded: (term, df, blob) for a short list, (term, df, ranges, blocks) for a long one."""
    lengths, keep = held.of(chunks)
    chunks, tf, lengths = chunks[keep], tf[keep], lengths[keep]
    if not len(chunks):
        return None
    if not ranged or len(chunks) < LONG:
        return term, len(chunks), encode_block(chunks, tf)
    rid = chunks >> RANGE_SHIFT
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    best = np.maximum.reduceat(term_scores(tf, lengths, average), starts)
    counts = np.diff(np.r_[starts, len(chunks)])
    blocks = [encode_block(chunks[i : i + BLOCK], tf[i : i + BLOCK]) for i in range(0, len(chunks), BLOCK)]
    return term, len(chunks), encode_ranges(rid[starts], counts, quantized(best)), blocks


def _read_only(path):
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=store.TIMEOUT)


def _shard(job):
    """The encoded postings of the terms from `lo` up to `hi` (None: all after), read out of a store's FTS5 index."""
    path, lo, hi, lengths_file, average = job
    db = _read_only(path)
    try:
        held = _Lengths(*np.load(lengths_file, mmap_mode="r"))  # one copy for all workers: the OS's
        db.execute("CREATE VIRTUAL TABLE temp.kw_vocabulary USING fts5vocab(main, fts, instance)")
        where, args = ("term >= ? AND term < ?", (lo, hi)) if hi is not None else ("term >= ?", (lo,))
        rows = db.execute(f"SELECT term, doc FROM temp.kw_vocabulary WHERE {where}", args)
        return [e for p in _postings(rows) if (e := _encoded(*p, held, average, True))]
    finally:
        db.close()


def _shards():
    """Term ranges [lo, hi) covering every term: one per one- and two-character prefix of digits and letters, the rest
    (non-ASCII) last. FTS5 orders terms by their UTF-8 bytes; its tokens hold letters and digits, lowercase."""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    bounds = sorted(set(alphabet) | {a + b for a in alphabet for b in alphabet})
    return [("", bounds[0]), *itertools.pairwise(bounds), (bounds[-1], "{"), ("{", None)]


def build(db, path, name="", workers=0):
    """Build a store's postings anew, out of its FTS5 index: one base segment, which searches read once it is done.
    Chunks that change meanwhile stay dirty, so searches read FTS5 for this store until the next `update`. A large
    store is read by `workers` processes."""
    db.execute("BEGIN IMMEDIATE")
    try:
        change = int(store.get_meta(db, "kw_change", 0))
        store.set_meta(db, "kw_built", "building")  # changes are marked from now on: see store.keywords
        for table in ("kw_terms", "kw_blocks", "kw_chunks", "kw_segments", "kw_masked"):
            db.execute(f"DELETE FROM {table}")
        db.commit()
    except BaseException:
        db.rollback()
        raise
    ids, lengths = _lengths(db, "fts_docsize")
    average = lengths.sum() / len(ids) if len(ids) else 1.0
    parallel = workers > 1 and len(ids) >= PARALLEL
    jobs = _shards() if parallel else [("", None)]
    keys = itertools.count(1)
    with tempfile.TemporaryDirectory() as folder:
        lengths_file = Path(folder) / "lengths.npy"
        np.save(lengths_file, np.stack([ids, lengths]))
        work = [(str(path), lo, hi, str(lengths_file), average) for lo, hi in jobs]
        spawn = multiprocessing.get_context("spawn")
        with (
            ProcessPoolExecutor(workers, mp_context=spawn) if parallel else nullcontext() as pool,
            console.Progress(len(jobs), "parts", prefix=f"{name}: keyword postings: ")
            if parallel
            else nullcontext() as progress,
        ):
            for done, encoded in enumerate((pool.map if pool else map)(_shard, work), 1):
                _write_terms(db, 1, encoded, keys)
                db.commit()
                if progress:
                    progress.update(done)
    db.execute("BEGIN IMMEDIATE")
    try:
        _write_chunks(db, 1, ids, lengths, _documents(db, ids))
        db.execute("INSERT INTO kw_segments(id, chunks, tokens) VALUES(1, ?, ?)", (len(ids), int(lengths.sum())))
        store.set_meta(db, "kw_built", change)
        db.execute("DELETE FROM kw_dirty WHERE change <= ?", (change,))
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _write_terms(db, segment, encoded, keys):
    rows, blocks = [], []
    for entry in encoded:
        if len(entry) == 3:
            term, df, data = entry
            rows.append((term, segment, df, None, data))
            continue
        term, df, ranges, parts = entry
        key = next(keys)
        rows.append((term, segment, df, key, None))
        blocks += [(key << 24, ranges), *((key << 24 | n, part) for n, part in enumerate(parts, 1))]
    db.executemany("INSERT INTO kw_terms(term, segment, df, list, data) VALUES(?, ?, ?, ?, ?)", rows)
    db.executemany("INSERT INTO kw_blocks(id, data) VALUES(?, ?)", blocks)


def _write_chunks(db, segment, ids, lengths, docs):
    db.executemany(
        "INSERT INTO kw_chunks(segment, first, data) VALUES(?, ?, ?)",
        [
            (segment, int(ids[i]), encode_chunks(*(a[i : i + CHUNK_BLOCK] for a in (ids, lengths, docs))))
            for i in range(0, len(ids), CHUNK_BLOCK)
        ],
    )


# --- updating ------------------------------------------------------------------


def built(db):
    """Whether the store's postings are built (if behind FTS5 meanwhile, see `ready`)."""
    return str(store.get_meta(db, "kw_built", "")).isdigit()


def behind(db):
    """How many chunks changed since the store's postings were last updated."""
    return db.execute("SELECT count(*) FROM kw_dirty").fetchone()[0]


def ready(db):
    """Whether searches can read the store's postings: built, and no chunk changed since."""
    return built(db) and not db.execute("SELECT 1 FROM kw_dirty LIMIT 1").fetchone()


def update(db, path, name="", workers=0):
    """Bring a store's postings up to FTS5: build them if there are none, else add a delta of the chunks changed
    since, or build them anew when deltas would grow too many or too large (MOST_SEGMENTS, REBUILD_SHARE)."""
    if not built(db):
        build(db, path, name, workers)
        return
    dirty = behind(db)
    if not dirty:
        return
    chunks = store.counts(db)["chunks"]
    segments = db.execute("SELECT count(*) FROM kw_segments").fetchone()[0]
    masked = db.execute("SELECT count(*) FROM kw_masked").fetchone()[0]
    if chunks < SMALL or segments >= MOST_SEGMENTS or masked + dirty > REBUILD_SHARE * chunks:
        build(db, path, name, workers)
    else:
        _delta(db)


def _delta(db):
    """Add a segment of the chunks changed since the postings were updated, tokenized as FTS5 tokenizes them, and mask
    their postings in the segments before."""
    change = int(store.get_meta(db, "kw_change", 0))
    dirty = [c for (c,) in db.execute("SELECT chunk FROM kw_dirty WHERE change <= ?", (change,))]
    encoded, ids, lengths = _tokenized(db, dirty)
    db.execute("BEGIN IMMEDIATE")
    try:
        segment, first_key = db.execute(
            "SELECT coalesce(max(id), 0) + 1, (SELECT coalesce(max(id) >> 24, 0) + 1 FROM kw_blocks) FROM kw_segments"
        ).fetchone()
        _write_terms(db, segment, encoded, itertools.count(first_key))
        if len(ids):
            _write_chunks(db, segment, ids, lengths, _documents(db, ids))
        db.execute(
            "INSERT INTO kw_segments(id, chunks, tokens) VALUES(?, ?, ?)", (segment, len(ids), int(lengths.sum()))
        )
        db.executemany(
            "INSERT INTO kw_masked(chunk, segment) VALUES(?, ?) ON CONFLICT(chunk) DO UPDATE SET segment = ?",
            [(c, segment, segment) for c in dirty],
        )
        db.executemany("DELETE FROM kw_dirty WHERE chunk = ? AND change <= ?", [(c, change) for c in dirty])
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _tokenized(db, chunks):
    """The postings of these chunks as they are now, tokenized by FTS5 in a table of their own: (encoded terms, chunk
    ids, lengths). Chunks deleted meanwhile have none."""
    db.execute(f"CREATE VIRTUAL TABLE temp.kw_tokens USING fts5(title, text, meta, tokenize='{store.TOKENIZER}')")
    try:
        db.execute(
            "INSERT INTO temp.kw_tokens(rowid, title, text, meta) SELECT id, title, text, meta FROM chunk_text "
            "WHERE id IN (SELECT value FROM json_each(?))",
            (json.dumps(chunks),),
        )
        ids, lengths = _lengths(db, "temp.kw_tokens_docsize")
        average = lengths.sum() / len(ids) if len(ids) else 1.0
        db.execute("CREATE VIRTUAL TABLE temp.kw_tokens_vocabulary USING fts5vocab(temp, kw_tokens, instance)")
        rows = db.execute("SELECT term, doc FROM temp.kw_tokens_vocabulary")
        held = _Lengths(ids, lengths)
        return [e for p in _postings(rows) if (e := _encoded(*p, held, average, False))], ids, lengths
    finally:
        db.commit()  # the temporary table's rows: it goes next
        db.execute("DROP TABLE IF EXISTS temp.kw_tokens_vocabulary")
        db.execute("DROP TABLE IF EXISTS temp.kw_tokens")


# --- reading -------------------------------------------------------------------


class Term:
    """A term's postings in one segment: `df` of them. A short list is read whole (everything); a long one by its
    ranges, then the postings of some of them."""

    def __init__(self, reader, segment, df, key, data):
        self.reader, self.segment, self.df, self.key, self.data = reader, segment, df, key, data

    @property
    def long(self):
        return self.key is not None

    def everything(self):
        """(chunk ids, counts), all of them."""
        if not self.long:
            return decode_block(self.data)
        _, starts, ends, _ = self.ranges()
        return self.postings(starts[:1], ends[-1:])

    def ranges(self):
        """(range ids, where each range's postings start and end in the list, its best term score)."""
        return self.reader.ranges(self.key)

    def postings(self, starts, ends):
        """(chunk ids, counts) of the postings from each start up to its end, in the list's order."""
        first, last = starts // BLOCK, (ends - 1) // BLOCK
        n = last - first + 1
        need = np.unique(np.repeat(first - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(n.sum()))
        blocks = [self.reader.block(self.key, int(b)) for b in need]
        ids = np.concatenate([b[0] for b in blocks])
        tf = np.concatenate([b[1] for b in blocks])
        at = np.zeros(int(need[-1]) + 1, np.int64)
        at[need] = np.arange(len(need)) * BLOCK  # every block but a list's last holds BLOCK postings
        size = ends - starts
        begin = at[first] + starts - first * BLOCK
        idx = np.repeat(begin - np.r_[0, np.cumsum(size)[:-1]], size) + np.arange(size.sum())
        return ids[idx], tf[idx]


class Reader:
    """A store's postings as searches read them: segments and their statistics, terms, chunks' lengths and documents,
    and which postings are masked. Made for a state of the postings (`state`), and read through caches."""

    def __init__(self, db):
        self.db = db
        self.segments = {s: (n, tokens) for s, n, tokens in db.execute("SELECT id, chunks, tokens FROM kw_segments")}
        masked = db.execute("SELECT chunk, segment FROM kw_masked ORDER BY chunk").fetchall()
        self.masked = np.array([m[0] for m in masked], np.int64)
        self.masked_in = np.array([m[1] for m in masked], np.int64)
        firsts = {}
        for segment, first, rowid in db.execute("SELECT segment, first, id FROM kw_chunks ORDER BY segment, first"):
            firsts.setdefault(segment, []).append((first, rowid))
        self._firsts = {s: (np.array([f for f, _ in v]), [r for _, r in v]) for s, v in firsts.items()}
        self.ranges = lru_cache(maxsize=1024)(self._ranges)
        self.block = lru_cache(maxsize=4096)(self._block)
        self._chunk_block = lru_cache(maxsize=1024)(self._read_chunks)

    @staticmethod
    def state(db):
        """What a Reader is made for: in another state, the postings need another Reader."""
        return store.get_meta(db, "kw_built"), store.get_meta(db, "kw_change")

    @property
    def chunks(self):
        return sum(n for n, _ in self.segments.values())

    @property
    def tokens(self):
        return sum(t for _, t in self.segments.values())

    def term(self, term):
        """The term's postings, a Term per segment that has any."""
        rows = self.db.execute("SELECT segment, df, list, data FROM kw_terms WHERE term = ?", (term,))
        return [Term(self, *row) for row in rows]

    def _ranges(self, key):
        return decode_ranges(self._blob(key << 24))

    def _block(self, key, n):
        return decode_block(self._blob(key << 24 | n + 1))

    def _blob(self, rowid):
        return self.db.execute("SELECT data FROM kw_blocks WHERE id = ?", (rowid,)).fetchone()[0]

    def chunks_of(self, segment, ids):
        """(lengths, documents) of these chunks, by id, as the segment holds them."""
        firsts, rowids = self._firsts[segment]
        which = np.searchsorted(firsts, ids, side="right") - 1
        lengths, docs = np.empty(len(ids), np.int64), np.empty(len(ids), np.int64)
        for w in np.unique(which).tolist():
            held_ids, held_lengths, held_docs = self._chunk_block(rowids[w])
            sel = which == w
            at = np.searchsorted(held_ids, ids[sel])
            lengths[sel], docs[sel] = held_lengths[at], held_docs[at]
        return lengths, docs

    def _read_chunks(self, rowid):
        return decode_chunks(self.db.execute("SELECT data FROM kw_chunks WHERE id = ?", (rowid,)).fetchone()[0])

    def alive(self, segment, ids):
        """Which of these postings of the segment count: those of chunks not masked since it was built."""
        if not len(self.masked):
            return np.ones(len(ids), bool)
        at = np.minimum(np.searchsorted(self.masked, ids), len(self.masked) - 1)
        return ~((self.masked[at] == ids) & (self.masked_in[at] > segment))
