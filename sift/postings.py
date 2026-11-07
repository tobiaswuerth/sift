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
- Per segment, each chunk's length in tokens (all columns), in blocks of LENGTHS consecutive chunk ids (kw_lengths):
  a search reads those of the chunks it scores. Their documents come from the chunks table, for those found.
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
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import console, store

RANGE_SHIFT = 7  # a range is 128 consecutive chunk ids: on PubMed, quicker and smaller than 64 (benchmarks.md); at
# most 8, as a range's postings are counted in a byte
BLOCK = 4096  # postings per block of a long list: a few kilobytes read for a few postings
LONG = 4096  # postings from which a base list comes in blocks, with ranges; shorter ones are read whole
LENGTHS_SHIFT = 10  # a block of lengths covers 1024 consecutive chunk ids: about a kilobyte or two to read
K1, B = 1.2, 0.75  # the parameters of FTS5's bm25()
TOP = K1 + 1  # the most a term score can be, idf aside
MOST_SEGMENTS = 8  # the base and its deltas; one more, and the deltas are merged into one
REBUILD_SHARE = 0.05  # changed chunks of all, from which the postings are built anew rather than given a delta
SMALL = 2_000  # chunks; a store this small is built anew by `sift index` rather than given a delta: in a moment
PARALLEL = 1_000_000  # chunks from which a build reads FTS5's index in worker processes
FETCH = 1_000_000  # rows fetched at a time
LEVEL = 6  # zlib's

_BLOCK = struct.Struct("<qIB")  # first chunk id, postings, bytes per count (1, 2 or 4)
_WIDTHS = {1: np.uint8, 2: np.uint16, 4: np.uint32}
_RANGES = struct.Struct("<qI")  # first range id, ranges


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
    """(chunk ids as int64, counts as stored: mostly a byte each) of an encoded block."""
    raw = zlib.decompress(data)
    first, n, width = _BLOCK.unpack_from(raw)
    at = _BLOCK.size
    ids = _ascending(first, np.frombuffer(raw, np.uint32, n - 1, at), n)
    return ids, np.frombuffer(raw, _WIDTHS[width], n, at + 4 * (n - 1))


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


def encode_lengths(ids, lengths):
    """{block: data} of chunks' lengths (ids ascending): per block of 2**LENGTHS_SHIFT consecutive ids, the length of
    each, 0 for an id no chunk of the segment has."""
    if not len(ids):
        return {}
    blocks = ids >> LENGTHS_SHIFT
    cuts = np.r_[0, np.flatnonzero(np.diff(blocks)) + 1, len(ids)]
    encoded = {}
    for start, end in itertools.pairwise(cuts.tolist()):
        dense = np.zeros(1 << LENGTHS_SHIFT, np.uint32)
        dense[ids[start:end] & ((1 << LENGTHS_SHIFT) - 1)] = lengths[start:end]
        encoded[int(blocks[start])] = zlib.compress(dense.tobytes(), LEVEL)
    return encoded


def decode_lengths(data):
    """The lengths of a block's chunk ids, by id from its first, as int64 (0 for none)."""
    return np.frombuffer(zlib.decompress(data), np.uint32).astype(np.int64)


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
    """Term ranges [lo, hi) covering every term: one per prefix of up to three digits and letters, the rest
    (non-ASCII) last. FTS5 orders terms by their UTF-8 bytes; its tokens hold letters and digits, lowercase. With two
    characters, one range held "the", "that", "this" and more, a tenth of all words, and the build waited on it."""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    two = {a + b for a in alphabet for b in alphabet}
    bounds = sorted(set(alphabet) | two | {ab + c for ab in two for c in alphabet})
    return [("", bounds[0]), *itertools.pairwise(bounds), (bounds[-1], "{"), ("{", None)]


@contextmanager
def _alone(path):
    """True while this process alone changes the postings of the store at `path`, False if another sift is changing
    them: an exclusive transaction on a database of its own beside the store, which the OS ends with the process. A
    build reads the store for minutes, without holding the store's own lock, so other writers go on meanwhile."""
    lock = sqlite3.connect(f"{path}.postings-lock", timeout=0, isolation_level=None)
    try:
        try:
            lock.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError:
            yield False
            return
        try:
            yield True
        finally:
            lock.execute("ROLLBACK")
    finally:
        lock.close()


def build(db, path, name="", workers=0):
    """Build a store's postings anew, out of its FTS5 index: one base segment, which searches read once it is done.
    Chunks that change meanwhile stay dirty, so searches read FTS5 for this store until the next `update`. A large
    store is read by `workers` processes. Another sift building them already? Then nothing happens here."""
    with _alone(path) as alone:
        if not alone:
            console.say(f"{name}: another sift is updating its keyword postings; searches read FTS5 meanwhile")
            return
        _build(db, path, name, workers)


def _build(db, path, name, workers):
    db.execute("BEGIN IMMEDIATE")
    try:
        change = int(store.get_meta(db, "kw_change", 0))
        store.set_meta(db, "kw_built", "building")  # changes are marked from now on: see store.keywords
        for table in ("kw_terms", "kw_blocks", "kw_lengths", "kw_segments", "kw_masked"):
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
        if parallel:
            _in_parallel(db, name, work, workers, keys)
        else:
            for encoded in map(_shard, work):
                _write_terms(db, 1, encoded, keys)
                db.commit()
    db.execute("BEGIN IMMEDIATE")
    try:
        _write_lengths(db, 1, ids, lengths)
        db.execute("INSERT INTO kw_segments(id, chunks, tokens) VALUES(1, ?, ?)", (len(ids), int(lengths.sum())))
        store.set_meta(db, "kw_built", change)
        db.execute("DELETE FROM kw_dirty WHERE change <= ?", (change,))
        db.commit()
    except BaseException:
        db.rollback()
        raise


PROGRESS_LINES = 200  # updates of a build's progress at most: printing is slow on Windows consoles


def _in_parallel(db, name, work, workers, keys):
    """Run the shards in worker processes, writing each one's postings as it comes."""
    spawn = multiprocessing.get_context("spawn")
    step = max(1, len(work) // PROGRESS_LINES)
    with (
        ProcessPoolExecutor(workers, mp_context=spawn) as pool,
        console.Progress(len(work), "parts", prefix=f"{name}: keyword postings: ") as progress,
    ):
        for done, future in enumerate(as_completed([pool.submit(_shard, job) for job in work]), 1):
            _write_terms(db, 1, future.result(), keys)
            db.commit()
            if done % step == 0 or done == len(work):
                progress.update(done)


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


def _write_lengths(db, segment, ids, lengths):
    db.executemany(
        "INSERT INTO kw_lengths(segment, block, data) VALUES(?, ?, ?)",
        [(segment, block, data) for block, data in encode_lengths(ids, lengths).items()],
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


def update(db, path, name="", workers=0, rebuild=True):
    """Bring a store's postings up to FTS5: build them if there are none, else add a delta of the chunks changed
    since; when deltas grow too many, merge them into one (MOST_SEGMENTS), and when the chunks changed since the
    postings were built grow too many, build them anew (REBUILD_SHARE). Without `rebuild`, as for a file an agent
    reindexes, nothing is built: that reads the whole store, so `sift index` does it. Another sift changing the
    postings meanwhile? Then nothing happens here, and searches read FTS5 until the next update."""
    if built(db) and not behind(db):
        return
    with _alone(path) as alone:
        if not alone:
            console.say(f"{name}: another sift is updating its keyword postings; searches read FTS5 meanwhile")
        elif not built(db):
            if rebuild:
                _build(db, path, name, workers)
        elif dirty := behind(db):
            chunks = store.counts(db)["chunks"]
            segments = db.execute("SELECT count(*) FROM kw_segments").fetchone()[0]
            masked = db.execute("SELECT count(*) FROM kw_masked").fetchone()[0]
            if rebuild and (chunks < SMALL or masked + dirty > REBUILD_SHARE * chunks):
                _build(db, path, name, workers)
            else:
                _delta(db, merge=segments >= MOST_SEGMENTS)


def _delta(db, merge=False):
    """Add a segment of the chunks changed since the postings were updated, tokenized as FTS5 tokenizes them, and mask
    their postings in the segments before. To `merge` the deltas, the chunks of those before go into it too, and it
    takes their place, after the base."""
    change = int(store.get_meta(db, "kw_change", 0))
    dirty = [c for (c,) in db.execute("SELECT chunk FROM kw_dirty WHERE change <= ?", (change,))]
    taken = sorted(set(dirty) | set(_delta_chunks(db))) if merge else dirty
    encoded, ids, lengths = _tokenized(db, taken)
    db.execute("BEGIN IMMEDIATE")
    try:
        if merge:  # the deltas go; their chunks, masked in the base, are the new one's
            db.execute("DELETE FROM kw_terms WHERE segment > 1")
            db.execute("DELETE FROM kw_lengths WHERE segment > 1")
            db.execute("DELETE FROM kw_segments WHERE id > 1")
            db.execute("UPDATE kw_masked SET segment = 2")
        segment, first_key = db.execute(
            "SELECT coalesce(max(id), 0) + 1, (SELECT coalesce(max(id) >> 24, 0) + 1 FROM kw_blocks) FROM kw_segments"
        ).fetchone()
        _write_terms(db, segment, encoded, itertools.count(first_key))
        _write_lengths(db, segment, ids, lengths)
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


def _delta_chunks(db):
    """The chunks the deltas hold: those with a length."""
    held = []
    for block, data in db.execute("SELECT block, data FROM kw_lengths WHERE segment > 1"):
        held += ((block << LENGTHS_SHIFT) + np.flatnonzero(decode_lengths(data))).tolist()
    return held


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
        """(chunk ids, counts) of the postings from each start up to its end (ascending), in the list's order: only
        those, from the blocks that hold them."""
        idx = slices(starts, ends)
        ids, tf = np.empty(len(idx), np.int64), np.empty(len(idx), np.int64)
        block = idx // BLOCK
        cuts = np.r_[0, np.flatnonzero(np.diff(block)) + 1, len(idx)]
        for start, end in itertools.pairwise(cuts.tolist()):
            n = int(block[start])
            held_ids, held_tf = self.reader.block(self.key, n)
            at = idx[start:end] - n * BLOCK
            ids[start:end], tf[start:end] = held_ids[at], held_tf[at]
        return ids, tf


def slices(starts, ends):
    """The indexes from each start up to its end, in order."""
    size = ends - starts
    return np.repeat(starts - np.r_[0, np.cumsum(size)[:-1]], size) + np.arange(size.sum())


class _Cache:
    """Values by key, the most recently used kept up to `budget` bytes."""

    def __init__(self, budget):
        self.budget, self.held, self.size = budget, OrderedDict(), 0

    def get(self, key, make):
        if key in self.held:
            self.held.move_to_end(key)
            return self.held[key]
        value = self.held[key] = make()
        self.size += sum(part.nbytes for part in value)
        while self.size > self.budget and len(self.held) > 1:
            _, old = self.held.popitem(last=False)
            self.size -= sum(part.nbytes for part in old)
        return value


BLOCK_CACHE = 512 << 20  # bytes of decoded blocks kept: a long question over PubMed reads about 150 MB of them
_BLOCKS = _Cache(BLOCK_CACHE)  # shared by every reader: one budget for the process
_readers = itertools.count()


class Reader:
    """A store's postings as searches read them: segments and their statistics, terms, chunks' lengths, and which
    postings are masked. Made for a state of the postings (`state`), and read through caches."""

    def __init__(self, db):
        self.db = db
        self.segments = {s: (n, tokens) for s, n, tokens in db.execute("SELECT id, chunks, tokens FROM kw_segments")}
        masked = db.execute("SELECT chunk, segment FROM kw_masked ORDER BY chunk").fetchall()
        self.masked = np.array([m[0] for m in masked], np.int64)
        self.masked_in = np.array([m[1] for m in masked], np.int64)
        self.ranges = lru_cache(maxsize=1024)(self._ranges)
        self._me = next(_readers)
        self._dense = {}  # segment: (its first block, the lengths read so far by id from it, the blocks read)

    def block(self, key, n):
        """Block n (from 0) of a long list, decoded: through the cache all readers share."""
        return _BLOCKS.get((self._me, key, n), lambda: self._block(key, n))

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

    def lengths(self, segment, ids):
        """The lengths of these chunks, by id, as the segment holds them: their blocks read once, into an array of
        the segment's ids, which takes memory only where written."""
        if segment not in self._dense:
            first, last = self.db.execute(
                "SELECT min(block), max(block) FROM kw_lengths WHERE segment = ?", (segment,)
            ).fetchone()
            blocks = last - first + 1 if first is not None else 0
            self._dense[segment] = (first or 0, np.zeros(blocks << LENGTHS_SHIFT, np.int32), np.zeros(blocks, bool))
        first, held, read = self._dense[segment]
        blocks = (ids >> LENGTHS_SHIFT) - first
        for block in np.unique(blocks[~read[blocks]]).tolist():
            row = self.db.execute(
                "SELECT data FROM kw_lengths WHERE segment = ? AND block = ?", (segment, block + first)
            ).fetchone()
            held[block << LENGTHS_SHIFT : (block + 1) << LENGTHS_SHIFT] = decode_lengths(row[0])
            read[block] = True
        return held[ids - (first << LENGTHS_SHIFT)]

    def documents(self, ids):
        """The documents of these chunks, by id."""
        owner = dict(
            self.db.execute(
                "SELECT id, doc FROM chunks WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(ids.tolist()),)
            )
        )
        return np.fromiter((owner[i] for i in ids.tolist()), np.int64, len(ids))

    def alive(self, segment, ids):
        """Which of these postings of the segment count: those of chunks not masked since it was built."""
        if not len(self.masked):
            return np.ones(len(ids), bool)
        at = np.minimum(np.searchsorted(self.masked, ids), len(self.masked) - 1)
        return ~((self.masked[at] == ids) & (self.masked_in[at] > segment))
