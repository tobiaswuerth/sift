"""The best documents (or passages) for a keyword query, from the postings (see `postings`): BM25 as FTS5's bm25
scores it, summed over the query's terms, with the statistics of all the stores searched together.

Block-Max Pruning (Mallia et al. 2024): chunk ids fall into ranges (postings.RANGE_SHIFT bits); each term's best score
in a range, weighted by its idf and summed over the query's terms, bounds the score of every chunk there. Ranges are
scored best bound first, a batch at a time, until no range left can beat the k-th best found: what remains is never
read. A query is an expression: ("term", text), ("and", [...]), ("or", [...]) or ("not", left, right), as FTS5 reads
AND, OR and NOT. It holds per chunk by the terms the chunk holds; a chunk's score sums the terms that count in it, as
FTS5 counts them: a term written twice counts twice, and one under an AND or NOT that fails in the chunk counts
nothing, even where the chunk is found otherwise ("x OR (a AND b)" in a chunk with x and a: x alone)."""

import math

import numpy as np

from . import postings

FIRST_BATCH = 16  # ranges scored first, doubling each batch: the threshold rises before much is read
MOST_BATCH = 4096


def idf(chunks, df):
    """FTS5's idf, which never falls to 0 or below, for a term in `df` of `chunks` chunks."""
    value = math.log((chunks - df + 0.5) / (df + 0.5))
    return value if value > 0 else 1e-6


def terms_of(node):
    """The terms of an expression, each as often as it occurs: FTS5 scores each occurrence."""
    kind = node[0]
    if kind == "term":
        return [node[1]]
    if kind == "not":
        return terms_of(node[1]) + terms_of(node[2])
    return [term for child in node[1] for term in terms_of(child)]


def counted(node, present):
    """Where an expression holds, given where each of its terms is present (boolean arrays by term), and where each of
    its terms counts, as FTS5 counts them after it checks a row (fts5ExprCheckPoslists): [(term, where)], a term as
    often as it occurs."""
    kind = node[0]
    if kind == "term":
        return present[node[1]], [(node[1], present[node[1]])]
    if kind == "not":
        held, terms = counted(node[1], present)
        held = held & ~counted(node[2], present)[0]
        return held, [(term, where & held) for term, where in terms]
    parts = [counted(child, present) for child in node[1]]
    terms = [term for _, part in parts for term in part]
    if kind == "or":
        return np.logical_or.reduce([held for held, _ in parts]), terms
    held = np.logical_and.reduce([held for held, _ in parts])
    return held, [(term, where & held) for term, where in terms]


def may_hold(node, present):
    """Where an expression may hold, given where each term may be present: a range holding a NOT's terms may still
    have chunks without them."""
    kind = node[0]
    if kind == "term":
        return present[node[1]]
    if kind == "not":
        return may_hold(node[1], present)
    parts = [may_hold(child, present) for child in node[1]]
    return np.logical_and.reduce(parts) if kind == "and" else np.logical_or.reduce(parts)


class Best:
    """The best k documents found so far, each by its best chunk, or the best k chunks; and the score to beat."""

    def __init__(self, k, by_document=True):
        self.k, self.by_document, self.found = k, by_document, {}
        self.threshold = 0.0

    def offer(self, chunks, docs, scores):
        found = self.found
        for chunk, doc, score in zip(chunks.tolist(), docs.tolist(), scores.tolist()):
            key = doc if self.by_document else chunk
            held = found.get(key)
            if held is None or score > held[0] or (score == held[0] and chunk < held[1]):
                found[key] = (score, chunk)
        if len(found) >= self.k:
            scores = np.fromiter((s for s, _ in found.values()), np.float64, len(found))
            self.threshold = float(np.partition(scores, -self.k)[-self.k])

    def ranked(self):
        """[(chunk, score)], best first; ties by chunk id."""
        best = sorted(self.found.values(), key=lambda held: (-held[0], held[1]))[: self.k]
        return [(chunk, score) for score, chunk in best]


def search(readers, query, k, by_document=True, allowed=None):
    """The best k documents for an expression (see the module), each by its best chunk, or the best k chunks:
    [(chunk id, score)], best first. `readers` are the stores' postings (postings.Reader); allowed(chunk ids), if
    given, says which of those chunks may be found, as a boolean array (a filter)."""
    chunks = sum(r.chunks for r in readers)
    best = Best(k, by_document)
    if not chunks:
        return []
    average = sum(r.tokens for r in readers) / chunks
    named = terms_of(query)
    terms = list(dict.fromkeys(named))  # in the query's order, not a set's: scores summed alike in every process
    lists = [{t: r.term(t) for t in terms} for r in readers]
    df = {t: sum(p.df for held in lists for p in held[t]) for t in terms}
    weight = {t: idf(chunks, df[t]) for t in terms}
    for reader, held in zip(readers, lists):
        _Store(reader, held, weight, named, average, query, best, allowed).search()
    return best.ranked()


class _Store:
    """The search in one store: its ranges bounded, then scored best first."""

    def __init__(self, reader, lists, weight, named, average, query, best, allowed):
        self.reader, self.weight, self.average = reader, weight, average
        self.bounding = {t: w * named.count(t) for t, w in weight.items()}  # a term written twice counts twice
        self.query, self.best, self.allowed = query, best, allowed
        self.shift = postings.RANGE_SHIFT
        # a short list (every delta's) read whole, its term scores exact; a long list's ranges, with how far their
        # bounds may fall short: computed with its segment's average length, and a term score grows with it
        self.whole = {t: [self._whole(p) for p in parts if not p.long] for t, parts in lists.items()}
        self.ranged = {t: [(p, p.ranges(), self._slack(p)) for p in parts if p.long] for t, parts in lists.items()}

    def _whole(self, part):
        ids, tf = part.everything()
        alive = self.reader.alive(part.segment, ids)
        ids, tf = ids[alive], tf[alive]
        return ids >> self.shift, ids, postings.term_scores(tf, self.reader.lengths(part.segment, ids), self.average)

    def _slack(self, part):
        chunks, tokens = self.reader.segments[part.segment]
        return max(1.0, self.average * chunks / tokens) if tokens else 1.0

    def search(self):
        first, bound, present = self._bounds()
        candidates = np.flatnonzero(may_hold(self.query, present) & (bound > 0))  # bound: all terms count, at most
        remaining, bound = candidates + first, bound[candidates]
        batch = FIRST_BATCH
        while len(remaining):
            beat = bound >= self.best.threshold  # a tie with the k-th best may yet take its place: by lower chunk id
            remaining, bound = remaining[beat], bound[beat]
            if len(remaining) > batch:
                pick = np.argpartition(-bound, batch - 1)[:batch]
                rest = np.ones(len(remaining), bool)
                rest[pick] = False
                group, remaining, bound = remaining[pick], remaining[rest], bound[rest]
            else:
                group, remaining, bound = remaining, remaining[:0], bound[:0]
            if len(group):
                self._score(np.sort(group))
            batch = min(2 * batch, MOST_BATCH)

    def _bounds(self):
        """(the first range id, then per range from it: the terms' bounds summed, weighted; where each term occurs)."""
        found = [rid for t in self.whole for rid, *_ in self.whole[t]]
        found += [ranges[0] for t in self.ranged for _, ranges, _ in self.ranged[t]]
        found = [f for f in found if len(f)]
        if not found:
            return 0, np.zeros(0), {t: np.zeros(0, bool) for t in self.weight}
        first = min(int(f[0]) for f in found)
        span = max(int(f[-1]) for f in found) - first + 1
        bound = np.zeros(span)
        present = {}
        for t in self.whole:
            here = np.zeros(span)
            for rid, _, scores in self.whole[t]:
                np.maximum.at(here, rid - first, scores)
            for _, (rid, _, _, best), slack in self.ranged[t]:
                at = rid - first
                here[at] = np.maximum(here[at], best * slack)
            bound += self.bounding[t] * here
            present[t] = here > 0
        return first, bound, present

    def _score(self, group):
        """Score every chunk of these ranges (ascending) that holds any term, and offer those the query finds to the
        best."""
        found = {t: list(self._in(t, group)) for t in self.whole}  # a group's ranges hold some term's postings
        every = [ids for parts in found.values() for ids, _ in parts]
        chunks, where = np.unique(np.concatenate(every), return_inverse=True)  # each posting's chunk among them
        scores, present, at = {}, {}, 0
        for t, parts in found.items():
            scores[t] = np.zeros(len(chunks))
            for ids, term_scores in parts:
                scores[t][where[at : at + len(ids)]] = term_scores
                at += len(ids)
            present[t] = scores[t] > 0
        holds, terms = counted(self.query, present)
        acc = np.zeros(len(chunks))
        for t, counts in terms:
            acc += np.where(counts, self.weight[t] * scores[t], 0.0)
        keep = np.flatnonzero(holds & (acc >= self.best.threshold))
        chunks, acc = chunks[keep], acc[keep]
        if self.allowed is not None and len(chunks):
            allowed = np.asarray(self.allowed(chunks), bool)
            chunks, acc = chunks[allowed], acc[allowed]
        if len(chunks):
            self.best.offer(chunks, self.reader.documents(chunks), acc)

    def _in(self, term, group):
        """(chunk ids, term scores) of a term's live postings in these ranges, a part per segment."""
        for rid, ids, scores in self.whole[term]:
            take = postings.slices(np.searchsorted(rid, group), np.searchsorted(rid, group, side="right"))
            yield ids[take], scores[take]
        for part, (rid, starts, ends, _), _ in self.ranged[term]:
            at = np.minimum(np.searchsorted(rid, group), len(rid) - 1)
            hit = rid[at] == group
            if not hit.any():
                continue
            ids, tf = part.postings(starts[at[hit]], ends[at[hit]])
            alive = self.reader.alive(part.segment, ids)
            ids, tf = ids[alive], tf[alive]
            yield ids, postings.term_scores(tf, self.reader.lengths(part.segment, ids), self.average)
