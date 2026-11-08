"""The best documents (or passages) for a keyword query, from the postings (see `postings`): BM25 as FTS5's bm25
scores it, summed over the query's terms, with the statistics of all the stores searched together.

Block-Max Pruning (Mallia et al. 2024): chunks' places (see `postings`) fall into ranges (postings.RANGE_SHIFT bits);
each term's best score in a range, weighted by its idf and summed over the query's terms, bounds the score of every
chunk there. Ranges are scored best bound first, a batch at a time, until no range left can beat the k-th best found:
what remains is never read. A query is an expression: ("term", text), ("phrase", tokens), ("and", [...]), ("or",
[...]) or ("not", left, right), as FTS5 reads AND, OR and NOT. It holds per chunk by the terms and phrases (its leaves)
the chunk holds; a chunk's score sums the leaves that count in it: a leaf written twice counts twice, and one under an
AND or NOT that fails in the chunk counts nothing, even where the chunk is found otherwise ("x OR (a AND b)" in a chunk
with x and a: x alone). That is FTS5's bm25, score for score, but where FTS5 counts by where its readers happen to
stand in their lists, in nested queries alone: it may count the right side of a NOT that fails ("x OR (a NOT b)" in a
chunk with x and b), and miss a phrase in an OR that holds. A phrase is bounded by its tokens, which the postings hold,
and counted with FTS5, which has their positions, in the chunks the bounds leave."""

import math
import time

import numpy as np

from . import postings

FIRST_BATCH = 16  # ranges scored first, doubling each batch: the threshold rises before much is read
MOST_BATCH = 4096
LEAVES = ("term", "phrase")


def idf(chunks, df):
    """FTS5's idf, which never falls to 0 or below, for a term in `df` of `chunks` chunks."""
    value = math.log((chunks - df + 0.5) / (df + 0.5))
    return value if value > 0 else 1e-6


def terms_of(node):
    """The leaves of an expression, terms (text) and phrases (tuples of tokens), each as often as it occurs: FTS5
    scores each occurrence."""
    kind = node[0]
    if kind in LEAVES:
        return [node[1]]
    if kind == "not":
        return terms_of(node[1]) + terms_of(node[2])
    return [term for child in node[1] for term in terms_of(child)]


def tokens_of(leaf):
    """The tokens of a leaf: a term's one, a phrase's several."""
    return (leaf,) if isinstance(leaf, str) else leaf


def counted(node, present):
    """Where an expression holds, given where each of its leaves is present (boolean arrays by leaf), and where each of
    its leaves counts, where it is present and no AND or NOT above it fails: [(leaf, where)], a leaf as often as it
    occurs."""
    kind = node[0]
    if kind in LEAVES:
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
    """Where an expression may hold, given where each leaf may be present: a range holding a NOT's leaves may still
    have chunks without them."""
    kind = node[0]
    if kind in LEAVES:
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


class Deadline:
    """When searches are to stop, `seconds` from now, and whether one stopped before it was done (`cut`): a search out
    of time returns the best it found so far, every store's best ranges scored, once it found any (nothing found is
    always the answer of the whole search)."""

    def __init__(self, seconds):
        self.at, self.cut = time.perf_counter() + seconds, False

    def over(self):
        return time.perf_counter() >= self.at  # monotonic() ticks every 16 ms on Windows


def search(readers, query, k, by_document=True, allowed=None, deadline=None):
    """The best k documents for an expression (see the module), each by its best chunk, or the best k chunks:
    [(chunk id, score)], best first. `readers` are the stores' postings (postings.Reader); allowed(chunk ids), if
    given, says which of those chunks may be found, as a boolean array (a filter); a Deadline, when to stop."""
    chunks = sum(r.chunks for r in readers)
    best = Best(k, by_document)
    if not chunks:
        return []
    average = sum(r.tokens for r in readers) / chunks
    named = terms_of(query)
    leaves = list(dict.fromkeys(named))  # in the query's order, not a set's: scores summed alike in every process
    tokens = list(dict.fromkeys(t for leaf in leaves for t in tokens_of(leaf)))
    lists = [{t: r.term(t) for t in tokens} for r in readers]
    df = {
        leaf: sum(p.df for held in lists for p in held[leaf])
        if isinstance(leaf, str)
        else sum(r.phrase_df(leaf) for r in readers)
        for leaf in leaves
    }
    weight = {leaf: idf(chunks, df[leaf]) for leaf in leaves}
    for reader, held in zip(readers, lists):
        _Store(reader, held, weight, named, average, query, best, allowed, deadline).search()
    return best.ranked()


class _Store:
    """The search in one store: its ranges bounded, then scored best first."""

    def __init__(self, reader, lists, weight, named, average, query, best, allowed, deadline):
        self.reader, self.weight, self.average, self.deadline = reader, weight, average, deadline
        self.bounding = {t: w * named.count(t) for t, w in weight.items()}  # a leaf written twice counts twice
        self.most = {t: b * postings.TOP for t, b in self.bounding.items()}  # the most a leaf adds to a chunk's score
        leaves_only = query[0] in ("and", "or") and all(child[0] in LEAVES for child in query[1])
        self.flat = leaves_only and query[0] == "or"  # each leaf counts where it is
        self.conjunction = leaves_only and query[0] == "and"  # a chunk counts if it holds them all
        self.query, self.best, self.allowed = query, best, allowed
        self.shift = postings.RANGE_SHIFT
        self.counts = {
            t: sum(p.df for p in parts) for t, parts in lists.items()
        }  # each token's postings: the rarest first
        # a short list (every delta's) read whole, its term scores exact; a long list's ranges, with how far their
        # bounds may fall short: computed with its segment's average length, and a term score grows with it
        self.whole = {t: [self._whole(p) for p in parts if not p.long] for t, parts in lists.items()}
        self.ranged = {t: [(p, p.ranges(), self._slack(p)) for p in parts if p.long] for t, parts in lists.items()}

    def _whole(self, part):
        places, tf = part.everything()
        alive = self.reader.alive(part.segment, places)
        places, tf = places[alive], tf[alive]
        scores = postings.term_scores(tf, self.reader.lengths(part.segment, places), self.average)
        return places >> self.shift, places, scores

    def _slack(self, part):
        chunks, tokens = self.reader.segments[part.segment]
        return max(1.0, self.average * chunks / tokens) if tokens else 1.0

    def search(self):
        first, bound, present = self._bounds()
        candidates = np.flatnonzero(may_hold(self.query, present) & (bound > 0))  # bound: all leaves count, at most
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
            if self.deadline is not None and self.best.found and self.deadline.over():
                self.deadline.cut |= bool((bound >= self.best.threshold).any())  # a range left might beat one found
                return

    def _bounds(self):
        """(the first range id, then per range from it: the leaves' bounds summed, weighted; where each leaf may be). A
        phrase is in a chunk at most as often as each of its tokens: bounded by the least of theirs."""
        found = [rid for t in self.whole for rid, *_ in self.whole[t]]
        found += [ranges[0] for t in self.ranged for _, ranges, _ in self.ranged[t]]
        found = [f for f in found if len(f)]
        if not found:
            return 0, np.zeros(0), {t: np.zeros(0, bool) for t in self.weight}
        first = min(int(f[0]) for f in found)
        span = max(int(f[-1]) for f in found) - first + 1
        heres = {}
        for t in self.whole:
            here = np.zeros(span)
            for rid, _, scores in self.whole[t]:
                np.maximum.at(here, rid - first, scores)
            for _, (rid, _, _, best), slack in self.ranged[t]:
                at = rid - first
                here[at] = np.maximum(here[at], best * slack)
            heres[t] = here
        bound, present = np.zeros(span), {}
        for leaf in self.weight:
            here = np.minimum.reduce([heres[t] for t in tokens_of(leaf)])
            bound += self.bounding[leaf] * here
            present[leaf] = here > 0
        return first, bound, present

    def _lazy(self):
        """The leaves that can't bring a chunk to the k-th best score, all of them together (MaxScore, Turtle and Flood
        1995): in an OR of leaves, a chunk must hold another to be found, so these are read only for the chunks the
        others bring within reach of it. The words of a question that are in every chunk, "of", "in", "the", were read
        and scored in every range scored, and made a question ten times as slow."""
        lazy, total = [], 0.0
        for leaf in sorted(self.most, key=self.most.get) if self.flat else ():
            if total + self.most[leaf] >= self.best.threshold:
                break
            lazy.append(leaf)
            total += self.most[leaf]
        return lazy

    def _score(self, group):
        """Score every chunk of these ranges (ascending) that holds any leaf, and offer those the query finds to the
        best."""
        if self.conjunction:
            places, acc = self._all_of(group)
            self._offer(self.reader.chunks_at(places), acc)
            return
        lazy = self._lazy()
        eager = [leaf for leaf in self.weight if leaf not in lazy]
        found = {t: list(self._in(t, group)) for leaf in eager for t in tokens_of(leaf)}  # the group holds some
        every = [places for parts in found.values() for places, _ in parts]
        if not any(len(places) for places in every):  # those of lazy leaves alone: none can reach the k-th best
            return
        places, where = np.unique(np.concatenate(every), return_inverse=True)  # each posting's chunk among them
        held, at = {}, 0
        for t, parts in found.items():
            held[t] = np.zeros(len(places))
            for postings_here, term_scores in parts:
                held[t][where[at : at + len(postings_here)]] = term_scores
                at += len(postings_here)
        if self.flat:
            places, acc = self._any_of(places, held, eager, lazy)
            chunks = self.reader.chunks_at(places)
        else:
            scores = {leaf: self._exact(leaf, places, held) for leaf in eager}
            holds, terms = counted(self.query, {leaf: s > 0 for leaf, s in scores.items()})
            acc = np.zeros(len(places))
            for t, counts in terms:
                acc += np.where(counts, self.weight[t] * scores[t], 0.0)
            keep = np.flatnonzero(holds & (acc >= self.best.threshold))
            chunks, acc = self.reader.chunks_at(places[keep]), acc[keep]
        self._offer(chunks, acc)

    def _bound(self, leaf, places, held):
        """A leaf's score at these places, unweighted, from its tokens' term scores there (`held`), or the most it can
        be: a term's own; a phrase's at most its rarest token's, where FTS5 finds it, 0 elsewhere."""
        if isinstance(leaf, str):
            return held[leaf]
        return np.where(self._holds(leaf, places), np.minimum.reduce([held[t] for t in leaf]), 0.0)

    def _holds(self, phrase, places):
        """Which of these places' chunks hold the phrase, as FTS5 found them (see postings.Reader.phrase_chunks)."""
        holding, chunks = self.reader.phrase_chunks(phrase), self.reader.chunks_at(places)
        at = np.minimum(np.searchsorted(holding, chunks), max(len(holding) - 1, 0))
        return holding[at] == chunks if len(holding) else np.zeros(len(places), bool)

    def _exact(self, leaf, places, held):
        """A leaf's score at these places, unweighted (see _bound), a phrase's counted where it is."""
        bound = self._bound(leaf, places, held)
        if isinstance(leaf, str):
            return bound
        out, there = np.zeros(len(places)), np.flatnonzero(bound > 0)
        out[there] = self._phrase_scores(leaf, places[there])
        return out

    def _phrase_scores(self, phrase, places):
        """The phrase's term score at these places, counted there (0 where it isn't)."""
        tf, lengths = self.reader.phrase_counts(phrase, self.reader.chunks_at(places))
        return postings.term_scores(tf, np.maximum(lengths, 1), self.average)

    def _offer(self, chunks, acc):
        """Offer these chunks, with their scores, to the best: those the filter allows, and not deleted meanwhile."""
        if self.allowed is not None and len(chunks):
            allowed = np.asarray(self.allowed(chunks), bool)
            chunks, acc = chunks[allowed], acc[allowed]
        if len(chunks):
            docs = self.reader.documents(chunks)
            here = docs >= 0  # not deleted meanwhile
            self.best.offer(chunks[here], docs[here], acc[here])

    def _all_of(self, group):
        """The places, and scores, of the chunks of these ranges that hold every leaf and reach the k-th best score:
        the rarest token read first, and each next one only in the ranges where chunks holding all before it are left;
        phrases counted last, for the chunks that hold them and could still reach it. Read all at once, the words of a
        question that no chunk holds together, "the" and "of" too, were read and scored in every range they all occur
        in: nothing found, nothing pruned, 1.2 s over 3.2 million abstracts."""
        places, held = np.zeros(0, np.int64), {}
        for n, token in enumerate(sorted(self.counts, key=self.counts.get)):
            ranges = group if n == 0 else np.unique(places >> self.shift)
            here, term_scores = self._merged(token, ranges)
            if n == 0:
                places, held = here, {token: term_scores}
                continue
            at = np.minimum(np.searchsorted(here, places), max(len(here) - 1, 0))
            hit = np.flatnonzero(here[at] == places) if len(here) else at[:0]
            places, held = places[hit], {t: s[hit] for t, s in held.items()} | {token: term_scores[at[hit]]}
        bounds = {leaf: self._bound(leaf, places, held) for leaf in self.weight}
        there = np.flatnonzero(np.logical_and.reduce([b > 0 for b in bounds.values()]))  # every phrase too
        places, bounds = places[there], {leaf: b[there] for leaf, b in bounds.items()}
        return self._counted_in(places, bounds, [leaf for leaf in self.weight if not isinstance(leaf, str)], [])

    def _any_of(self, places, held, eager, lazy):
        """The places, and scores, of the chunks that reach the k-th best score in an OR of leaves: the eager ones
        bounded (see _bound), the lazy ones (see _lazy) left to the chunks the others bring within reach of it."""
        bounds = {leaf: self._bound(leaf, places, held) for leaf in eager}
        return self._counted_in(places, bounds, [leaf for leaf in eager if not isinstance(leaf, str)], lazy)

    def _counted_in(self, places, bounds, phrases, lazy):
        """The places, and scores, of the chunks that reach the k-th best score, from the leaves' scores there or the
        most they can be (`bounds`): the phrases among them counted, then the lazy leaves read, each only for the
        chunks still within reach of it, the largest first: "the", last, for the fewest."""
        acc = sum((self.bounding[leaf] * b for leaf, b in bounds.items()), np.zeros(len(places)))
        lazy = sorted(lazy, key=self.most.get, reverse=True)
        rest = sum(self.most[leaf] for leaf in lazy)
        for phrase in sorted(phrases, key=self.most.get, reverse=True):
            near = np.flatnonzero(acc + rest >= self.best.threshold)
            places, acc, bounds = places[near], acc[near], {leaf: b[near] for leaf, b in bounds.items()}
            there = np.flatnonzero(bounds[phrase] > 0)
            counted_here = self._phrase_scores(phrase, places[there]) if len(there) else np.zeros(0)
            acc[there] += self.bounding[phrase] * (counted_here - bounds[phrase][there])
        for n, leaf in enumerate(lazy):
            near = np.flatnonzero(acc + sum(self.most[u] for u in lazy[n:]) >= self.best.threshold)
            places, acc = places[near], acc[near]
            if not isinstance(leaf, str):
                there = np.flatnonzero(self._holds(leaf, places))
                acc[there] += self.bounding[leaf] * self._phrase_scores(leaf, places[there]) if len(there) else 0.0
                continue
            ranges = np.unique(places >> self.shift)
            for held, term_scores in self._in(leaf, ranges) if len(ranges) else ():
                at = np.minimum(np.searchsorted(held, places), len(held) - 1)
                hit = np.flatnonzero(held[at] == places) if len(held) else at[:0]
                acc[hit] += self.bounding[leaf] * term_scores[at[hit]]
        keep = np.flatnonzero((acc >= self.best.threshold) & (acc > 0))  # 0: a phrase's tokens alone, no leaf
        return places[keep], acc[keep]

    def _merged(self, term, ranges):
        """(places, term scores) of a term's live postings in these ranges, of every segment, ascending."""
        parts = list(self._in(term, ranges)) if len(ranges) else []
        if len(parts) == 1:
            return parts[0]
        held = np.concatenate([np.zeros(0, np.int64), *(p for p, _ in parts)])
        term_scores = np.concatenate([np.zeros(0), *(s for _, s in parts)])
        order = np.argsort(held, kind="stable")
        return held[order], term_scores[order]

    def _in(self, term, group):
        """(places, term scores) of a term's live postings in these ranges, a part per segment."""
        for rid, places, scores in self.whole[term]:
            take = postings.slices(np.searchsorted(rid, group), np.searchsorted(rid, group, side="right"))
            yield places[take], scores[take]
        for part, (rid, starts, ends, _), _ in self.ranged[term]:
            at = np.minimum(np.searchsorted(rid, group), len(rid) - 1)
            hit = rid[at] == group
            if not hit.any():
                continue
            places, tf = part.postings(starts[at[hit]], ends[at[hit]])
            alive = self.reader.alive(part.segment, places)
            places, tf = places[alive], tf[alive]
            yield places, postings.term_scores(tf, self.reader.lengths(part.segment, places), self.average)
