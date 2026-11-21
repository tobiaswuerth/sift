"""A query as text: its words as the phrases FTS5 takes, any of them, and those that mean something; and what the
reranker reads of a hit beside them, and how its scores blend with the cosines. None of it reads a store."""

import json
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .filters import each
from .hits import Hit

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it of on or that the this to "
    "was what when where which who why with".split()
)
QUOTED = r'"(?:[^"]|"")*"'
PHRASE = re.compile(rf"(?:{QUOTED}|[^\s()\"*]+)(?:\s*\*)?")  # a quoted phrase or a bare word, prefix or not
PLACED = re.compile(r"NEAR\s*\(|[{}+^:]")  # operators that join phrases or tie them to columns or places
COSINE_WEIGHT = 0.5  # how much the cosine counts in `query`, in units of the reranker's spread: see docs/benchmarks.md


def as_phrases(query: str) -> str:
    """Fallback for input that isn't valid FTS5 syntax: every word becomes an exact phrase, all required."""
    return " ".join('"' + w.replace('"', '""') + '"' for w in query.split())


def any_term(query: str) -> str | None:
    """The terms of a keyword query joined by OR, for when no chunk holds them all. None for a single term, and
    for queries with operators, which say what they want."""
    found: list[str] = re.findall(r'"[^"]*"|[^\s"]+', query)
    terms = [t for t in found if re.search(r"\w", t)]
    if len(terms) < 2 or any(t in ("AND", "OR", "NOT") or t.startswith("NEAR(") for t in terms):
        return None
    if re.search(r"(?:^|\s)-?(?:title|text|meta|\{[^}]*\})\s*:", query):  # a column filter: "title: ...", "{a b}: ..."
        return None
    return " OR ".join(t if t[0] == '"' else f'"{t[:-1]}" *' if t[-1] == "*" else f'"{t}"' for t in terms)


def phrases_of(match: str) -> list[str] | None:
    """The phrases of an FTS5 query that is phrases and AND, OR, NOT and parentheses alone, each as a query of its own;
    None for another, such as one with NEAR or a column filter."""
    if PLACED.search(re.sub(QUOTED, " ", match)):
        return None
    found: list[str] = PHRASE.findall(match)
    return [phrase for phrase in found if phrase not in ("AND", "OR", "NOT")]


def words(text: object) -> list[str]:
    """The meaningful words of a text: lowercase, without stopwords, single Latin letters ("a", the "s" of "it's") and
    underscores alone, where FTS5 parts words, in order. A digit stays, and a word of one character in another script,
    as Chinese and Japanese write many."""
    found: list[str] = re.findall(r"\w+", str(text).lower())
    return [
        w for w in found if w not in STOPWORDS and w.strip("_") and not (len(w) == 1 and w.isascii() and w.isalpha())
    ]


def any_word(query: str) -> str:
    """OR query over the meaningful words, used to collect keyword candidates for hybrid search."""
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words(query)))


def named(hit: Hit, asked: set[str]) -> list[str]:
    """The front-matter values of a hit, but its title, that share a word with the query `asked` (a set of words):
    an author's name, an id."""
    meta: dict[str, Any] = json.loads(hit["meta"])
    return [
        str(value)
        for key, field in meta.items()
        if key != "title"
        for value in each(field)
        if asked & set(words(value))
    ]


def passage(hit: Hit, named: list[str]) -> str:
    """What the reranker reads of a hit: its title, the front-matter values the query names, and its text. Only those:
    all of them (links, licences, long author lists) diluted the passage, and topical queries ranked worse."""
    return f"{hit['title']}\n{' '.join(named)}\n\n{hit['text']}" if named else f"{hit['title']}\n\n{hit['text']}"


def blend(
    rerank: NDArray[np.float64], cosine: NDArray[np.float64], named: NDArray[np.bool_], weight: float = COSINE_WEIGHT
) -> NDArray[np.float64]:
    """`query`'s scores: the reranker's, plus the candidates' cosine with the query, centred and in units of the
    spread of the reranker's scores, so one weight fits any reranker's scale and the scores keep it. A reranker
    trained on web search ranked scientific claims worse than the embedding did. Candidates whose front matter the
    query names keep the best cosine: the embedding never sees front matter."""
    lifted = cosine.copy()
    lifted[named] = cosine.max()
    spread = float(rerank.std() / lifted.std()) if np.ptp(lifted) > 0 else 0.0  # ptp: exactly 0 when all agree
    return rerank + weight * spread * (lifted - lifted.mean())
