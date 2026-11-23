import math
import unicodedata
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sift.domain.text.tokens import tokens

STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    )
)
COSINE_WEIGHT = 0.5
ADAPTIVE_SHARE = 0.9
ADAPTIVE_BUFFER = 5
FEWEST_TO_CUT = 3
DOI_LINKS = tuple(f"{scheme}{host}doi.org/" for scheme in ("https://", "http://") for host in ("", "www.", "dx."))
DOI_PREFIXES = (*DOI_LINKS, "doi:")

type Scores = NDArray[np.float64]
type Flags = NDArray[np.bool_]


def _latin_letter(character: str) -> bool:
    base = unicodedata.normalize("NFKD", character)[0]
    return base.isascii() and base.isalpha()


def words(text: str) -> list[str]:
    return [w for w in tokens(text) if w not in STOPWORDS and not (len(w) == 1 and _latin_letter(w))]


def named(values: Sequence[str], asked: frozenset[str]) -> list[str]:
    return [value for value in values if asked & set(words(value))]


def passage(title: str, names: Sequence[str], text: str) -> str:
    return f"{title}\n{' '.join(names)}\n\n{text}" if names else f"{title}\n\n{text}"


def blend(rerank: Scores, cosine: Scores, names: Flags, weight: float = COSINE_WEIGHT) -> Scores:
    if not len(rerank):
        return rerank
    known = np.isfinite(cosine)
    lifted = np.where(known, cosine, cosine[known].mean() if known.any() else 0.0)
    lifted[names] = lifted.max()
    spread = float(rerank.std() / lifted.std()) if np.ptp(lifted) > 0 else 0.0
    return rerank + weight * spread * (lifted - lifted.mean())


def standing_out(scores: Sequence[float]) -> int:
    if len(scores) < FEWEST_TO_CUT:
        return len(scores)
    held = np.array(scores, np.float64)
    gaps = held[:-1] - held[1:]
    considered = gaps[: math.ceil(ADAPTIVE_SHARE * len(gaps))]
    if not considered.max() > 0:
        return len(scores)
    return min(len(scores), int(considered.argmax()) + 1 + ADAPTIVE_BUFFER)


def work_of(doi: object) -> str | None:
    if not isinstance(doi, str):
        return None
    value = doi.strip().lower()
    for prefix in DOI_PREFIXES:
        value = value.removeprefix(prefix).strip()
    return value if value.startswith("10.") else None
