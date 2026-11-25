import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sift.domain.codec.codes import Floats, Halves
from sift.inference.prepared import ModelsPrepared

BROKEN = "unembeddable"
DIRECTIONLESS = "directionless"
LENGTH_WEIGHT = 0.5


def _word(word: str, width: int) -> Floats:
    seed = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(seed).normal(size=width).astype(np.float32)


def _vector(text: str, width: int) -> Floats:
    words = text.lower().split()
    if BROKEN in words:
        return np.full(width, np.nan, np.float32)
    if DIRECTIONLESS in words:
        return np.zeros(width, np.float32)
    summed = np.sum([_word(word, width) for word in words], axis=0) if words else np.ones(width, np.float32)
    return (summed / np.linalg.norm(summed)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class Hashed:
    name: str = "hashed-words"
    width: int = 48
    asked: list[tuple[tuple[str, ...], bool]] = field(default_factory=list[tuple[tuple[str, ...], bool]])
    prepared: list[bool] = field(default_factory=list[bool])
    ranker_prepared: list[bool] = field(default_factory=list[bool])
    reranked: list[tuple[str, tuple[str, ...]]] = field(default_factory=list[tuple[str, tuple[str, ...]]])
    scored: bool = False

    def prepare(self) -> ModelsPrepared:
        built = not self.prepared
        self.prepared.append(True)
        return ModelsPrepared((), built=built)

    def prepare_ranker(self) -> ModelsPrepared:
        built = not self.ranker_prepared
        self.ranker_prepared.append(True)
        return ModelsPrepared((), built=built)

    def release_idle(self) -> bool:
        return False

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        self.asked.append((tuple(texts), query))
        return (
            np.array([_vector(text, self.width) for text in texts], np.float32)
            .reshape(-1, self.width)
            .astype(np.float16)
        )

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        self.reranked.append((query, tuple(texts)))
        asked = set(query.lower().split())
        return np.array([self._score(asked, text.lower().split()) for text in texts], np.float32)

    def _score(self, asked: set[str], words: list[str]) -> float:
        shared = len(asked & set(words))
        return shared + LENGTH_WEIGHT / (1 + len(words)) if self.scored else shared
