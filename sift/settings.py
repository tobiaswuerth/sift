"""sift's settings: five optional environment variables, read once per process. None are needed; they are for
special cases, and every module asks `current()` instead of reading the environment itself."""

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path  # SIFT_HOME: where the index lives
    embed_model: str  # SIFT_EMBED_MODEL: any Sentence Transformers embedding model
    rerank_model: str  # SIFT_RERANK_MODEL: any Sentence Transformers cross-encoder
    remote: str | None  # SIFT_EMBED_REMOTE: a `sift worker` to embed and rerank on, as it prints its address
    ocr: str = "auto"  # SIFT_OCR: where scans are read: auto (on a GPU), on (also on the CPU), off (see `ocr`)

    @classmethod
    def from_env(cls, environ=os.environ):
        """The settings an environment gives. An empty variable counts as unset."""
        return cls(
            home=Path(environ.get("SIFT_HOME") or Path.home() / ".sift"),
            embed_model=environ.get("SIFT_EMBED_MODEL") or "google/embeddinggemma-2",
            rerank_model=environ.get("SIFT_RERANK_MODEL") or "Alibaba-NLP/gte-reranker-modernbert-base",
            remote=environ.get("SIFT_EMBED_REMOTE") or None,
            ocr=(environ.get("SIFT_OCR") or "auto").strip().lower(),
        )


@cache
def current():
    """This process's settings, from its environment, read on first use."""
    return Settings.from_env()
