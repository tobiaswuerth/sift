"""sift's settings: the optional environment variables in the README's table, read once per process. None are
needed; they are for special cases, and every module asks `current()` instead of reading the environment itself.
Beside them CUDA's CUDA_VISIBLE_DEVICES, Hugging Face's HF_HUB_OFFLINE (offline, sift asks PyPI for no update
either), and where Numba keeps the code it compiles (`keep_compiled_code`), which it reads from the environment."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class Settings:
    home: Path  # SIFT_HOME: where the catalog, logs and worker's key live, and the collections' stores unless moved
    embed_model: str  # SIFT_EMBED_MODEL: any Sentence Transformers embedding model
    rerank_model: str  # SIFT_RERANK_MODEL: any Sentence Transformers cross-encoder
    remote: str | None  # SIFT_EMBED_REMOTE: a `sift worker` to embed and rerank on, as it prints its address
    ocr: str = "auto"  # SIFT_OCR: where scans are read: auto (on a GPU), on (also on the CPU), off (see `ocr`)
    gpus: str | None = None  # CUDA_VISIBLE_DEVICES, CUDA's own, as CUDA reads it: "" or -1 hides every GPU
    update_check: bool = True  # SIFT_UPDATE_CHECK: off never asks PyPI for a newer sift (see `update`); nor offline

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Self:
        """The settings an environment gives. An empty variable counts as unset."""
        return cls(
            home=Path(environ.get("SIFT_HOME") or Path.home() / ".sift"),
            embed_model=environ.get("SIFT_EMBED_MODEL") or "google/embeddinggemma-2",
            rerank_model=environ.get("SIFT_RERANK_MODEL") or "Alibaba-NLP/gte-reranker-modernbert-base",
            remote=environ.get("SIFT_EMBED_REMOTE") or None,
            ocr=(environ.get("SIFT_OCR") or "auto").strip().lower(),
            gpus=environ.get("CUDA_VISIBLE_DEVICES"),
            update_check=_on(environ.get("SIFT_UPDATE_CHECK") or "on")
            and not _on(environ.get("HF_HUB_OFFLINE") or "0"),
        )


def _on(value: str) -> bool:
    """Whether a switch is on: 1, on, true or yes, in any case."""
    return value.strip().lower() in ("1", "on", "true", "yes")


@cache
def current() -> Settings:
    """This process's settings, from its environment, read on first use."""
    return Settings.from_env()


def keep_compiled_code() -> str:
    """Numba keeps the code it compiles beside the module, where an installation may not write: in SIFT_HOME instead,
    unless NUMBA_CACHE_DIR says where. Before Numba is imported, which reads it then. Returns the folder."""
    return os.environ.setdefault("NUMBA_CACHE_DIR", str(current().home / "numba"))
