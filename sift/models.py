"""What embeds and reranks: the models of this machine (`local.Local`), or a `sift worker` on another one
(`remote.Remote`). Both offer the `Models` interface, and `default()` picks one per process, so the rest of sift never
asks which it has.

This module loads neither PyTorch nor a model: a machine searching through a worker never needs them.
"""

from functools import cache
from typing import Protocol

from . import remote, settings


class Models(Protocol):
    """An embedding model and a reranker, wherever they run."""

    name: str  # the embedding model, as the index records it
    address: str  # where the models compute, to show: a device or a worker's address
    batch: int  # texts per encode() call: small enough to fit, large enough to be quick

    def load(self) -> None:
        """Get ready now, so the first request is quick."""

    def prepare(self, longest) -> None:
        """Size batches for the longest text to come, longest(), if that matters where the models run."""

    def encode(self, texts, query=False):
        """Normalised fp16 embeddings, one row per text."""

    def rerank(self, query, texts):
        """Scores of (query, text) pairs, higher for better matches."""

    def vectors(self, ids, matrix):
        """The index's vectors, ready to be scored against queries where these models run (see `vectors`)."""

    def release_idle(self) -> bool:
        """Hand the GPU back after a while without use. Returns whether anything moved."""


def default():
    """The models this process uses: the worker set in SIFT_EMBED_REMOTE, else those of this machine."""
    url = settings.current().remote
    return _remote(url) if url else _local()


@cache
def _remote(url):
    return remote.Remote(url)  # one per worker, so what it said about itself is asked once


@cache
def _local():
    from .local import Local  # PyTorch, only when this machine computes

    return Local()
