"""The models of this machine: an embedding model and a reranker, run with PyTorch on the accelerator's device."""

import logging

import numpy as np

from . import settings
from .accelerator import Accelerator
from .vectors import InRam, OnGpu

MAX_TOKENS = 1024

# The MCP server logs at INFO, where loading a model logs every HTTP request to Hugging Face.
for _name in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _quietly():
    """No "Loading weights" bar per model: they filled the log of `sift serve`, garbled on Windows consoles."""
    from transformers.utils import logging

    logging.disable_progress_bar()


def _embedder(name, dtype):
    from sentence_transformers import SentenceTransformer

    _quietly()
    model = SentenceTransformer(name, device="cpu")  # into RAM; it moves to the GPU when used
    if dtype:
        model.to(dtype)
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model


def _reranker(name, dtype):
    from sentence_transformers import CrossEncoder

    _quietly()
    model = CrossEncoder(name, device="cpu", max_length=MAX_TOKENS)  # into RAM, like the embedder
    if dtype:
        model.to(dtype)  # the whole model: its scoring head too, or PyTorch refuses to mix the two
    return model


class Local:
    """The models of this machine. Each loads on first use, into RAM in the accelerator's half precision, and moves
    to the GPU while in use. If the accelerator falls back to the CPU, they load again, in float32."""

    def __init__(self, accelerator=None):
        self.accelerator = accelerator or Accelerator()
        self.name, self.reranker_name = settings.current().embed_model, settings.current().rerank_model
        self._loaded, self._dtype = {}, None  # model by kind, all in this precision

    @property
    def address(self):
        """Where the models compute, to show: the device."""
        return self.accelerator.device

    @property
    def batch(self):
        """Texts per encode() call: the accelerator's batch size."""
        return self.accelerator.batch_size

    def embedder(self):
        return self._model(_embedder, self.name)

    def reranker(self):
        return self._model(_reranker, self.reranker_name)

    def _model(self, load, name):
        self.accelerator.budget()  # caps the GPU memory, or falls back to the CPU, before anything loads
        dtype = self.accelerator.half_precision
        if dtype != self._dtype:  # a device of another precision: load again
            self._loaded, self._dtype = {}, dtype
        if name not in self._loaded:
            self._loaded[name] = load(name, dtype)
        return self._loaded[name]

    def _ready(self, get):
        """The model get() returns, on the GPU while in use. The accelerator looks at the GPU first: it may have
        fallen back to the CPU, and then get() returns a model for the CPU."""
        self.accelerator.resume()
        model = get()
        self.accelerator.acquire(id(model), lambda: model.to(self.accelerator.device), lambda: model.to("cpu"))
        return model

    def load(self):
        """Load both models now, so the first request is quick."""
        self.embedder()
        self.reranker()

    def prepare(self, longest):
        """Size the batches to the GPU's memory, with the longest text to come, longest(), or None for the longest
        input the model takes."""
        if self.accelerator.kind != "cuda":
            return
        model = self._ready(self.embedder)
        sample = longest() or "x " * (4 * model.max_seq_length)  # truncated to the longest input
        self.accelerator.calibrate(lambda n: model.encode_document([sample] * n, batch_size=n, show_progress_bar=False))

    def encode(self, texts, query=False):
        """Normalised fp16 embeddings, one row per text. Sentence Transformers sorts by length to minimise padding."""
        model = self._ready(self.embedder)
        fn = model.encode_query if query else model.encode_document
        return self.accelerator.run(
            lambda t, size: fn(
                t, batch_size=size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            ),
            texts,
        ).astype(np.float16)

    def rerank(self, query, texts):
        """Raw scores (logits) of (query, text) pairs, higher for better matches, whatever activation the model's
        configuration names, so that scores mean the same with any reranker."""
        import torch

        model, raw = self._ready(self.reranker), torch.nn.Identity()
        return self.accelerator.run(
            lambda t, size: model.predict(
                [(query, x) for x in t], batch_size=size, show_progress_bar=False, activation_fn=raw
            ),
            texts,
        )

    def vectors(self, ids, matrix):
        """The vectors to search, where these models compute: on the GPU, or in RAM."""
        return OnGpu(ids, matrix, self.accelerator) if self.accelerator.gpu else InRam(ids, matrix)

    def release_idle(self):
        """Move models and vectors back to RAM if the GPU has been idle long enough. Returns whether it did."""
        return self.accelerator.release_idle()
