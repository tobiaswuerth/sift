"""The models of this machine: an embedding model and a reranker, run with PyTorch on the accelerator's device.

Texts are embedded as Sentence Transformers' encode_document and encode_query embed them, through the same public
steps (the model's prompts, preprocess, its forward pass), split so that the next batch is tokenized while the GPU
embeds the one before (encode_batches)."""

import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import accelerator, settings
from .accelerator import UNSET, Accelerator, Unset

if TYPE_CHECKING:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from tokenizers import Tokenizer

MAX_TOKENS = 1024
RERANK_BATCH = 16  # pairs per batch at most, which Sentence Transformers sorts by length first: one batch of a query's
# 80 candidates was padded to the longest, 1.7 to 3.5 times their tokens; reranked in 16s, 180 ms a question on a
# GPU against 331 (8: 259, 32: 213)
LENGTHS_AT_ONCE = 1000  # texts tokenized at a time to count their tokens
PROMPTS = {False: ("document", "passage", "corpus"), True: ("query",)}  # as encode_document and encode_query pick
TASKS = {False: "document", True: "query"}  # what Sentence Transformers routes texts by, in models that route
_Model = TypeVar("_Model", bound="torch.nn.Module")


# What sift uses of Sentence Transformers' models and of a fast tokenizer, typed by sift: their own types name
# optional packages (audio, video) that are Unknown where those aren't installed. The type checker holds the real
# classes to these where they are handed over.
class _Embedder(Protocol):
    @property
    def max_seq_length(self) -> int | None: ...
    @property
    def device(self) -> "torch.device": ...
    @property
    def prompts(self) -> Mapping[str, str | None]: ...
    @property
    def default_prompt_name(self) -> str | None: ...
    def preprocess(self, inputs: Any, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _Reranker(Protocol):
    def predict(
        self, inputs: Any, *, batch_size: int = ..., show_progress_bar: bool | None = ..., activation_fn: Any = ...
    ) -> Any: ...


class _Truncating(Protocol):
    def enable_truncation(self, max_length: int, /, **kwargs: Any) -> None: ...


def _longest(model: _Embedder) -> int:
    """The most tokens the model reads of a text: as the loaders set it, MAX_TOKENS at most."""
    return model.max_seq_length or MAX_TOKENS


# The MCP server logs at INFO, where loading a model logs every HTTP request to Hugging Face: through httpx, or
# httpx2, which huggingface_hub 2 uses.
for _name in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _quietly() -> None:
    """No "Loading weights" bar per model: they filled the log of `sift serve`, garbled on Windows consoles."""
    from transformers.utils import logging

    logging.disable_progress_bar()


def _in(dtype: "torch.dtype | None") -> dict[str, "torch.dtype"]:
    """How to load a model in `dtype` (None: float32), not the checkpoint's: embeddinggemma's is bfloat16, slow on
    CPUs."""
    import torch

    return {"dtype": dtype or torch.float32}


def _embedder(name: str, dtype: "torch.dtype | None") -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    _quietly()
    model = SentenceTransformer(name, device="cpu", model_kwargs=_in(dtype))  # into RAM; to the GPU when used
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model.eval()  # as encode() sets it: in training mode, dropout would change every vector at random


def _reranker(name: str, dtype: "torch.dtype | None") -> "CrossEncoder":
    from sentence_transformers import CrossEncoder

    _quietly()
    model = CrossEncoder(name, device="cpu", model_kwargs=_in(dtype))  # into RAM, its scoring head in dtype too
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)  # MiniLM reads at most 512 tokens
    return model


class Local:
    """The models of this machine. Each loads on first use, into RAM in the accelerator's half precision, and moves
    to the GPU while in use. If the accelerator falls back to the CPU, they load again, in float32."""

    main_thread = True  # PyTorch

    def __init__(self, accelerator: Accelerator | None = None):
        self.accelerator = accelerator or Accelerator()
        self.name, self.reranker_name = settings.current().embed_model, settings.current().rerank_model
        self._loaded: dict[str, Any] = {}  # the models loaded, by name
        self._dtype: torch.dtype | None = None  # the precision all of them are in
        self._lock = threading.RLock()  # loading, which threads may ask for at once
        # a tokenizer of the embedding model's own, to count tokens with (see lengths)
        self._counter: Tokenizer | Unset | None = UNSET

    @property
    def address(self) -> str:
        """Where the models compute, to show: the device."""
        return self.accelerator.device

    @property
    def batch(self) -> int:
        """Texts per encode() call: the accelerator's batch size."""
        return self.accelerator.batch_size

    def embedder(self) -> "SentenceTransformer":
        return self._model(_embedder, self.name)

    def reranker(self) -> "CrossEncoder":
        return self._model(_reranker, self.reranker_name)

    def _model(self, load: Callable[[str, "torch.dtype | None"], _Model], name: str) -> _Model:
        with self._lock:
            self.accelerator.budget()  # caps the GPU memory, or falls back to the CPU, before anything loads
            dtype = self.accelerator.half_precision
            if dtype != self._dtype:  # a device of another precision: load again
                self._loaded, self._dtype = {}, dtype
            if name not in self._loaded:
                self._loaded[name] = load(name, dtype)
            return self._loaded[name]

    def _ready(self, get: Callable[[], _Model]) -> _Model:
        """The model get() returns, on the GPU while in use. The accelerator looks at the GPU first: it may have
        fallen back to the CPU, and then get() returns a model for the CPU."""
        self.accelerator.resume()
        model = get()
        self.accelerator.acquire(id(model), lambda: model.to(self.accelerator.device), lambda: model.to("cpu"))
        return model

    def load(self) -> None:
        """Load both models now, so the first request is quick."""
        self.embedder()
        self.reranker()

    def lengths(self, texts: list[str]) -> list[int]:
        """The tokens of each text, cut at the longest input the model takes, without the prompt it puts before every
        text alike, counted LENGTHS_AT_ONCE at a time. With a tokenizer of its own and no PyTorch, so it runs in any
        thread (PyTorch crashes in other threads than the main one on Python 3.15), and the model's tokenizer, which
        Sentence Transformers sets to pad each batch, is never shared. For a model without a Hugging Face fast
        tokenizer, such as a static embedding model, characters: they only group texts."""
        counter = self._counting()
        if counter is None:
            return [len(text) for text in texts]
        found: list[int] = []
        for start in range(0, len(texts), LENGTHS_AT_ONCE):
            found += [len(tokens) for tokens in counter.encode_batch(texts[start : start + LENGTHS_AT_ONCE])]
        return found

    def _counting(self) -> "Tokenizer | None":
        """The tokenizer lengths() counts with, made on first use: its first call loads the model, so it belongs on
        the models' thread (prepare does it)."""
        with self._lock:
            if isinstance(self._counter, Unset):
                model = self.embedder()
                backend = getattr(model.tokenizer, "backend_tokenizer", None)
                self._counter = None if backend is None else _counter_of(backend, _longest(model))
            return self._counter

    def prepare(self) -> None:
        """Get ready to embed: the tokenizer that counts tokens (lengths), and on CUDA, batches sized to the GPU's
        memory with texts of the longest input the model takes: a batch that fits them fits any text, whatever the
        model."""
        self._counting()
        if self.accelerator.kind != "cuda":
            return
        steps = _Steps(self._ready(self.embedder), query=False)
        longest = "x " * (4 * _longest(steps.model))  # truncated to the longest input
        self.accelerator.calibrate(lambda n: steps.collect(steps.launch(steps.tokenize([longest] * n))))

    def encode(self, texts: list[str], query: bool = False) -> NDArray[np.float16]:
        """Normalised fp16 embeddings, one row per text, in batches of the accelerator's size."""
        steps = _Steps(self._ready(self.embedder), query)
        return self.accelerator.run(lambda t, size: _stacked(_pipelined(steps, _parts(t, size))), texts)

    def encode_batches(self, batches: Iterable[list[str]], query: bool = False) -> Iterator[NDArray[np.float16]]:
        """The embeddings of each batch of texts, in order: each batch tokenized while the GPU embeds the one before.
        Iterate it on the models' thread. A batch the GPU has no memory for is embedded by encode(), which embeds it
        in parts; so is one larger than the batch size, since halved."""
        steps = _Steps(self._ready(self.embedder), query)
        return _pipelined(steps, batches, (lambda texts: self.encode(texts, query), lambda: self.batch))

    def rerank(self, query: str, texts: list[str]) -> NDArray[np.float32]:
        """Raw scores (logits) of (query, text) pairs, higher for better matches, whatever activation the model's
        configuration names, so that scores mean the same with any reranker."""
        import torch

        model, raw = self._ready(self.reranker), torch.nn.Identity()
        return self.accelerator.run(
            lambda t, size: _scores(model, [(query, x) for x in t], size, raw), texts, most=RERANK_BATCH
        )

    def release_idle(self) -> bool:
        """Move the models back to RAM if the GPU has been idle long enough. Returns whether it did."""
        return self.accelerator.release_idle()


class _Steps:
    """A batch's way through an embedding model, as Sentence Transformers' encode_document (encode_query) takes it,
    in three steps: tokenize it on the CPU, launch the forward pass (the GPU runs it while the CPU goes on), collect
    the vectors (waiting for the GPU)."""

    def __init__(self, model: _Embedder, query: bool):
        self.model, self.task = model, TASKS[query]
        name = next((name for name in PROMPTS[query] if name in model.prompts), model.default_prompt_name)
        self.prompt = model.prompts.get(name) if name else None

    def tokenize(self, texts: list[str]) -> dict[str, Any]:
        return self.model.preprocess(texts, prompt=self.prompt, task=self.task)

    def launch(self, features: dict[str, Any]) -> "torch.Tensor":
        import torch
        from sentence_transformers.util import batch_to_device

        with torch.inference_mode():
            vectors = self.model(batch_to_device(features, self.model.device), task=self.task)["sentence_embedding"]
            return torch.nn.functional.normalize(vectors, p=2, dim=1)

    @staticmethod
    def collect(launched: "torch.Tensor") -> NDArray[np.float16]:
        """Normalised fp16 rows, in an array of their own: freed in any thread, it frees no PyTorch tensor."""
        return launched.float().cpu().numpy().astype(np.float16)


def _pipelined(
    steps: _Steps,
    batches: Iterable[list[str]],
    fallback: tuple[Callable[[list[str]], NDArray[np.float16]], Callable[[], int]] | None = None,
) -> Iterator[NDArray[np.float16]]:
    """The vectors of each batch, in order, each tokenized while the GPU runs the one before. With a `fallback`,
    (again, size): a batch of more texts than size(), or one the GPU has no memory for, is embedded by again(texts)."""
    launched = None
    for texts in batches:
        if fallback is not None and len(texts) > fallback[1]():
            if launched is not None:
                yield steps.collect(launched)
                launched = None
            yield fallback[0](texts)
            continue
        features = steps.tokenize(texts)
        if launched is not None:
            yield steps.collect(launched)
            launched = None
        try:
            launched = steps.launch(features)
            continue
        except RuntimeError as error:
            if fallback is None or not accelerator.out_of_memory(error):
                raise
        # after the `except`, whose traceback held the failed batch's tensors on the GPU while it lasted
        yield fallback[0](texts)
    if launched is not None:
        yield steps.collect(launched)


def _parts(texts: list[str], size: int) -> Iterator[list[str]]:
    return (texts[start : start + size] for start in range(0, len(texts), size))


def _stacked(arrays: Iterable[NDArray[np.float16]]) -> NDArray[np.float16]:
    stacked = list(arrays)
    return np.concatenate(stacked) if stacked else np.empty((0, 0), np.float16)


def _counter_of(backend: "Tokenizer", longest: int) -> "Tokenizer":
    """A copy of a fast tokenizer's (a `tokenizers.Tokenizer`), unpadded and cut at `longest` tokens, to count with."""
    from tokenizers import Tokenizer

    counter = Tokenizer.from_str(backend.to_str())
    counter.no_padding()
    _truncate(counter, longest)
    return counter


def _truncate(tokenizer: _Truncating, longest: int) -> None:
    tokenizer.enable_truncation(longest)


def _scores(model: _Reranker, pairs: list[tuple[str, str]], size: int, activation: Any) -> NDArray[np.float32]:
    return model.predict(pairs, batch_size=size, show_progress_bar=False, activation_fn=activation)
