"""sift's settings: the optional environment variables in the README's table, read once per process. None are
needed; they are for special cases, and every module asks `current()` instead of reading the environment itself.
Beside them CUDA's CUDA_VISIBLE_DEVICES, Hugging Face's HF_HUB_OFFLINE (offline, sift asks PyPI for no update
either), and where Numba keeps the code it compiles (`keep_compiled_code`), which it reads from the environment."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Self

# The models sift uses unless told otherwise, each pinned to a commit of its repository on the Hugging Face Hub: what
# its owner pushes later changes neither the vectors of new passages beside the old ones nor the code that computes
# them. A release of sift moves a pin, after a look at what changed, and a collection embedded at another commit is
# embedded again (`sift embed --reset`), as with another model.
PINNED = {
    "google/embeddinggemma-2": "f9972ac22e696a105c482202646979e33e4b5a1e",
    "Alibaba-NLP/gte-reranker-modernbert-base": "f7481e6055501a30fb19d090657df9ec1f79ab2c",
    "lightonai/LightOnOCR-2-1B": "be7c41313c30e940fe5571de59e5fabc779e00c8",
}
REPOSITORY = re.compile(r"\w[\w.-]*(/\w[\w.-]*)?")  # a repository of the Hub, "name" or "owner/name": no folder's path
REVISION = re.compile(r"[\w.-]+")  # a commit, branch or tag of the Hub


def pinned(name: str) -> str:
    """A model as sift names it, and records what it embedded with: one of PINNED's, given by its name alone, at its
    commit ("name@commit"); any other as given."""
    return f"{name}@{PINNED[name]}" if name in PINNED else name


def hub(name: str) -> tuple[str, str | None]:
    """A model's repository or folder, and the revision to load: the one a repository names after its "@", else its
    pin, else None, its latest commit. A folder is taken whole, an "@" in its path too."""
    repository, at, revision = name.rpartition("@")
    if at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision):
        return repository, revision
    return name, PINNED.get(name)


@dataclass(frozen=True)
class Settings:
    home: Path  # SIFT_HOME: where the catalog, logs and worker's key live, and the collections' stores unless moved
    embed_model: str  # SIFT_EMBED_MODEL: any Sentence Transformers embedding model, at a revision after an "@"
    rerank_model: str  # SIFT_RERANK_MODEL: any Sentence Transformers cross-encoder, at a revision after an "@"
    remote: str | None  # SIFT_EMBED_REMOTE: a `sift worker` to embed and rerank on, as it prints its address
    ocr: str = "auto"  # SIFT_OCR: where scans are read: auto (on a GPU), on (also on the CPU), off (see `ocr`)
    gpus: str | None = None  # CUDA_VISIBLE_DEVICES, CUDA's own, as CUDA reads it: "" or -1 hides every GPU
    update_check: bool = True  # SIFT_UPDATE_CHECK: off never asks PyPI for a newer sift (see `update`); nor offline

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Self:
        """The settings an environment gives. An empty variable counts as unset. The home is a whole path: "~" as the
        user's folder, which no shell expanded for an MCP client's config or `setx`, and a relative one from where the
        command runs, so that processes it starts elsewhere find the same."""
        return cls(
            home=Path(environ.get("SIFT_HOME") or Path.home() / ".sift").expanduser().absolute(),
            embed_model=pinned(environ.get("SIFT_EMBED_MODEL") or "google/embeddinggemma-2"),
            rerank_model=pinned(environ.get("SIFT_RERANK_MODEL") or "Alibaba-NLP/gte-reranker-modernbert-base"),
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
