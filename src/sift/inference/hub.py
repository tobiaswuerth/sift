import re
from pathlib import Path
from types import MappingProxyType

from sift.api.errors import ModelUnavailableError

EMBED_MODEL = "google/embeddinggemma-2"
RERANK_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
OCR_MODEL = "lightonai/LightOnOCR-2-1B"
PINNED = MappingProxyType(
    {
        EMBED_MODEL: "f9972ac22e696a105c482202646979e33e4b5a1e",
        RERANK_MODEL: "f7481e6055501a30fb19d090657df9ec1f79ab2c",
        OCR_MODEL: "be7c41313c30e940fe5571de59e5fabc779e00c8",
    }
)
REPOSITORY = re.compile(r"\w[\w.-]*(/\w[\w.-]*)?")
REVISION = re.compile(r"[\w.-]+")


def pinned(name: str) -> str:
    if name in PINNED:
        return f"{name}@{PINNED[name]}"
    repository, at, revision = name.rpartition("@")
    if (at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision)) or Path(name).is_dir():
        return name
    raise ModelUnavailableError(
        f"the model {name} names no revision, so its vectors could not be told from those of its next weights",
        f"name a commit of it, e.g. {name}@<commit> from its page under Files and versions",
    )


def hub(name: str) -> tuple[str, str | None]:
    repository, at, revision = name.rpartition("@")
    if at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision):
        return repository, revision
    return name, PINNED.get(name)
