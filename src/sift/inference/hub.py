import hashlib
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
COMMIT = re.compile(r"[0-9a-f]{40}")
STAMP_BYTES = 8
MOVING = (
    "a branch or tag moves, and a short commit may mean several, so its vectors could not be told from those of "
    "other weights"
)
COMMIT_REMEDY = "name a commit of it, e.g. {name}@<commit> from its page under Files and versions"


def _stamp(folder: Path) -> str:
    stamped = hashlib.blake2b(digest_size=STAMP_BYTES)
    for path in sorted(each for each in folder.rglob("*") if each.is_file()):
        held = path.stat()
        stamped.update(f"{path.relative_to(folder).as_posix()}\0{held.st_size}\0{held.st_mtime_ns}\0".encode())
    return stamped.hexdigest()


def _revision_refused(name: str, repository: str, revision: str) -> ModelUnavailableError:
    said = f"the model {name} names {revision}, no whole commit: {MOVING}"
    return ModelUnavailableError(said, COMMIT_REMEDY.format(name=repository))


def pinned(name: str) -> str:
    if name in PINNED:
        return f"{name}@{PINNED[name]}"
    folder = Path(name)
    if folder.is_dir():
        whole = folder.resolve()
        return f"{whole}#{_stamp(whole)}"
    repository, at, revision = name.rpartition("@")
    if at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision):
        if COMMIT.fullmatch(revision) is None:
            raise _revision_refused(name, repository, revision)
        return name
    raise ModelUnavailableError(
        f"the model {name} names no commit, so its vectors could not be told from those of its next weights",
        COMMIT_REMEDY.format(name=name.rstrip("@")),
    )


def hub(name: str) -> tuple[str, str | None]:
    repository, at, revision = name.rpartition("@")
    if at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision):
        return repository, revision
    return name, PINNED.get(name)
