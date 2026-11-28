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
READ_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".json", ".model", ".txt", ".tiktoken", ".py"})
MOVING = (
    "a branch or tag moves, and a short commit may mean several, so its vectors could not be told from those of "
    "other weights"
)
COMMIT_REMEDY = "name a commit of it, e.g. {name}@<commit> from its page under Files and versions"
SHADOWED = "{folder} lies here and is named like the model {model}, which it would stand in for"
SHADOWED_REMEDY = "run sift in another folder, or rename that one"
SOME_MODEL = "owner/model"


def _read_by_the_model(folder: Path, path: Path) -> bool:
    parts = path.relative_to(folder).parts
    return path.suffix in READ_SUFFIXES and not any(part.startswith(".") for part in parts) and path.is_file()


def _stamp(folder: Path) -> str:
    stamped = hashlib.blake2b(digest_size=STAMP_BYTES)
    for path in sorted(each for each in folder.rglob("*") if _read_by_the_model(folder, each)):
        held = path.stat()
        stamped.update(f"{path.relative_to(folder).as_posix()}\0{held.st_size}\0{held.st_mtime_ns}\0".encode())
    return stamped.hexdigest()


def _revision_refused(name: str, repository: str, revision: str) -> ModelUnavailableError:
    said = f"the model {name} names {revision}, no whole commit: {MOVING}"
    return ModelUnavailableError(said, COMMIT_REMEDY.format(name=repository))


def _commit(revision: str) -> str | None:
    lowered = revision.lower()
    return lowered if COMMIT.fullmatch(lowered) else None


def _unnamed(name: str, repository: str, at: str) -> ModelUnavailableError:
    if at and not repository:
        said = f"the model {name} names no model, only a commit"
        return ModelUnavailableError(said, COMMIT_REMEDY.format(name=SOME_MODEL))
    said = f"the model {name} names no commit, so its vectors could not be told from those of its next weights"
    return ModelUnavailableError(said, COMMIT_REMEDY.format(name=name.rstrip("@")))


def pinned(name: str) -> str:
    if name in PINNED:
        return f"{name}@{PINNED[name]}"
    folder = Path(name)
    if folder.is_dir():
        whole = folder.resolve()
        return f"{whole}#{_stamp(whole)}"
    repository, at, revision = name.rpartition("@")
    if at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision):
        commit = _commit(revision)
        if commit is None:
            raise _revision_refused(name, repository, revision)
        return f"{repository}@{commit}"
    raise _unnamed(name, repository, at)


def _named(name: str) -> tuple[str, str | None]:
    if name in PINNED:
        return name, PINNED[name]
    if Path(name).is_dir():
        return name, None
    repository, at, revision = name.rpartition("@")
    if at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision):
        return repository, _commit(revision) or revision
    return name, None


def hub(name: str) -> tuple[str, str | None]:
    repository, revision = _named(name)
    if revision is not None and Path(repository).exists():
        said = SHADOWED.format(folder=Path(repository).resolve(), model=repository)
        raise ModelUnavailableError(said, SHADOWED_REMEDY)
    return repository, revision
