import re
from types import MappingProxyType

EMBED_MODEL = "google/embeddinggemma-2"
PINNED = MappingProxyType({EMBED_MODEL: "f9972ac22e696a105c482202646979e33e4b5a1e"})
REPOSITORY = re.compile(r"\w[\w.-]*(/\w[\w.-]*)?")
REVISION = re.compile(r"[\w.-]+")


def pinned(name: str) -> str:
    return f"{name}@{PINNED[name]}" if name in PINNED else name


def hub(name: str) -> tuple[str, str | None]:
    repository, at, revision = name.rpartition("@")
    if at and REPOSITORY.fullmatch(repository) and REVISION.fullmatch(revision):
        return repository, revision
    return name, PINNED.get(name)
