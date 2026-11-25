from dataclasses import dataclass
from typing import Annotated

from sift.api.described import PATH_LENGTH, Bounds, FileOrId, Help
from sift.api.embedding import EmbedResult
from sift.api.indexing import MOST_FILES, IndexResult
from sift.api.requests import Request

REINDEXED = (
    "the files to read again now, as written, changed or deleted: ids (collection/path) or absolute paths; their "
    "new passages are embedded too where their collection is"
)


@dataclass(frozen=True, slots=True)
class ReindexFiles(Request):
    files: Annotated[tuple[str, ...], Help(REINDEXED), FileOrId(), Bounds(1, MOST_FILES), PATH_LENGTH]


@dataclass(frozen=True, slots=True)
class Reindexed:
    indexed: IndexResult
    embedded: EmbedResult | None = None
