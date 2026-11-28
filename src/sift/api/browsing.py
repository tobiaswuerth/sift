from dataclasses import dataclass
from typing import Annotated

from sift.api.described import NAME_LENGTH, Bounds, Help, Length
from sift.api.requests import Request
from sift.api.search import MOST_HITS

FOLDER_LENGTH = Length(0, 32_767)
PAGE = 30


@dataclass(frozen=True, slots=True)
class BrowseFolder(Request):
    collection: Annotated[str, Help("the collection to browse"), NAME_LENGTH]
    folder: Annotated[str, Help('a folder in it, e.g. "2024/" (default: its top)'), FOLDER_LENGTH] = ""
    k: Annotated[int, Help("how many of its documents, in order of their paths"), Bounds(1, MOST_HITS)] = PAGE


@dataclass(frozen=True, slots=True)
class ListedDocument:
    id: str
    title: str


@dataclass(frozen=True, slots=True)
class Browsed:
    collection: str
    folder: str
    folders: tuple[str, ...]
    documents: tuple[ListedDocument, ...]
    more: bool
