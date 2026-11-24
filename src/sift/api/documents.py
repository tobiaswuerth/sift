from dataclasses import dataclass
from typing import Annotated

from sift.api.described import Composed, Help
from sift.api.errors import Problem
from sift.api.fields import FIELDS, Fields
from sift.api.requests import Request

IDS = (
    "documents by id: collection/path, the path shortened to its last parts if no other document's ends so, or a "
    "file's absolute path"
)


@dataclass(frozen=True, slots=True)
class GetDocuments(Request):
    ids: Annotated[tuple[str, ...], Help(IDS)]
    lines: Annotated[str | None, Help('only these lines, e.g. "17-26" or "40"')] = None
    fields: Annotated[tuple[str, ...], Help(FIELDS), Composed()] = ()


@dataclass(frozen=True, slots=True)
class DocumentText:
    id: str
    title: str
    text: str
    fields: Fields | None = None


@dataclass(frozen=True, slots=True)
class Unfound:
    id: str
    message: str
    remedy: str | None
    code: Problem


@dataclass(frozen=True, slots=True)
class Documents:
    found: tuple[DocumentText, ...]
    unfound: tuple[Unfound, ...]
