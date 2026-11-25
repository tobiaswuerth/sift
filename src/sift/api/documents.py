from dataclasses import dataclass
from typing import Annotated

from sift.api.described import FIELD_LENGTH, LISTED, PATH_LENGTH, Bounds, Composed, Help
from sift.api.errors import Problem
from sift.api.fields import FIELDS, Fields
from sift.api.requests import Request

ID = (
    "a document's id: collection/path, the path shortened to its last parts if no other document's ends so, or a "
    "file's absolute path"
)
MOST_LINE = 2**31 - 1
MOST_DOCUMENTS = 1_000


@dataclass(frozen=True, slots=True)
class WantedDocument(Request):
    id: Annotated[str, Help(ID), PATH_LENGTH]
    first_line: Annotated[int | None, Help("the first line to read (default: the first)"), Bounds(1, MOST_LINE)] = None
    last_line: Annotated[int | None, Help("the last line to read (default: the last)"), Bounds(1, MOST_LINE)] = None


@dataclass(frozen=True, slots=True)
class GetDocuments(Request):
    documents: Annotated[
        tuple[WantedDocument, ...], Help("the documents to read, each with its lines"), Bounds(1, MOST_DOCUMENTS)
    ]
    fields: Annotated[tuple[str, ...], Help(FIELDS), Composed(), LISTED, FIELD_LENGTH] = ()


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
