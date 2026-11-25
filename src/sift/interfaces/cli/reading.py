import argparse
import re
from dataclasses import dataclass
from typing import Annotated

from sift.api.described import Help
from sift.api.documents import GetDocuments, WantedDocument
from sift.api.errors import InvalidRequestError
from sift.api.fields import FIELDS
from sift.api.requests import Request
from sift.interfaces.cli.arguments import request_of

RANGE = re.compile(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?")
RANGE_REMEDY = "lines are a line number or a range such as 17-26"
IDS = (
    "documents by id: collection/path, the path shortened to its last parts if no other document's ends so, or a "
    "file's absolute path"
)


@dataclass(frozen=True, slots=True)
class ReadDocuments(Request):
    ids: Annotated[tuple[str, ...], Help(IDS)]
    lines: Annotated[str | None, Help('only these lines, e.g. "17-26" or "40"')] = None
    fields: Annotated[tuple[str, ...], Help(FIELDS)] = ()


def line_range(written: str) -> tuple[int, int]:
    matched = RANGE.fullmatch(written)
    first, last = (int(matched[1]), int(matched[2] or matched[1])) if matched else (0, 0)
    if not matched or last < first:
        raise InvalidRequestError(f"{written!r}: {RANGE_REMEDY}", "give a line number, or two joined by a dash")
    return first, last


def wanted(namespace: argparse.Namespace) -> GetDocuments:
    read = request_of(ReadDocuments, namespace)
    first, last = (None, None) if read.lines is None else line_range(read.lines)
    return GetDocuments(tuple(WantedDocument(each, first, last) for each in read.ids), read.fields)
