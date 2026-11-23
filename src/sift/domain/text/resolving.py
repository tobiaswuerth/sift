import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from sift.api.ids import DocumentId
from sift.domain.text.links import LinkKind, folded, name_key

RELATIVE = ("./", "../")
LINE_CHARS = 300
ELLIPSIS = "\N{HORIZONTAL ELLIPSIS}"


class Via(StrEnum):
    PATH = "path"
    NAME = "name"
    ALIAS = "alias"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


LEADING = frozenset({Via.PATH, Via.NAME, Via.ALIAS})


@dataclass(frozen=True, slots=True)
class Candidate:
    document: DocumentId
    path: str


@dataclass(frozen=True, slots=True)
class Resolution:
    via: Via
    documents: tuple[DocumentId, ...]


UNRESOLVED = Resolution(Via.UNRESOLVED, ())


def link_path(source: str, kind: LinkKind, target: str) -> str:
    if kind is LinkKind.MARKDOWN or target.startswith(RELATIVE):
        return posixpath.normpath(posixpath.join(posixpath.dirname(source), target))
    return target.lstrip("/")


def _one(via: Via, found: Sequence[Candidate]) -> Resolution:
    if len(found) == 1:
        return Resolution(via, (found[0].document,))
    return Resolution(Via.AMBIGUOUS, tuple(each.document for each in found))


def _by_path(path: str, named: Sequence[Candidate]) -> Resolution:
    found = [each for each in named if name_key(each.path) == name_key(path)]
    found = found or [each for each in named if name_key(each.path).endswith("/" + name_key(path))]
    return _one(Via.PATH, found) if found else UNRESOLVED


def resolved(
    source: str | None, kind: LinkKind, target: str, named: Sequence[Candidate], aliased: Sequence[Candidate]
) -> Resolution:
    relative = target.startswith(RELATIVE)
    if (relative or kind is LinkKind.MARKDOWN) and source is not None:
        at = name_key(link_path(source, kind, target))
        found = [each for each in named if name_key(each.path) == at]
        if found:
            return _one(Via.PATH, found)
    if relative:
        return UNRESOLVED
    path = target.lstrip("/")
    if "/" in path:
        return _by_path(path, named)
    if len(named) > 1:
        here = None if source is None else folded(posixpath.dirname(source))
        found = [each for each in named if folded(posixpath.dirname(each.path)) == here]
        return _one(Via.NAME, found if len(found) == 1 else named)
    if named:
        return Resolution(Via.NAME, (named[0].document,))
    return _one(Via.ALIAS, aliased) if aliased else UNRESOLVED


@dataclass(frozen=True, slots=True)
class Linking:
    line: int
    field: str | None
    target: str


def around(text: str, target: str) -> str:
    content = text.strip()
    if len(content) <= LINE_CHARS:
        return content
    start = max(0, content.find(target) - LINE_CHARS // 3)
    end = start + LINE_CHARS
    return (ELLIPSIS if start else "") + content[start:end] + (ELLIPSIS if end < len(content) else "")


def linking_lines(links: Sequence[Linking], lines: Mapping[int, str]) -> list[tuple[int, str | None, str]]:
    found: dict[int, tuple[int, str | None, str]] = {}
    for link in links:
        if link.line not in found:
            found[link.line] = (link.line, link.field, around(lines[link.line], link.target))
    return list(found.values())
