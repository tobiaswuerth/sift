import posixpath
from collections.abc import Mapping
from enum import Enum, auto
from types import MappingProxyType


class FileKind(Enum):
    NOTE = auto()
    RECORDS = auto()
    DOCUMENT = auto()
    MAIL = auto()


NOTES = (".md", ".markdown", ".txt")
RECORDS = (".jsonl", ".ndjson")
DOCUMENTS = (".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".rtf", ".epub", ".html", ".htm")
MAILS = (".eml",)
KINDS: Mapping[str, FileKind] = MappingProxyType(
    {
        **dict.fromkeys(NOTES, FileKind.NOTE),
        **dict.fromkeys(RECORDS, FileKind.RECORDS),
        **dict.fromkeys(DOCUMENTS, FileKind.DOCUMENT),
        **dict.fromkeys(MAILS, FileKind.MAIL),
    }
)
STREAMED = frozenset({FileKind.RECORDS})
KEYED = frozenset({FileKind.RECORDS, FileKind.MAIL})
CONVERTED = frozenset({FileKind.DOCUMENT, FileKind.MAIL})
KEY_MARK = "#"


def kind_of(path: str) -> FileKind | None:
    return KINDS.get(posixpath.splitext(path)[1].lower())


def streamed(path: str) -> bool:
    return kind_of(path) in STREAMED


def keyed_path(path: str, key: str) -> str:
    return f"{path}{KEY_MARK}{key}" if key else path


def public_id(collection: str, path: str, key: str) -> str:
    return f"{collection}/{keyed_path(path, key)}"


def split_keyed_path(relative: str) -> tuple[str, str]:
    at = relative.find(KEY_MARK)
    while at >= 0:
        if kind_of(relative[:at]) in KEYED:
            return relative[:at], relative[at + 1 :]
        at = relative.find(KEY_MARK, at + 1)
    return relative, ""
