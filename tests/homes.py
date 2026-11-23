from pathlib import Path

from sift.api.collections import AddCollection, Counted, ShowStatus, UnreadableFile
from sift.api.documents import GetDocuments
from sift.app.collections import add, status
from sift.app.documents import documents
from tests.files import write


def collection(tmp_path: Path, home: Path, name: str, files: dict[str, str]) -> None:
    for path, text in files.items():
        write(tmp_path / name / path, text)
    add(home, AddCollection(str(tmp_path / name)))


def unreadable_files(home: Path) -> tuple[UnreadableFile, ...]:
    (shown,) = status(home, ShowStatus(unreadable=True)).collections
    assert isinstance(shown.contents, Counted)
    return shown.contents.unreadable_files


def text_of(home: Path, id_: str) -> str:
    (one,) = documents(home, GetDocuments((id_,))).found
    return one.text
