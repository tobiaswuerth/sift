from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from sift.api.collections import AddCollection, Counted, ShowStatus, UnreadableFile
from sift.api.documents import GetDocuments, WantedDocument
from sift.api.embedding import EmbedCollections
from sift.api.events import Advanced, Event
from sift.api.indexing import IndexCollections
from sift.api.search import SearchKeywords
from sift.app.collections import add, status
from sift.app.documents import documents
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.search import search
from sift.inference.models import Models
from tests.files import write


def quiet(_said: object) -> None:
    return None


def indexed(home: Path) -> None:
    index(home, IndexCollections(), quiet)


def home_of(tmp_path: Path, files: Mapping[str, str], *, embedded: Models | None = None) -> Path:
    home = tmp_path / "home"
    for path, text in files.items():
        write(tmp_path / path, text)
    for folder in dict.fromkeys(PurePosixPath(path).parts[0] for path in files):
        add(home, AddCollection(str(tmp_path / folder)))
    indexed(home)
    if embedded is not None:
        embed(home, EmbedCollections(), quiet, embedded)
    return home


def stopped_when_advanced(event: Event) -> None:
    if isinstance(event, Advanced):
        raise KeyboardInterrupt


def indexed_home(tmp_path: Path, folder: Path) -> Path:
    home = tmp_path / "home"
    add(home, AddCollection(str(folder)))
    indexed(home)
    return home


def found_ids(home: Path, query: str) -> list[str]:
    return [hit.id for hit in search(home, SearchKeywords(query)).hits]


def collection(tmp_path: Path, home: Path, name: str, files: dict[str, str]) -> None:
    for path, text in files.items():
        write(tmp_path / name / path, text)
    add(home, AddCollection(str(tmp_path / name)))


def unreadable_files(home: Path) -> tuple[UnreadableFile, ...]:
    (shown,) = status(home, ShowStatus(unreadable=True)).collections
    assert isinstance(shown.contents, Counted)
    return shown.contents.unreadable_files


def asked(*ids: str, lines: tuple[int | None, int | None] = (None, None), fields: tuple[str, ...] = ()) -> GetDocuments:
    return GetDocuments(tuple(WantedDocument(each, *lines) for each in ids), fields)


def text_of(home: Path, id_: str) -> str:
    (one,) = documents(home, asked(id_)).found
    return one.text


def long_parts(tmp_path: Path) -> Path:
    root = tmp_path / "parts"
    for name, word in (("alpha", "comets"), ("beta", "glaciers")):
        sections = "".join(f"## Part {n}\n\n{f'About {word} and more {word}. ' * 120}\n\n" for n in range(20))
        write(root / f"{name}.md", f"# {name}\n\n{sections}")
    return root
