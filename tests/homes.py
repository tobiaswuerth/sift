from pathlib import Path

from sift.api.collections import AddCollection
from sift.app.collections import add
from tests.files import write


def collection(tmp_path: Path, home: Path, name: str, files: dict[str, str]) -> None:
    for path, text in files.items():
        write(tmp_path / name / path, text)
    add(home, AddCollection(str(tmp_path / name)))
