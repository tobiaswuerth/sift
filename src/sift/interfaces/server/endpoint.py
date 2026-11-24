import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from sift.api.wire import decoded
from sift.storage.database import PRIVATE_FILE
from sift.storage.store_files import WRITER_SUFFIX
from sift.storage.writer import held_by

ENDPOINT = "server.json"
SERVER = "server"
SERVING = "sift serve"


@dataclass(frozen=True, slots=True)
class Endpoint:
    port: int
    pid: int
    token: str
    version: str


def lock_of(home: Path) -> Path:
    return home / f"{SERVER}{WRITER_SUFFIX}"


def written(home: Path, endpoint: Endpoint) -> None:
    path = home / ENDPOINT
    part = path.with_name(f"{ENDPOINT}.{os.getpid()}.part")
    part.unlink(missing_ok=True)
    os.close(os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, PRIVATE_FILE))
    part.write_bytes(json.dumps(asdict(endpoint)).encode())
    part.replace(path)


def found(home: Path) -> Endpoint | None:
    if held_by(lock_of(home)) is None:
        return None
    try:
        read = cast("object", json.loads((home / ENDPOINT).read_bytes()))
    except FileNotFoundError:
        return None
    return decoded(Endpoint, read)
